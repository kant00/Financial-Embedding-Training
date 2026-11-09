"""
本地生成式文本匹配 SFT：默认 LoRA，可选全参数微调。

用途与流程位置：
    BERT 主线之外的可选实验；将句对转换为聊天指令，学习生成“相似”或“不相似”。
    对提示词和 padding 屏蔽损失，仅保留回答监督；回答 token 数取决于分词器和模板。

输入与输出：
    --model_path 为完整本地指令模型目录，需可用 chat_template；不能假定默认目录存在。
    --data_dir 传入固定协议；按类别平衡抽样，--num_train -1 仍不是全部原始训练样本。
    calibration 前 500 条用于验证，按验证 loss 选最佳模型。
    --output_dir 下输出 sft_adapter 或 sft_full_ckpt，以及 logs/train_sft.json 或 train_full_ft.json。

使用方式（在项目根目录执行，Bash）：
    python src_llm/train_sft.py --model_path models/qwen-0.5b --data_dir data/experiments/afqmc_seed42 --output_dir outputs/sft --num_train 5000 --epochs 3 --batch_size 4 --grad_accum 4 --max_length 256 --seed 42

注意事项：
    当前为 float32 训练；不承诺固定显存下限、收敛轮数或训练时间。
    先拼接后截断可能丢失回答监督，256 也不保证覆盖全部样本，需检查长度。
    --full_ft 切换全参数微调；资源成本更高，非主线必需步骤。
    依赖统一见 requirements.txt，环境、数据口径与输出归档见使用指南。

说明核对：2026-09-28；完整流程、前置条件及自定义模型路径见 USAGE_GUIDE.md。
"""

import os
import argparse
import json
import random
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

try:
    from peft import get_peft_model, LoraConfig, TaskType

    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False

ROOT = Path(__file__).parent.parent
DATA_DIR = ROOT / "data" / "afqmc"
MODEL_PATH = ROOT / "models" / "qwen-0.5b"
OUTPUT_DIR = ROOT / "outputs"

SYSTEM_PROMPT = (
    "你是一个语义匹配助手。判断两句话语义是否相同，"
    "只输出【相似】或【不相似】，不要输出其他内容。"
)

LABEL_MAP = {1: "【相似】", 0: "【不相似】"}


# ══════════════════════════════════════════════════════════════════════════════
# Dataset
# ══════════════════════════════════════════════════════════════════════════════


class SFTDataset(Dataset):
    """把二分类句对转换成带答案监督的聊天样本。

    用途：
        把二分类句对转换成带答案监督的聊天样本。

    使用场景与调用位置：
        在 src_llm/train_sft.py 的“将问句匹配转换为生成任务，训练 LoRA 或全量语言模型”流程中使用。 本文件直接调用位置：main()。

    参数：
        构造参数见本类 __init__；实例成员与样本/模型结构见注意事项。

    返回值：
        PyTorch Dataset 实例。

    注意事项：
        train_sft.main 构建 train/val 数据；使用 system+user 模板作为 prompt，标签文本和 EOS 为答案；prompt label=-100，动态 padding 由 collate_fn 完成。
    """

    def __init__(self, data, tokenizer, max_length=128):
        """保留 SFT 原始记录、tokenizer 和序列总长。

        用途：
            保留 SFT 原始记录、tokenizer 和序列总长。

        使用场景与调用位置：
            调用 SFTDataset(...) 创建实例时自动执行；在训练或评估加载模型/数据的准备阶段使用。

        参数：
            data: 已解析的记录或待保存对象，结构见返回值与注意事项。
            tokenizer: 与模型权重配套的 tokenizer；负责特殊 token、截断和 padding。
            max_length: 编码序列长度上限，含特殊 token；单句/句对语义见注意事项。

        返回值：
            None；初始化 self.data/tokenizer/max_length。

        注意事项：
            main 采样后调用；只保存引用，未预分词或校验每条标签，实际转换发生在 __getitem__。
        """
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        """返回当前 SFT 样本数。

        用途：
            返回当前 SFT 样本数。

        使用场景与调用位置：
            通过 dataset[idx] / len(dataset) 或 DataLoader 的采样和组批协议隐式调用，所属类为 SFTDataset。

        参数：
            无显式业务参数；实例方法通过 self 读取内部状态，入口函数从命令行获取配置。

        返回值：
            len(self.data) 的整数。

        注意事项：
            DataLoader 和采样器隐式调用；统计的是采样后的记录而非原始全集。
        """
        return len(self.data)

    def __getitem__(self, idx):
        """组装一个聊天 prompt 与标签答案并生成 loss mask。

        用途：
            组装一个聊天 prompt 与标签答案并生成 loss mask。

        使用场景与调用位置：
            通过 dataset[idx] / len(dataset) 或 DataLoader 的采样和组批协议隐式调用，所属类为 SFTDataset。

        参数：
            idx: 样本在当前数据集中的整数索引。

        返回值：
            input_ids、labels 两个 [L] CPU long 张量，L<=max_length。

        注意事项：
            DataLoader 按索引调用；答案使用 LABEL_MAP 并追加 EOS。先拼接再截断，超长 prompt 可能截掉全部答案，导致所有 label=-100；调用方需监控有效监督 token。
        """
        item = self.data[idx]
        label_text = LABEL_MAP[item["label"]]

        # ── Step 1：构建 prompt 文本（tokenize=False，兼容 transformers 5.x）──
        prompt_text = self.tokenizer.apply_chat_template(
            [
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": f"句子A：{item['sentence1']}\n句子B：{item['sentence2']}\n是否相似：",
                },
            ],
            tokenize=False,
            add_generation_prompt=True,
        )
        prompt_ids = self.tokenizer.encode(prompt_text, add_special_tokens=False)

        # ── Step 2：response = 标签文本 + EOS ─────────────────────────────────
        response_ids = self.tokenizer.encode(label_text, add_special_tokens=False) + [
            self.tokenizer.eos_token_id
        ]

        # ── Step 3：拼接 + 截断 + loss mask ──────────────────────────────────
        input_ids = (prompt_ids + response_ids)[: self.max_length]
        labels = ([-100] * len(prompt_ids) + response_ids)[: self.max_length]

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


def collate_fn(batch, pad_id):
    """将不同长度 SFT 样本右填充后组成 batch。

    用途：
        将不同长度 SFT 样本右填充后组成 batch。

    使用场景与调用位置：
        在 src_llm/train_sft.py 的“将问句匹配转换为生成任务，训练 LoRA 或全量语言模型”流程中使用。 本文件直接调用位置：main()。

    参数：
        batch: 单条 Dataset 样本字典组成的非空列表。
        pad_id: tokenizer 使用的 padding token ID。

    返回值：
        input_ids、labels、attention_mask 三个 [B,Lmax] CPU long 张量。

    注意事项：
        main 通过 lambda 绑定 pad_id 后作为 DataLoader.collate_fn；input 使用 pad_id，labels 使用 -100，mask 的 padding 区域为 0；batch 必须非空。
    """
    max_len = max(item["input_ids"].size(0) for item in batch)
    input_ids_list, labels_list, mask_list = [], [], []
    for item in batch:
        n = item["input_ids"].size(0)
        pad = max_len - n
        input_ids_list.append(
            torch.cat([item["input_ids"], torch.full((pad,), pad_id, dtype=torch.long)])
        )
        labels_list.append(
            torch.cat([item["labels"], torch.full((pad,), -100, dtype=torch.long)])
        )
        mask_list.append(
            torch.cat(
                [torch.ones(n, dtype=torch.long), torch.zeros(pad, dtype=torch.long)]
            )
        )
    return {
        "input_ids": torch.stack(input_ids_list),
        "labels": torch.stack(labels_list),
        "attention_mask": torch.stack(mask_list),
    }


# ══════════════════════════════════════════════════════════════════════════════
# Training
# ══════════════════════════════════════════════════════════════════════════════


def parse_args():
    """解析将问句匹配转换为生成任务，训练 LoRA 或全量语言模型所需的命令行配置。

    用途：
        解析将问句匹配转换为生成任务，训练 LoRA 或全量语言模型所需的命令行配置。

    使用场景与调用位置：
        本文件 main() 的启动配置阶段调用；独立运行脚本时首先经过此处。 本文件直接调用位置：main()。

    参数：
        无显式参数；所需配置由内部逻辑、模块常量或命令行提供。

    返回值：
        argparse.Namespace；成员名对应下方 add_argument 定义。

    注意事项：
        由本文件 main 在任何主要工作开始前调用；读取 sys.argv，处理默认值、类型和 choices；--help 或不合法参数通过 argparse 触发 SystemExit，不启动训练或推理。
    """
    parser = argparse.ArgumentParser(
        description="LLM SFT 文本匹配训练（LoRA / 全量微调）"
    )
    parser.add_argument("--model_path", default=str(MODEL_PATH))
    parser.add_argument("--data_dir", default=str(DATA_DIR))
    parser.add_argument("--output_dir", default=str(OUTPUT_DIR))
    parser.add_argument(
        "--num_train", default=5000, type=int, help="训练样本数，-1 使用全部 34K 条"
    )
    parser.add_argument("--epochs", default=3, type=int)
    parser.add_argument("--batch_size", default=4, type=int)
    parser.add_argument("--grad_accum", default=4, type=int)
    parser.add_argument(
        "--lr",
        default=None,
        type=float,
        help="学习率；默认 LoRA=2e-4，全量=2e-5（自动判断）",
    )
    parser.add_argument("--max_length", default=128, type=int)
    # 全量微调开关
    parser.add_argument(
        "--full_ft",
        action="store_true",
        help="全量微调：跳过 LoRA，更新所有 495M 参数（需显存 ≥ 16GB）",
    )
    # LoRA 超参（full_ft 时忽略）
    parser.add_argument("--lora_r", default=8, type=int)
    parser.add_argument("--lora_alpha", default=16, type=int)
    parser.add_argument("--seed", default=42, type=int)
    return parser.parse_args()


def main():
    """组织本脚本的完整流程：将问句匹配转换为生成任务，训练 LoRA 或全量语言模型。

    用途：
        组织本脚本的完整流程：将问句匹配转换为生成任务，训练 LoRA 或全量语言模型。

    使用场景与调用位置：
        直接运行此文件时由 if __name__ == "__main__" 入口调用；也可显式调用 main()，此时仍读取进程命令行。

    参数：
        无显式参数；所需配置由内部逻辑、模块常量或命令行提供。

    返回值：
        None；产物及副作用见注意事项，配置由命令行读取。

    注意事项：
        正负等量抽样，即使 num_train=-1 也会截到少数类数量；使用 validation 前 500 条按 loss 选模。新协议须显式传入数据目录；写 adapter/全量权重、tokenizer 与日志。
    """
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.lr is None:
        args.lr = 2e-5 if args.full_ft else 2e-4

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    ckpt_dir = output_dir / ("sft_full_ckpt" if args.full_ft else "sft_adapter")
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mode_str = "全量微调" if args.full_ft else "LoRA 微调"
    print(f"使用设备: {device}  |  微调模式: {mode_str}")

    # ── 加载数据（JSONL 格式）────────────────────────────────────────────────
    def load_jsonl(path):
        """读取 UTF-8 JSONL 并按文件顺序还原记录。

        用途：
            读取 UTF-8 JSONL 并按文件顺序还原记录。

        使用场景与调用位置：
            在 src_llm/train_sft.py 的“将问句匹配转换为生成任务，训练 LoRA 或全量语言模型”流程中使用。 所属作用域：main。

        参数：
            path: 待读取或写入的文件路径，具体模式见用途与注意事项。

        返回值：
            list[dict]；每个非空行对应一条记录。

        注意事项：
            用于将问句匹配转换为生成任务，训练 LoRA 或全量语言模型阶段的数据读取；忽略空行，不执行字段、标签或重复样本校验；缺失文件和非法 JSON 异常直接上抛，整个文件驻留内存。
        """
        with open(path, encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]

    train_raw = load_jsonl(data_dir / "train.jsonl")
    val_raw = load_jsonl(data_dir / "validation.jsonl")

    # ── 正负平衡采样（AFQMC 正负比 31:69，不平衡会导致模型全预测负例）──────
    pos = [r for r in train_raw if r["label"] == 1]
    neg = [r for r in train_raw if r["label"] == 0]
    if args.num_train > 0:
        n_each = min(args.num_train // 2, len(pos), len(neg))
    else:
        n_each = min(len(pos), len(neg))  # 全量时取较少类的数量
    train_raw = random.sample(pos, n_each) + random.sample(neg, n_each)
    random.shuffle(train_raw)
    print(
        f"训练集: {len(train_raw)} 条（正{n_each}+负{n_each}，平衡采样）| 验证集（前500条）: 500 条"
    )

    # ── 加载 Tokenizer ─────────────────────────────────────────────────────────
    print(f"\n加载 tokenizer: {args.model_path}")
    tokenizer = AutoTokenizer.from_pretrained(
        str(Path(args.model_path).resolve()), trust_remote_code=True
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # ── 构建数据集 ─────────────────────────────────────────────────────────────
    train_dataset = SFTDataset(train_raw, tokenizer, args.max_length)
    val_dataset = SFTDataset(val_raw[:500], tokenizer, args.max_length)

    # DataLoader 只向回调传入样本列表 b；此闭包补入 tokenizer 的 pad ID，
    # 返回 collate_fn 生成的三种 [B,L] 张量，供训练和验证共用。
    _collate = lambda b: collate_fn(b, tokenizer.pad_token_id)
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=_collate
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size * 2, shuffle=False, collate_fn=_collate
    )

    # ── 加载模型 ───────────────────────────────────────────────────────────────
    print(f"加载 base model: {args.model_path}")
    model = AutoModelForCausalLM.from_pretrained(
        str(Path(args.model_path).resolve()),
        dtype=torch.float32,  # transformers 5.x 用 dtype= 不用 torch_dtype=
        trust_remote_code=True,
    )

    # ── LoRA 或全量微调 ────────────────────────────────────────────────────────
    if args.full_ft:
        total = sum(p.numel() for p in model.parameters())
        print(
            f"trainable params: {total:,} || all params: {total:,} || trainable%: 100.0000"
        )
    else:
        if not PEFT_AVAILABLE:
            raise ImportError("LoRA 模式需要 peft 库：pip install peft>=0.14.0")
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            lora_dropout=0.05,
            bias="none",
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

    model = model.to(device)

    # ── 优化器 ────────────────────────────────────────────────────────────────
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    total_steps = len(train_loader) * args.epochs // args.grad_accum
    print(
        f"总训练步数: {total_steps}（batch={args.batch_size}, "
        f"grad_accum={args.grad_accum}, epochs={args.epochs}, lr={args.lr}）\n"
    )

    # ── 训练循环 ──────────────────────────────────────────────────────────────
    best_val_loss = float("inf")
    log_records = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss, total_tokens = 0.0, 0
        optimizer.zero_grad()
        t0 = time.time()

        pbar = tqdm(
            train_loader, desc=f"Epoch {epoch}/{args.epochs} [Train]", leave=False
        )
        for step, batch in enumerate(pbar):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(
                input_ids=input_ids, attention_mask=attention_mask, labels=labels
            )
            loss = outputs.loss

            (loss / args.grad_accum).backward()
            should_step = (step + 1) % args.grad_accum == 0 or (step + 1) == len(
                train_loader
            )
            if should_step:
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()

            n_tokens = (labels != -100).sum().item()
            total_loss += loss.item() * n_tokens
            total_tokens += n_tokens
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        avg_train_loss = total_loss / max(total_tokens, 1)

        # ── 验证 loss ─────────────────────────────────────────────────────────
        model.eval()
        val_loss, val_tokens = 0.0, 0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Val", leave=False):
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)
                outputs = model(
                    input_ids=input_ids, attention_mask=attention_mask, labels=labels
                )
                n_tokens = (labels != -100).sum().item()
                val_loss += outputs.loss.item() * n_tokens
                val_tokens += n_tokens
        avg_val_loss = val_loss / max(val_tokens, 1)

        elapsed = time.time() - t0
        print(
            f"Epoch {epoch}/{args.epochs} | "
            f"train_loss={avg_train_loss:.4f}  val_loss={avg_val_loss:.4f} | {elapsed:.0f}s"
        )

        log_records.append(
            {
                "epoch": epoch,
                "train_loss": avg_train_loss,
                "val_loss": avg_val_loss,
                "elapsed_s": elapsed,
            }
        )

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            model.save_pretrained(ckpt_dir)
            tokenizer.save_pretrained(ckpt_dir)
            ckpt_label = "完整模型" if args.full_ft else "LoRA adapter"
            print(
                f"  ✓ 最优{ckpt_label}已保存 → {ckpt_dir}  (val_loss={avg_val_loss:.4f})"
            )

    # ── 保存训练日志 ──────────────────────────────────────────────────────────
    log_tag = "full_ft" if args.full_ft else "sft"
    log_dir = output_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"train_{log_tag}.json"
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(log_records, f, ensure_ascii=False, indent=2)

    ckpt_label = "完整模型" if args.full_ft else "LoRA adapter"
    print(f"\n训练完成。最优 val_loss={best_val_loss:.4f}")
    print(f"训练日志 → {log_path}")
    print(f"{ckpt_label} → {ckpt_dir}")
    print(f"\n下一步：python evaluate_sft.py 查看 Accuracy / F1 与多方对比")


if __name__ == "__main__":
    main()
