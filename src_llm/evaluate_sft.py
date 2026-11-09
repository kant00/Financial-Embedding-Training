"""
SFT 评估入口：加载 LoRA/全参数模型并对固定句对生成分类回答。

用途与流程位置：
    SFT 方案冻结后执行；生成文本经解析得到 0/1 标签，再计算统一分类指标。
    ckpt_dir 含 adapter_config.json 时按 LoRA 加载，否则按完整模型加载。

输入与输出：
    --manifest 默认指向新协议 final-test manifest；必须检查它存在。
    若 manifest 缺失会回退 data_dir/validation.jsonl，避免在不知情时改变评估集合。
    --num_samples -1 表示全量，--demo 仅抽 5 条；子集不能与全量 BERT 直接排名。
    输出固定为 outputs/logs/sft_results.json 和 sft_predictions.jsonl，没有 output_dir 参数。

使用方式（在项目根目录执行，Bash）：
    python src_llm/evaluate_sft.py --model_path models/qwen-0.5b --ckpt_dir outputs/sft/sft_adapter --data_dir data/experiments/afqmc_seed42 --manifest data/experiments/afqmc_seed42/evaluation_manifest.jsonl --num_samples -1 --seed 42

注意事项：
    主指标将解析失败按错误计入，同时保存失败率、valid_only 诊断和 bootstrap 区间。
    失败时用真实标签反转仅是计分约定，不能作为线上预测回退逻辑。
    不自动读取 BERT/API 日志做横向比较；需统一样本、协议和失败处理。
    输出会覆盖，运行后及时复制到本次实验目录；设备以实际环境为准。

说明核对：2026-09-28；完整流程、前置条件及自定义模型路径见 USAGE_GUIDE.md。
"""

import os
import argparse
import json
import random
import time
from pathlib import Path

import torch
from sklearn.metrics import accuracy_score, f1_score, classification_report

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

try:
    from peft import PeftModel

    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False

from transformers import AutoTokenizer, AutoModelForCausalLM

# src_llm 与 src 分目录，显式加入公共指标模块路径。
import sys

SRC_DIR = Path(__file__).parent.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
from experiment_utils import environment_snapshot, save_json, write_jsonl
from metrics import bootstrap_confidence_intervals, compute_binary_metrics

ROOT = Path(__file__).parent.parent
DATA_DIR = ROOT / "data" / "afqmc"
MODEL_PATH = ROOT / "models" / "qwen-0.5b"
ADAPTER_DIR = ROOT / "outputs" / "sft_adapter"
LOG_DIR = ROOT / "outputs" / "logs"
DEFAULT_MANIFEST = (
    ROOT / "data" / "experiments" / "afqmc_seed42" / "evaluation_manifest.jsonl"
)

SYSTEM_PROMPT = (
    "你是一个语义匹配助手。判断两句话语义是否相同，"
    "只输出【相似】或【不相似】，不要输出其他内容。"
)


# ══════════════════════════════════════════════════════════════════════════════
# 模型加载
# ══════════════════════════════════════════════════════════════════════════════


def load_model(model_path: str, ckpt_dir: str, device: torch.device):
    """识别并加载 LoRA adapter 或完整生成模型。

    用途：
        识别并加载 LoRA adapter 或完整生成模型。

    使用场景与调用位置：
        在 src_llm/evaluate_sft.py 的“加载 SFT 模型生成分类文本并计算保守评估指标”流程中使用。 本文件直接调用位置：main()。

    参数：
        model_path: 本地生成模型或基座权重目录。
        ckpt_dir: LoRA adapter 或全量微调权重目录。
        device: torch.device 或兼容设备描述，指定推理/训练位置。

    返回值：
        (model, tokenizer)；model 已切换 eval 模式。

    注意事项：
        main 评估前调用；adapter_config.json 决定 LoRA 分支，加载 base 后 merge_and_unload。CUDA 用 float16/device_map=auto，CPU 用 float32；LoRA 需 peft，本地模型配置与权重须匹配。
    """
    ckpt_path = Path(ckpt_dir)
    is_lora = (ckpt_path / "adapter_config.json").exists()

    if is_lora:
        if not PEFT_AVAILABLE:
            raise ImportError(
                "加载 LoRA adapter 需要 peft 库：pip install peft>=0.14.0"
            )
        print(f"检测到 LoRA adapter，加载 base model: {model_path}")
        tokenizer = AutoTokenizer.from_pretrained(
            str(Path(model_path).resolve()), trust_remote_code=True
        )
        base_model = AutoModelForCausalLM.from_pretrained(
            str(Path(model_path).resolve()),
            dtype=torch.float16 if device.type == "cuda" else torch.float32,
            device_map="auto" if device.type == "cuda" else None,
            trust_remote_code=True,
        )
        print(f"加载 LoRA adapter: {ckpt_dir}")
        model = PeftModel.from_pretrained(base_model, str(ckpt_path))
        model = model.merge_and_unload()
    else:
        print(f"检测到全量微调 checkpoint，直接加载: {ckpt_dir}")
        tokenizer = AutoTokenizer.from_pretrained(
            str(ckpt_path), trust_remote_code=True
        )
        model = AutoModelForCausalLM.from_pretrained(
            str(ckpt_path),
            dtype=torch.float16 if device.type == "cuda" else torch.float32,
            device_map="auto" if device.type == "cuda" else None,
            trust_remote_code=True,
        )

    if device.type != "cuda":
        model = model.to(device)
    model.eval()
    ckpt_type = "LoRA adapter 已合并" if is_lora else "全量微调模型"
    print(f"模型加载完成（{ckpt_type}）\n")
    return model, tokenizer


# ══════════════════════════════════════════════════════════════════════════════
# 推理与解析
# ══════════════════════════════════════════════════════════════════════════════


def classify_pair(
    s1: str, s2: str, model, tokenizer, device: torch.device, max_new_tokens: int = 8
) -> str:
    """将一对文本组成聊天请求并进行确定性生成。

    用途：
        将一对文本组成聊天请求并进行确定性生成。

    使用场景与调用位置：
        在 src_llm/evaluate_sft.py 的“加载 SFT 模型生成分类文本并计算保守评估指标”流程中使用。 本文件直接调用位置：main()。

    参数：
        s1: 句子 A 的文本。
        s2: 句子 B 的文本。
        model: 当前操作的模型实例；结构和设备须匹配输入。
        tokenizer: 与模型权重配套的 tokenizer；负责特殊 token、截断和 padding。
        device: torch.device 或兼容设备描述，指定推理/训练位置。
        max_new_tokens: 最多生成的新 token 数，不含 prompt。

    返回值：
        只含新增 token 解码内容的字符串。

    注意事项：
        main 逐样本评估时调用；do_sample=False，最多生成 max_new_tokens；不会在此解析成类别，也不计算正确率。输入张量设备应与加载模型兼容。
    """
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"句子A：{s1}\n句子B：{s2}\n是否相似："},
    ]
    encoding = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
    )
    input_ids = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)
    prompt_len = input_ids.shape[-1]

    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    new_tokens = output_ids[0][prompt_len:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def parse_prediction(raw_output: str) -> int:
    """把生成文本解析成相似/不相似标签。

    用途：
        把生成文本解析成相似/不相似标签。

    使用场景与调用位置：
        在 src_llm/evaluate_sft.py 的“加载 SFT 模型生成分类文本并计算保守评估指标”流程中使用。 本文件直接调用位置：main()。

    参数：
        raw_output: 模型生成的原始答案文本。

    返回值：
        0=含不相似，1=含相似，-1=无法识别。

    注意事项：
        main 在 classify_pair 后调用；先检查不相似以避免子串误判。采用包含匹配而非严格格式校验，同时含两种表述时优先返回 0。
    """
    if "不相似" in raw_output:
        return 0
    if "相似" in raw_output:
        return 1
    return -1


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════


def parse_args():
    """解析加载 SFT 模型生成分类文本并计算保守评估指标所需的命令行配置。

    用途：
        解析加载 SFT 模型生成分类文本并计算保守评估指标所需的命令行配置。

    使用场景与调用位置：
        本文件 main() 的启动配置阶段调用；独立运行脚本时首先经过此处。 本文件直接调用位置：main()。

    参数：
        无显式参数；所需配置由内部逻辑、模块常量或命令行提供。

    返回值：
        argparse.Namespace；成员名对应下方 add_argument 定义。

    注意事项：
        由本文件 main 在任何主要工作开始前调用；读取 sys.argv，处理默认值、类型和 choices；--help 或不合法参数通过 argparse 触发 SystemExit，不启动训练或推理。
    """
    parser = argparse.ArgumentParser(description="LLM SFT 文本匹配评估")
    parser.add_argument("--model_path", default=str(MODEL_PATH))
    parser.add_argument(
        "--ckpt_dir",
        default=str(ADAPTER_DIR),
        help="checkpoint 目录；含 adapter_config.json → LoRA，否则 → 全量",
    )
    parser.add_argument("--data_dir", default=str(DATA_DIR))
    parser.add_argument(
        "--manifest",
        default=str(DEFAULT_MANIFEST),
        help="所有模型共用的固定最终评估清单",
    )
    parser.add_argument(
        "--num_samples",
        default=-1,
        type=int,
        help="-1 使用完整清单；正数仅用于成本受限的显式子集实验",
    )
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--demo", action="store_true", help="只跑 5 条示例，快速演示")
    return parser.parse_args()


def main():
    """组织本脚本的完整流程：加载 SFT 模型生成分类文本并计算保守评估指标。

    用途：
        组织本脚本的完整流程：加载 SFT 模型生成分类文本并计算保守评估指标。

    使用场景与调用位置：
        直接运行此文件时由 if __name__ == "__main__" 入口调用；也可显式调用 main()，此时仍读取进程命令行。

    参数：
        无显式参数；所需配置由内部逻辑、模块常量或命令行提供。

    返回值：
        None；产物及副作用见注意事项，配置由命令行读取。

    注意事项：
        优先使用 manifest，缺失时回退 data_dir/validation.jsonl；默认全集，demo/num_samples 可抽样。解析失败被替换为真实标签反类仅用于保守计分，不是实际模型输出；原始 pred=-1 保留。
    """
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    ckpt_dir = Path(args.ckpt_dir)
    if not ckpt_dir.exists():
        print(f"[错误] checkpoint 目录不存在：{ckpt_dir}")
        print("请先运行 train_sft.py 完成训练。")
        print("  LoRA（默认）保存到:   outputs/sft_adapter/")
        print("  全量微调保存到:        outputs/sft_full_ckpt/")
        return

    # ── 加载数据 ──────────────────────────────────────────────────────────────
    manifest_path = Path(args.manifest)
    data_path = (
        manifest_path
        if manifest_path.exists()
        else Path(args.data_dir) / "validation.jsonl"
    )
    with open(data_path, encoding="utf-8") as file_obj:
        val_data = [json.loads(line) for line in file_obj if line.strip()]

    random.seed(args.seed)
    n = 5 if args.demo else args.num_samples
    samples = val_data if n < 0 else random.sample(val_data, min(n, len(val_data)))
    print(f"评估样本数: {len(samples)}\n")

    # ── 加载模型 ──────────────────────────────────────────────────────────────
    model, tokenizer = load_model(args.model_path, str(ckpt_dir), device)

    # ── 推理 ──────────────────────────────────────────────────────────────────
    gold_labels, pred_labels = [], []
    parse_fail = 0
    detail_records = []
    t0 = time.time()

    for i, item in enumerate(samples, 1):
        s1, s2 = item["sentence1"], item["sentence2"]
        gold = item["label"]
        raw = classify_pair(s1, s2, model, tokenizer, device)
        pred = parse_prediction(raw)

        if pred == -1:
            parse_fail += 1
        else:
            gold_labels.append(gold)
            pred_labels.append(pred)

        detail_records.append(
            {
                "sentence1": s1,
                "sentence2": s2,
                "label": gold,
                "pred": pred,
                "raw_output": raw,
            }
        )

        gold_str = "相似" if gold == 1 else "不相似"
        pred_str = "相似" if pred == 1 else ("不相似" if pred == 0 else "解析失败")
        status = "✓" if pred == gold else ("?" if pred == -1 else "✗")
        print(
            f"[{i:3d}/{len(samples)}] {status}  真实:{gold_str}  预测:{pred_str}"
            f"  |  {s1[:20]} / {s2[:20]}"
        )

    elapsed = time.time() - t0

    # 有效预测指标用于诊断生成格式；主指标将解析失败按错误处理，避免乐观偏差。
    valid_metrics = (
        compute_binary_metrics(gold_labels, pred_labels) if gold_labels else None
    )
    all_gold = [int(record["label"]) for record in detail_records]
    conservative_predictions = [
        (
            int(record["pred"])
            if int(record["pred"]) in (0, 1)
            else 1 - int(record["label"])
        )
        for record in detail_records
    ]
    metrics = compute_binary_metrics(all_gold, conservative_predictions)
    metrics["parse_fail"] = parse_fail
    metrics["parse_fail_rate"] = parse_fail / max(len(samples), 1)
    metrics["valid_only"] = valid_metrics
    metrics["confidence_intervals"] = bootstrap_confidence_intervals(
        all_gold, conservative_predictions, seed=args.seed
    )

    print(f"\n{'='*65}")
    print(f"LLM SFT 文本匹配评估结果")
    print(f"{'='*65}")
    print(
        f"  样本数      : {len(samples)}（有效: {len(gold_labels)}，parse_fail: {parse_fail}）"
    )
    print(f"  Accuracy      : {metrics['accuracy']:.4f}")
    print(f"  F1 (weighted): {metrics['f1_weighted']:.4f}")
    print(f"  F1 (macro)   : {metrics['f1_macro']:.4f}")
    print(f"  F1 (正例)    : {metrics['f1_pos']:.4f}")
    print(f"  均值耗时     : {elapsed/len(samples):.2f}s/条（GPU）")

    print("  注：解析失败已按错误计入上述主指标；valid_only 指标仅保存在日志供诊断。")

    # ── 保存结果 ──────────────────────────────────────────────────────────────
    out_path = LOG_DIR / "sft_results.json"
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    save_json(
        {
            **metrics,
            "manifest": str(data_path.resolve()),
            "environment": environment_snapshot(),
        },
        out_path,
    )
    write_jsonl(detail_records, LOG_DIR / "sft_predictions.jsonl")
    print(f"结果已保存 → {out_path}")


if __name__ == "__main__":
    main()
