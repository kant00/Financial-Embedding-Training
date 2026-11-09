"""
Cross-Encoder 训练入口：联合编码句对并进行二分类。

用途与流程位置：
    在固定 train/calibration 协议上训练交互模型，供句对分类或候选精排使用。
    输入为 [CLS] s1 [SEP] s2 [SEP]；跨句注意力允许两句交互。
    模型输出两类 logits，训练使用交叉熵；推理取 argmax，不在本脚本搜索分类阈值。

输入与输出：
    --data_dir 指向协议目录，validation 对应 calibration；默认仍是旧 data/afqmc。
    --class_weighting 默认 balanced；分类头学习率由 --head_lr_mult 控制。
    按 calibration 正例 F1 选最佳模型，输出 checkpoints/crossencoder_best.pt。
    logs/crossencoder_log.json 保存配置、环境与逐轮指标。

使用方式（在项目根目录执行，Bash）：
    python src/train_crossencoder.py --data_dir data/experiments/afqmc_seed42 --output_dir outputs/dev/crossencoder_seed42

注意事项：
    不能像双塔一样预计算独立句向量用于直接向量召回，但可对召回候选精排。
    当前独立评估与错误分析固定句对长度 128；修改训练长度需同步评估实现。
    不保证其精度一定优于双塔；需在统一协议上实测。

说明核对：2026-09-28；完整流程、前置条件及自定义模型路径见 USAGE_GUIDE.md。
"""

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import AdamW
from tqdm import tqdm
from transformers import BertTokenizer, get_linear_schedule_with_warmup

from dataset import build_crossencoder_loaders
from evaluate import eval_crossencoder
from experiment_utils import DEFAULT_BERT_MODEL, environment_snapshot, save_json, set_global_seed
from model import build_crossencoder

# ── 默认路径 ──────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent
DATA_DIR = ROOT / "data" / "afqmc"
OUTPUT_DIR = ROOT / "outputs"
CKPT_DIR = OUTPUT_DIR / "checkpoints"
LOG_DIR = OUTPUT_DIR / "logs"


# ── 训练一个 epoch ────────────────────────────────────────────────────────


def train_one_epoch(
    model,
    loader,
    optimizer,
    scheduler,
    criterion,
    device,
    epoch,
    total_epochs,
    grad_accum,
):
    """执行一个 epoch 的 交叉熵 训练。

    用途：
        执行一个 epoch 的 交叉熵 训练。

    使用场景与调用位置：
        在 src/train_crossencoder.py 的“训练交叉编码器并按验证正例 F1 保存最佳模型”流程中使用。 本文件直接调用位置：main()。

    参数：
        model: 当前操作的模型实例；结构和设备须匹配输入。
        loader: 提供 batch 字典的 DataLoader；评估时应保持样本顺序。
        optimizer: 维护可训练参数及优化状态的优化器，将被原地更新。
        scheduler: 随 optimizer.step 更新的学习率调度器。
        criterion: 可调用的分类损失，可能包含类别权重。
        device: torch.device 或兼容设备描述，指定推理/训练位置。
        epoch: 当前 epoch 序号，用于进度显示。
        total_epochs: 总 epoch 数，用于进度显示。
        grad_accum: 每次参数更新累积的 batch 数，必须是正整数。

    返回值：
        (平均训练 loss, 训练 accuracy)。

    注意事项：
        criterion 由 main 构造，可能包含类别权重；train_acc 采用 logits.argmax，与 loss 的类别加权口径不同。 调用 model.train，反向传播并更新 optimizer/scheduler；梯度范数裁剪为 1，末尾不足 grad_accum 也更新。尾组 loss 仍除以完整 grad_accum，实际梯度尺度需注意；loader 需非空。
    """
    model.train()
    total_loss, total_correct, total_samples = 0.0, 0, 0
    optimizer.zero_grad()

    pbar = tqdm(
        loader, desc=f"Epoch {epoch}/{total_epochs} [CrossEncoder]", leave=False
    )
    for step, batch in enumerate(pbar):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        token_type_ids = batch["token_type_ids"].to(device)
        labels = batch["label"].to(device)

        logits = model(input_ids, attention_mask, token_type_ids)
        loss = criterion(logits, labels)

        (loss / grad_accum).backward()
        should_step = (step + 1) % grad_accum == 0 or (step + 1) == len(loader)
        if should_step:
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        preds = logits.argmax(dim=-1)
        total_loss += loss.item() * labels.size(0)
        total_correct += (preds == labels).sum().item()
        total_samples += labels.size(0)
        pbar.set_postfix(
            loss=f"{total_loss / total_samples:.4f}",
            acc=f"{total_correct / total_samples:.4f}",
        )

    return total_loss / total_samples, total_correct / total_samples


# ── 主训练流程 ────────────────────────────────────────────────────────────


def main():
    """组织本脚本的完整流程：训练交叉编码器并按验证正例 F1 保存最佳模型。

    用途：
        组织本脚本的完整流程：训练交叉编码器并按验证正例 F1 保存最佳模型。

    使用场景与调用位置：
        直接运行此文件时由 if __name__ == "__main__" 入口调用；也可显式调用 main()，此时仍读取进程命令行。

    参数：
        无显式参数；所需配置由内部逻辑、模块常量或命令行提供。

    返回值：
        None；产物及副作用见注意事项，配置由命令行读取。

    注意事项：
        默认使用 balanced 交叉熵；用验证正例 F1 选模并保存环境。默认 data_dir 仍是原始 AFQMC，需显式指定新协议目录；全部 epoch F1 为 0 时不会产生最佳 checkpoint。
    """
    args = parse_args()

    set_global_seed(args.seed, deterministic=not args.allow_nondeterministic)

    output_dir = Path(args.output_dir)
    ckpt_dir = output_dir / "checkpoints"
    log_dir = output_dir / "logs"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"设备: {device}")
    print(
        f"BERT 层数: {args.num_hidden_layers}  Epochs: {args.epochs}  "
        f"Batch size: {args.batch_size}"
    )

    # ── Tokenizer & DataLoader ────────────────────────────────────────────
    tokenizer = BertTokenizer.from_pretrained(args.bert_path)
    print("\nDataLoader 构建中...")
    train_loader, val_loader, _ = build_crossencoder_loaders(
        args.data_dir,
        tokenizer,
        max_length=args.max_length,
        batch_size=args.batch_size,
    )

    # ── 模型 ─────────────────────────────────────────────────────────────
    print("\n构建模型...")
    model = build_crossencoder(
        bert_path=args.bert_path,
        num_hidden_layers=args.num_hidden_layers,
    ).to(device)

    # ── 分层学习率 ────────────────────────────────────────────────────────
    bert_params = list(model.bert.parameters())
    head_params = list(model.dropout.parameters()) + list(model.classifier.parameters())

    optimizer = AdamW(
        [
            {"params": bert_params, "lr": args.lr},
            {"params": head_params, "lr": args.lr * args.head_lr_mult},
        ],
        weight_decay=0.01,
    )

    total_steps = (
        (len(train_loader) + args.grad_accum - 1) // args.grad_accum
    ) * args.epochs
    warmup_steps = int(total_steps * args.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )
    print(f"总训练步数: {total_steps}  Warmup 步数: {warmup_steps}")

    if args.class_weighting == "balanced":
        labels = [int(row["label"]) for row in train_loader.dataset.rows]
        counts = torch.bincount(torch.tensor(labels), minlength=2).float()
        class_weights = len(labels) / (2.0 * counts.clamp_min(1.0))
        criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
        print(f"类别权重: neg={class_weights[0]:.4f}, pos={class_weights[1]:.4f}")
    else:
        criterion = nn.CrossEntropyLoss()

    # ── 训练循环 ──────────────────────────────────────────────────────────
    ckpt_path = ckpt_dir / "crossencoder_best.pt"
    best_val_f1 = 0.0
    log_records = []

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        train_loss, train_acc = train_one_epoch(
            model,
            train_loader,
            optimizer,
            scheduler,
            criterion,
            device,
            epoch,
            args.epochs,
            args.grad_accum,
        )

        val_metrics = eval_crossencoder(model, val_loader, device)
        elapsed = time.time() - t0

        val_acc = val_metrics["accuracy"]
        # 与 BiEncoder 使用同一正例 F1 选模口径。
        val_f1 = val_metrics["f1_pos"]
        print(
            f"Epoch {epoch}/{args.epochs} | "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
            f"val_acc={val_acc:.4f} val_f1={val_f1:.4f} | "
            f"{elapsed:.0f}s"
        )

        log_records.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_acc": val_acc,
                "val_f1": val_f1,
                "val_f1_pos": val_metrics["f1_pos"],
                "val_f1_macro": val_metrics["f1_macro"],
                "val_f1_weighted": val_metrics["f1_weighted"],
                "elapsed_s": elapsed,
            }
        )

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save(
                {
                    "epoch": epoch,
                    "state_dict": model.state_dict(),
                    "val_acc": val_acc,
                    "val_f1": val_f1,
                    "val_metrics": val_metrics,
                    "args": vars(args),
                    "environment": environment_snapshot(),
                },
                ckpt_path,
            )
            print(f"  ✓ 新最优模型已保存 → {ckpt_path}  (val_f1={val_f1:.4f})")

    # ── 训练完成，保存日志 ────────────────────────────────────────────────
    log_path = log_dir / "crossencoder_log.json"
    save_json(
        {
            "config": vars(args),
            "environment": environment_snapshot(),
            "epochs": log_records,
        },
        log_path,
    )
    print(f"\n训练完成。最优 val_f1={best_val_f1:.4f}")
    print(f"训练日志 → {log_path}")
    print(f"最优 checkpoint → {ckpt_path}")
    print(
        f"\n运行评估：python evaluate.py --model_type crossencoder --ckpt {ckpt_path}"
    )


# ── 参数解析 ──────────────────────────────────────────────────────────────


def parse_args():
    """解析训练交叉编码器并按验证正例 F1 保存最佳模型所需的命令行配置。

    用途：
        解析训练交叉编码器并按验证正例 F1 保存最佳模型所需的命令行配置。

    使用场景与调用位置：
        本文件 main() 的启动配置阶段调用；独立运行脚本时首先经过此处。 本文件直接调用位置：main()。

    参数：
        无显式参数；所需配置由内部逻辑、模块常量或命令行提供。

    返回值：
        argparse.Namespace；成员名对应下方 add_argument 定义。

    注意事项：
        由本文件 main 在任何主要工作开始前调用；读取 sys.argv，处理默认值、类型和 choices；--help 或不合法参数通过 argparse 触发 SystemExit，不启动训练或推理。
    """
    parser = argparse.ArgumentParser(description="CrossEncoder 训练（交互型文本匹配）")
    parser.add_argument("--bert_path", default=DEFAULT_BERT_MODEL, type=str)
    parser.add_argument("--data_dir", default=str(DATA_DIR), type=str)
    parser.add_argument("--output_dir", default=str(OUTPUT_DIR), type=str)
    parser.add_argument(
        "--num_hidden_layers",
        default=12,
        type=int,
        help="BERT Transformer 层数（默认 4 层；全量 12 层留给学生自行实验）",
    )
    parser.add_argument("--epochs", default=3, type=int)
    parser.add_argument("--batch_size", default=32, type=int)
    parser.add_argument(
        "--max_length",
        default=128,
        type=int,
        help="句对总最大 token 数（两句拼接，建议 128）",
    )
    parser.add_argument("--lr", default=2e-5, type=float, help="BERT 层学习率")
    parser.add_argument(
        "--head_lr_mult", default=5.0, type=float, help="分类头学习率倍数"
    )
    parser.add_argument("--warmup_ratio", default=0.1, type=float)
    parser.add_argument("--grad_accum", default=1, type=int)
    parser.add_argument(
        "--class_weighting",
        choices=["none", "balanced"],
        default="balanced",
        help="默认按训练集逆频率加权，缓解 31:69 类别不均衡",
    )
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--allow_nondeterministic", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    main()
