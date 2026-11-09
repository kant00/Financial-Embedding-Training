"""
Bi-Encoder 训练入口：Cosine、随机/预计算 Triplet 与 in-batch InfoNCE。

用途与流程位置：
    在固定 train/calibration 协议上学习句向量；由人工开发流程或实验套件调用。
    两句共享 BERT、分别编码并做 L2 归一化，支持 cls/mean/max 池化。

训练目标与数据：
    cosine：读取 0/1 句对，负例项为 max(0, cosine-margin)，增大 margin 会放宽约束。
    triplet：在归一化向量上使用 TripletMarginLoss 的距离约束，不是直接加余弦 margin。
    默认三元组在数据集初始化时采样；--hard_negative_file 可读取离线挖掘的三元组。
    inbatch：读取正例句对，以同 batch 候选构造双向 InfoNCE，并屏蔽部分重复/冲突候选。
    batch_size 应大于 1；梯度累积不会扩大 InfoNCE 的 batch 内负样本池。

输入与输出：
    正式流程指定 --data_dir 和 --output_dir；默认 data_dir 仍是旧 data/afqmc。
    示例使用默认 BERT 位置，模型位于其他目录时追加 --bert_path。
    每轮在 validation（新协议 calibration）选择正例 F1 与阈值，保存最佳 checkpoint。
    输出 checkpoints/biencoder_{loss}_best.pt 与 logs/biencoder_{loss}_log.json。
    日志包含配置、环境和逐轮记录；阈值随 checkpoint 保存，最终测试不得重新校准。

使用方式（在项目根目录执行，Bash）：
    python src/train_biencoder.py --data_dir data/experiments/afqmc_seed42 --output_dir outputs/dev/inbatch_seed42 --loss inbatch

注意事项：
    Hard Triplet 从预训练模型重新初始化，不从挖掘用 checkpoint 续训。
    限制层数不保证耗时线性减少；不承诺固定训练时间或指标提升。
    最佳 F1 始终为 0 时可能没有最佳文件；先检查 checkpoint 再进入评估。

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
import torch.nn.functional as F
from torch.optim import AdamW
from tqdm import tqdm
from transformers import BertTokenizer, get_linear_schedule_with_warmup

from dataset import build_inbatch_loader, build_pair_loaders, build_triplet_loader
from experiment_utils import DEFAULT_BERT_MODEL, environment_snapshot, save_json, set_global_seed
from evaluate import eval_biencoder
from model import build_biencoder

# ── 默认路径 ──────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent
DATA_DIR = ROOT / "data" / "afqmc"
OUTPUT_DIR = ROOT / "outputs"
CKPT_DIR = OUTPUT_DIR / "checkpoints"
LOG_DIR = OUTPUT_DIR / "logs"


# ── 训练一个 epoch ────────────────────────────────────────────────────────


def train_one_epoch_cosine(
    model, loader, optimizer, scheduler, device, epoch, total_epochs, margin, grad_accum
):
    """执行一个 epoch 的 CosineEmbeddingLoss 训练。

    用途：
        执行一个 epoch 的 CosineEmbeddingLoss 训练。

    使用场景与调用位置：
        在 src/train_biencoder.py 的“训练 Cosine、Triplet 或双向 InfoNCE 双编码器并保存最佳模型”流程中使用。 本文件直接调用位置：main()。

    参数：
        model: 当前操作的模型实例；结构和设备须匹配输入。
        loader: 提供 batch 字典的 DataLoader；评估时应保持样本顺序。
        optimizer: 维护可训练参数及优化状态的优化器，将被原地更新。
        scheduler: 随 optimizer.step 更新的学习率调度器。
        device: torch.device 或兼容设备描述，指定推理/训练位置。
        epoch: 当前 epoch 序号，用于进度显示。
        total_epochs: 总 epoch 数，用于进度显示。
        margin: 损失中的间隔超参数；不同 loss 的数学含义见注意事项。
        grad_accum: 每次参数更新累积的 batch 数，必须是正整数。

    返回值：
        按 batch 样本数加权的平均训练 loss；Python 浮点数。

    注意事项：
        标签 0/1 转成 -1/+1；正例 loss=1-cos，负例 loss=max(0,cos-margin)。 调用 model.train，反向传播并更新 optimizer/scheduler；梯度范数裁剪为 1，末尾不足 grad_accum 也更新。尾组 loss 仍除以完整 grad_accum，实际梯度尺度需注意；loader 需非空。
    """
    model.train()
    total_loss, total_samples = 0.0, 0
    optimizer.zero_grad()

    pbar = tqdm(loader, desc=f"Epoch {epoch}/{total_epochs} [Cosine]", leave=False)
    for step, batch in enumerate(pbar):
        batch_a = {
            "input_ids": batch["input_ids_a"].to(device),
            "attention_mask": batch["attention_mask_a"].to(device),
            "token_type_ids": batch["token_type_ids_a"].to(device),
        }
        batch_b = {
            "input_ids": batch["input_ids_b"].to(device),
            "attention_mask": batch["attention_mask_b"].to(device),
            "token_type_ids": batch["token_type_ids_b"].to(device),
        }
        labels = batch["label"].to(device)

        emb_a, emb_b = model(batch_a, batch_b)

        # label 0→-1，1→+1（cosine_embedding_loss 要求 target ∈ {-1, +1}）
        cos_target = labels.float() * 2 - 1
        loss = F.cosine_embedding_loss(emb_a, emb_b, cos_target, margin=margin)

        (loss / grad_accum).backward()
        should_step = (step + 1) % grad_accum == 0 or (step + 1) == len(loader)
        if should_step:
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        total_loss += loss.item() * labels.size(0)
        total_samples += labels.size(0)
        pbar.set_postfix(loss=f"{total_loss / total_samples:.4f}")

    return total_loss / total_samples


def train_one_epoch_triplet(
    model, loader, optimizer, scheduler, device, epoch, total_epochs, margin, grad_accum
):
    """执行一个 epoch 的 TripletMarginLoss 训练。

    用途：
        执行一个 epoch 的 TripletMarginLoss 训练。

    使用场景与调用位置：
        在 src/train_biencoder.py 的“训练 Cosine、Triplet 或双向 InfoNCE 双编码器并保存最佳模型”流程中使用。 本文件直接调用位置：main()。

    参数：
        model: 当前操作的模型实例；结构和设备须匹配输入。
        loader: 提供 batch 字典的 DataLoader；评估时应保持样本顺序。
        optimizer: 维护可训练参数及优化状态的优化器，将被原地更新。
        scheduler: 随 optimizer.step 更新的学习率调度器。
        device: torch.device 或兼容设备描述，指定推理/训练位置。
        epoch: 当前 epoch 序号，用于进度显示。
        total_epochs: 总 epoch 数，用于进度显示。
        margin: 损失中的间隔超参数；不同 loss 的数学含义见注意事项。
        grad_accum: 每次参数更新累积的 batch 数，必须是正整数。

    返回值：
        按 batch 样本数加权的平均训练 loss；Python 浮点数。

    注意事项：
        a/p/n 独立编码，使用欧氏距离 margin；L2 归一化不意味着欧氏 margin 与余弦 margin 数值相同。 调用 model.train，反向传播并更新 optimizer/scheduler；梯度范数裁剪为 1，末尾不足 grad_accum 也更新。尾组 loss 仍除以完整 grad_accum，实际梯度尺度需注意；loader 需非空。
    """
    model.train()
    total_loss, total_samples = 0.0, 0
    optimizer.zero_grad()

    pbar = tqdm(loader, desc=f"Epoch {epoch}/{total_epochs} [Triplet]", leave=False)
    for step, batch in enumerate(pbar):
        enc_a = {
            "input_ids": batch["input_ids_a"].to(device),
            "attention_mask": batch["attention_mask_a"].to(device),
            "token_type_ids": batch["token_type_ids_a"].to(device),
        }
        enc_p = {
            "input_ids": batch["input_ids_p"].to(device),
            "attention_mask": batch["attention_mask_p"].to(device),
            "token_type_ids": batch["token_type_ids_p"].to(device),
        }
        enc_n = {
            "input_ids": batch["input_ids_n"].to(device),
            "attention_mask": batch["attention_mask_n"].to(device),
            "token_type_ids": batch["token_type_ids_n"].to(device),
        }

        emb_a = model.encode(**enc_a)
        emb_p = model.encode(**enc_p)
        emb_n = model.encode(**enc_n)

        # triplet_margin_loss 默认用欧氏距离；
        # 由于 encode() 已 L2 归一化，向量在单位球上，欧氏距离与余弦距离单调相关
        loss = F.triplet_margin_loss(emb_a, emb_p, emb_n, margin=margin)

        (loss / grad_accum).backward()
        should_step = (step + 1) % grad_accum == 0 or (step + 1) == len(loader)
        if should_step:
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        bs = emb_a.size(0)
        total_loss += loss.item() * bs
        total_samples += bs
        pbar.set_postfix(loss=f"{total_loss / total_samples:.4f}")

    return total_loss / total_samples


def train_one_epoch_inbatch(
    model,
    loader,
    optimizer,
    scheduler,
    device,
    epoch,
    total_epochs,
    temperature,
    grad_accum,
):
    """执行一个 epoch 的 双向 InfoNCE 训练。

    用途：
        执行一个 epoch 的 双向 InfoNCE 训练。

    使用场景与调用位置：
        在 src/train_biencoder.py 的“训练 Cosine、Triplet 或双向 InfoNCE 双编码器并保存最佳模型”流程中使用。 本文件直接调用位置：main()。

    参数：
        model: 当前操作的模型实例；结构和设备须匹配输入。
        loader: 提供 batch 字典的 DataLoader；评估时应保持样本顺序。
        optimizer: 维护可训练参数及优化状态的优化器，将被原地更新。
        scheduler: 随 optimizer.step 更新的学习率调度器。
        device: torch.device 或兼容设备描述，指定推理/训练位置。
        epoch: 当前 epoch 序号，用于进度显示。
        total_epochs: 总 epoch 数，用于进度显示。
        temperature: InfoNCE 的正温度系数，用于缩放相似度 logits。
        grad_accum: 每次参数更新累积的 batch 数，必须是正整数。

    返回值：
        按 batch 样本数加权的平均训练 loss；Python 浮点数。

    注意事项：
        点积矩阵 [B,B] 除以 temperature；对角线是正例，两方向交叉熵取平均。屏蔽共享 anchor/重复文本，不保证过滤所有语义假负例；B=1 无有效负例。 调用 model.train，反向传播并更新 optimizer/scheduler；梯度范数裁剪为 1，末尾不足 grad_accum 也更新。尾组 loss 仍除以完整 grad_accum，实际梯度尺度需注意；loader 需非空。
    """
    model.train()
    total_loss, total_samples = 0.0, 0
    optimizer.zero_grad()
    pbar = tqdm(loader, desc=f"Epoch {epoch}/{total_epochs} [InBatch]", leave=False)
    for step, batch in enumerate(pbar):
        batch_a = {
            "input_ids": batch["input_ids_a"].to(device),
            "attention_mask": batch["attention_mask_a"].to(device),
            "token_type_ids": batch["token_type_ids_a"].to(device),
        }
        batch_b = {
            "input_ids": batch["input_ids_b"].to(device),
            "attention_mask": batch["attention_mask_b"].to(device),
            "token_type_ids": batch["token_type_ids_b"].to(device),
        }
        emb_a, emb_b = model(batch_a, batch_b)
        logits = emb_a @ emb_b.T / temperature
        targets = torch.arange(logits.size(0), device=device)
        # 同一 batch 可能出现共享 anchor 或完全相同句子；这些位置并非可靠负例。
        # 将它们从 softmax 分母中屏蔽，降低 in-batch false negative 风险。
        text_a = ["".join(text.strip().lower().split()) for text in batch["text_a"]]
        text_b = ["".join(text.strip().lower().split()) for text in batch["text_b"]]
        mask_ab = torch.zeros_like(logits, dtype=torch.bool)
        mask_ba = torch.zeros_like(logits, dtype=torch.bool)
        for row in range(len(text_a)):
            for column in range(len(text_a)):
                if row == column:
                    continue
                mask_ab[row, column] = (
                    text_a[row] == text_a[column]
                    or text_a[row] == text_b[column]
                    or text_b[row] == text_b[column]
                )
                mask_ba[row, column] = (
                    text_b[row] == text_b[column]
                    or text_b[row] == text_a[column]
                    or text_a[row] == text_a[column]
                )
        logits_ab = logits.masked_fill(mask_ab, float("-inf"))
        logits_ba = logits.T.masked_fill(mask_ba, float("-inf"))
        # 双向约束可避免只有 A->B 方向获得充分监督。
        loss = (
            F.cross_entropy(logits_ab, targets) + F.cross_entropy(logits_ba, targets)
        ) / 2.0
        (loss / grad_accum).backward()

        should_step = (step + 1) % grad_accum == 0 or (step + 1) == len(loader)
        if should_step:
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        batch_size = emb_a.size(0)
        total_loss += loss.item() * batch_size
        total_samples += batch_size
        pbar.set_postfix(loss=f"{total_loss / total_samples:.4f}")
    return total_loss / total_samples


# ── 主训练流程 ────────────────────────────────────────────────────────────


def main():
    """组织本脚本的完整流程：训练 Cosine、Triplet 或双向 InfoNCE 双编码器并保存最佳模型。

    用途：
        组织本脚本的完整流程：训练 Cosine、Triplet 或双向 InfoNCE 双编码器并保存最佳模型。

    使用场景与调用位置：
        直接运行此文件时由 if __name__ == "__main__" 入口调用；也可显式调用 main()，此时仍读取进程命令行。

    参数：
        无显式参数；所需配置由内部逻辑、模块常量或命令行提供。

    返回值：
        None；产物及副作用见注意事项，配置由命令行读取。

    注意事项：
        根据 loss 选择 DataLoader；用验证正例 F1 选 checkpoint 并保存阈值。默认 data_dir 仍是原 AFQMC；使用新协议须显式传入 data/experiments/...。best 初值为 0，所有 epoch 正例 F1 都为 0 时不会保存模型。
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
        f"Loss 类型: {args.loss}  池化策略: {args.pool}  "
        f"BERT 层数: {args.num_hidden_layers}  Epochs: {args.epochs}"
    )

    # ── Tokenizer & DataLoader ────────────────────────────────────────────
    tokenizer = BertTokenizer.from_pretrained(args.bert_path)
    print("\nDataLoader 构建中...")

    if args.loss == "cosine":
        train_loader, val_loader, _ = build_pair_loaders(
            args.data_dir,
            tokenizer,
            max_length=args.max_length,
            batch_size=args.batch_size,
        )
    elif args.loss == "triplet":
        train_loader, val_loader = build_triplet_loader(
            args.data_dir,
            tokenizer,
            max_length=args.max_length,
            batch_size=args.batch_size,
            hard_negative_file=args.hard_negative_file,
        )
    else:
        train_loader, val_loader = build_inbatch_loader(
            args.data_dir,
            tokenizer,
            max_length=args.max_length,
            batch_size=args.batch_size,
        )

    # ── 模型 ─────────────────────────────────────────────────────────────
    print("\n构建模型...")
    model = build_biencoder(
        bert_path=args.bert_path,
        pool=args.pool,
        num_hidden_layers=args.num_hidden_layers,
    ).to(device)

    # ── 分层学习率 ────────────────────────────────────────────────────────
    # BERT 骨干用较小 lr，防止预训练知识被过度破坏
    bert_params = list(model.bert.parameters())
    # 当前 BiEncoder 没有可训练投影头；Dropout 不含参数，不能使用 head_lr_mult。
    optimizer = AdamW(bert_params, lr=args.lr, weight_decay=0.01)

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

    # ── 训练循环 ──────────────────────────────────────────────────────────
    ckpt_name = f"biencoder_{args.loss}_best.pt"
    ckpt_path = ckpt_dir / ckpt_name
    best_val_f1 = 0.0
    log_records = []

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        if args.loss == "cosine":
            train_loss = train_one_epoch_cosine(
                model,
                train_loader,
                optimizer,
                scheduler,
                device,
                epoch,
                args.epochs,
                args.margin,
                args.grad_accum,
            )
        elif args.loss == "triplet":
            train_loss = train_one_epoch_triplet(
                model,
                train_loader,
                optimizer,
                scheduler,
                device,
                epoch,
                args.epochs,
                args.margin,
                args.grad_accum,
            )
        else:
            train_loss = train_one_epoch_inbatch(
                model,
                train_loader,
                optimizer,
                scheduler,
                device,
                epoch,
                args.epochs,
                args.temperature,
                args.grad_accum,
            )

        # 每个 epoch 末评估 val（BiEncoder：相似度 + 阈值搜索）
        val_metrics = eval_biencoder(model, val_loader, device)
        elapsed = time.time() - t0

        val_acc = val_metrics["accuracy"]
        # 类别不均衡任务以正例 F1 选模；weighted-F1 仍完整记录但不再混用。
        val_f1 = val_metrics["f1_pos"]
        val_thr = val_metrics["threshold"]
        print(
            f"Epoch {epoch}/{args.epochs} | "
            f"train_loss={train_loss:.4f} | "
            f"val_acc={val_acc:.4f} val_f1={val_f1:.4f} threshold={val_thr:.2f} | "
            f"{elapsed:.0f}s"
        )

        log_records.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_acc": val_acc,
                "val_f1": val_f1,
                "val_f1_pos": val_metrics["f1_pos"],
                "val_f1_macro": val_metrics["f1_macro"],
                "val_f1_weighted": val_metrics["f1_weighted"],
                "threshold": val_thr,
                "elapsed_s": elapsed,
            }
        )

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save(
                {
                    "epoch": epoch,
                    "state_dict": model.state_dict(),
                    "threshold": val_thr,
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
    log_path = log_dir / f"biencoder_{args.loss}_log.json"
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
    print(f"\n运行评估：python evaluate.py --model_type biencoder --ckpt {ckpt_path}")


# ── 参数解析 ──────────────────────────────────────────────────────────────


def parse_args():
    """解析训练 Cosine、Triplet 或双向 InfoNCE 双编码器并保存最佳模型所需的命令行配置。

    用途：
        解析训练 Cosine、Triplet 或双向 InfoNCE 双编码器并保存最佳模型所需的命令行配置。

    使用场景与调用位置：
        本文件 main() 的启动配置阶段调用；独立运行脚本时首先经过此处。 本文件直接调用位置：main()。

    参数：
        无显式参数；所需配置由内部逻辑、模块常量或命令行提供。

    返回值：
        argparse.Namespace；成员名对应下方 add_argument 定义。

    注意事项：
        由本文件 main 在任何主要工作开始前调用；读取 sys.argv，处理默认值、类型和 choices；--help 或不合法参数通过 argparse 触发 SystemExit，不启动训练或推理。
    """
    parser = argparse.ArgumentParser(description="BiEncoder 训练（表示型文本匹配）")
    parser.add_argument("--bert_path", default=DEFAULT_BERT_MODEL, type=str)
    parser.add_argument("--data_dir", default=str(DATA_DIR), type=str)
    parser.add_argument("--output_dir", default=str(OUTPUT_DIR), type=str)
    parser.add_argument(
        "--loss",
        default="cosine",
        choices=["cosine", "triplet", "inbatch"],
        help="cosine / triplet / inbatch(对称 InfoNCE)",
    )
    parser.add_argument(
        "--pool",
        default="mean",
        choices=["cls", "mean", "max"],
        help="句向量池化策略（Sentence-BERT 论文推荐 mean）",
    )
    parser.add_argument(
        "--num_hidden_layers",
        default=12,
        type=int,
        help="BERT Transformer 层数（默认 4 层快速验证；全量 12 层留给学生）",
    )
    parser.add_argument("--epochs", default=3, type=int)
    parser.add_argument("--batch_size", default=32, type=int)
    parser.add_argument("--max_length", default=64, type=int, help="单句最大 token 数")
    parser.add_argument("--lr", default=2e-5, type=float, help="BERT 层学习率")
    parser.add_argument("--warmup_ratio", default=0.1, type=float)
    parser.add_argument("--grad_accum", default=1, type=int)
    parser.add_argument(
        "--margin",
        default=0.3,
        type=float,
        help="CosineEmbeddingLoss 的 margin；TripletLoss 的 margin 同参数",
    )
    parser.add_argument(
        "--temperature", default=0.05, type=float, help="in-batch InfoNCE 温度系数"
    )
    parser.add_argument(
        "--hard_negative_file",
        default=None,
        help="离线难负样本 JSONL；仅 triplet 模式使用",
    )
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument(
        "--allow_nondeterministic",
        action="store_true",
        help="允许非确定性 CUDA 算子以换取速度",
    )
    return parser.parse_args()


if __name__ == "__main__":
    main()
