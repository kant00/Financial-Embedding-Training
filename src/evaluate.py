"""
二分类评估：固定阈值测试、calibration 校准、指标与预测明细。

用途与流程位置：
    训练器导入 eval_biencoder / eval_crossencoder 做逐轮验证；
    独立入口加载 checkpoint，在 validation（calibration）或 test（final test）评估。

评估规则：
    双塔默认使用 checkpoint 阈值（缺失时回退 0.5），score >= threshold 判为正例。
    --tune_threshold 仅允许 validation；按分数中点和边界搜索正例 F1 最优阈值。
    显式校准不会自动改写 checkpoint；最终测试不得调阈值。
    交叉塔使用 argmax；其独立评估长度固定 128，--max_length 只控制双塔。

输入与输出：
    正式流程显式传 --data_dir、--ckpt，避免读取旧 data/afqmc；自定义模型位置用 --bert_path。
    --output_dir 保存 {model_type}_{split}_metrics.json 和对应 predictions.jsonl。
    包含正例/Macro/Weighted F1、ROC-AUC、AP 与 bootstrap 区间；f1 兼容字段指 Weighted-F1。
    双塔分布图仍写全局 outputs/figures，同 split 会覆盖；不同模型请隔离指标目录。

使用方式（在项目根目录执行，Bash）：
    python src/evaluate.py --model_type biencoder --ckpt outputs/dev/cosine_seed42/checkpoints/biencoder_cosine_best.pt --data_dir data/experiments/afqmc_seed42 --split validation --max_length 64 --output_dir outputs/dev/cosine_seed42/calibration

注意事项：
    新协议 test 是有标签 final test，不能用原官方标签为 -1 的 test 计算指标。
    预测 ID 与 manifest ID 命名不同，逐样本对比需核对文本、标签和顺序。

说明核对：2026-09-28；完整流程、前置条件及自定义模型路径见 USAGE_GUIDE.md。
"""

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse
import faulthandler
import sys
from pathlib import Path

if __name__ == "__main__":
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(line_buffering=True)
    print(f"启动评估，正在加载依赖…\nPython: {sys.executable}", flush=True)
    faulthandler.enable()
    # 若原生库导入或 CUDA 探测阻塞，打印栈，避免只留下启动提示。
    faulthandler.dump_traceback_later(30, repeat=True)


def _startup_stage(message):
    """仅独立入口显示启动阶段，不影响训练器导入本模块。"""
    if __name__ == "__main__":
        print(f"[启动] {message}", flush=True)

# Windows Conda nlp 环境中，先加载 matplotlib/torch 再加载 sklearn
# 会在其 pandas 原生扩展导入链触发 stack overflow。先初始化指标依赖。
_startup_stage("加载 scikit-learn（指标库）")
from sklearn.metrics import classification_report
_startup_stage("加载 PyTorch")
import torch
import torch.nn.functional as F
_startup_stage("加载 NumPy")
import numpy as np
_startup_stage("加载 matplotlib（绘图库）")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
_startup_stage("加载 Transformers / BertTokenizer")
from transformers import BertTokenizer
_startup_stage("加载 tqdm")
from tqdm.auto import tqdm

_startup_stage("加载项目指标模块")
from experiment_utils import DEFAULT_BERT_MODEL, environment_snapshot, save_json, write_jsonl
from metrics import (
    bootstrap_confidence_intervals,
    compute_binary_metrics,
    find_best_threshold,
)
_startup_stage("依赖加载完成")

# ── 默认路径 ──────────────────────────────────────────────────────────────
ROOT       = Path(__file__).parent.parent
DATA_DIR   = ROOT / "data" / "afqmc"
OUTPUT_DIR = ROOT / "outputs"
CKPT_DIR   = OUTPUT_DIR / "checkpoints"
FIG_DIR    = OUTPUT_DIR / "figures"


# ── BiEncoder 评估 ────────────────────────────────────────────────────────

@torch.no_grad()
def eval_biencoder(model, loader, device, find_threshold=True, threshold=0.5, show_progress=False):
    """评估独立句向量的余弦相似度并生成二分类指标。

    用途：
        评估独立句向量的余弦相似度并生成二分类指标。

    使用场景与调用位置：
        在 src/evaluate.py 的“加载模型执行二分类评估并输出指标、图像和预测明细”流程中使用。 本文件直接调用位置：main()。 跨文件调用：src/compare_methods.py:load_and_eval()、src/train_biencoder.py:main()。

    参数：
        model: 当前操作的模型实例；结构和设备须匹配输入。
        loader: 提供 batch 字典的 DataLoader；评估时应保持样本顺序。
        device: torch.device 或兼容设备描述，指定推理/训练位置。
        find_threshold: 是否根据当前标签重新校准阈值，最终测试必须为 False。
        threshold: score>=threshold 判为正例；应由 calibration 选择。
        show_progress: 独立评估显示批次进度，训练器调用默认关闭。

    返回值：
        字典：similarities、labels、threshold、auc 及统一指标；f1 是 weighted-F1 别名。

    注意事项：
        训练脚本每 epoch 调用并默认搜索正例 F1 最佳阈值；final test 必须 find_threshold=False 并传入 calibration 阈值。切换 eval，使用 no_grad，不恢复之前训练模式；无文件写入。
    """
    model.eval()
    all_sims, all_labels = [], []

    for batch in tqdm(loader, desc="BiEncoder evaluation", disable=not show_progress):
        batch_a = {
            "input_ids":      batch["input_ids_a"].to(device),
            "attention_mask": batch["attention_mask_a"].to(device),
            "token_type_ids": batch["token_type_ids_a"].to(device),
        }
        batch_b = {
            "input_ids":      batch["input_ids_b"].to(device),
            "attention_mask": batch["attention_mask_b"].to(device),
            "token_type_ids": batch["token_type_ids_b"].to(device),
        }
        emb_a, emb_b = model(batch_a, batch_b)
        sims = F.cosine_similarity(emb_a, emb_b, dim=-1).cpu().tolist()
        all_sims.extend(sims)
        all_labels.extend(batch["label"].tolist())

    sims   = np.array(all_sims)
    labels = np.array(all_labels)

    if find_threshold:
        threshold, _ = find_best_threshold(sims, labels, metric="f1_pos")

    preds    = (sims >= threshold).astype(int)
    metrics = compute_binary_metrics(labels, preds, sims)
    return {
        "similarities": all_sims,
        "labels":       all_labels,
        "threshold":    threshold,
        "auc":           metrics["roc_auc"],  # 兼容旧日志字段
        **metrics,
    }


def _find_best_threshold(sims, labels):
    """提供旧接口兼容的正例 F1 阈值搜索。

    用途：
        提供旧接口兼容的正例 F1 阈值搜索。

    使用场景与调用位置：
        在 src/evaluate.py 的“加载模型执行二分类评估并输出指标、图像和预测明细”流程中使用。

    参数：
        sims: 按样本顺序的余弦相似度数组或列表。
        labels: 一维真实标签序列，0 为不相似、1 为相似。

    返回值：
        float 最佳阈值。

    注意事项：
        内部转调 metrics.find_best_threshold；当前主要评估路径已直接调用公共实现，此函数为兼容保留。只可在校准集调参。
    """
    threshold, _ = find_best_threshold(sims, labels, metric="f1_pos")
    return threshold


# ── CrossEncoder 评估 ─────────────────────────────────────────────────────

@torch.no_grad()
def eval_crossencoder(model, loader, device, show_progress=False):
    """根据句对 logits 评估交叉编码器。

    用途：
        根据句对 logits 评估交叉编码器。

    使用场景与调用位置：
        在 src/evaluate.py 的“加载模型执行二分类评估并输出指标、图像和预测明细”流程中使用。 本文件直接调用位置：main()。 跨文件调用：src/compare_methods.py:load_and_eval()、src/train_crossencoder.py:main()。

    参数：
        model: 当前操作的模型实例；结构和设备须匹配输入。
        loader: 提供 batch 字典的 DataLoader；评估时应保持样本顺序。
        device: torch.device 或兼容设备描述，指定推理/训练位置。
        show_progress: 独立评估显示批次进度，训练器调用默认关闭。

    返回值：
        字典：logits、labels、probabilities 和统一二分类指标。

    注意事项：
        训练器、独立评估和方法对比调用；pred 为 argmax，probabilities 为 softmax 正类分数；不进行阈值优化。eval/no_grad，不写文件。
    """
    model.eval()
    all_logits, all_labels = [], []

    for batch in tqdm(loader, desc="CrossEncoder evaluation", disable=not show_progress):
        input_ids      = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        token_type_ids = batch["token_type_ids"].to(device)
        labels         = batch["label"]

        logits = model(input_ids, attention_mask, token_type_ids).cpu()
        all_logits.extend(logits.tolist())
        all_labels.extend(labels.tolist())

    logits_array = np.asarray(all_logits)
    preds    = np.argmax(logits_array, axis=1)
    labels   = np.array(all_labels)
    probabilities = torch.softmax(torch.tensor(logits_array), dim=-1)[:, 1].numpy()
    metrics = compute_binary_metrics(labels, preds, probabilities)

    return {
        "logits":   all_logits,
        "labels":   all_labels,
        "probabilities": probabilities.tolist(),
        **metrics,
    }


# ── 可视化 ────────────────────────────────────────────────────────────────

def plot_similarity_distribution(sims, labels, threshold, save_path, title="相似度分布"):
    """绘制正负例的余弦分数直方图及决策阈值。

    用途：
        绘制正负例的余弦分数直方图及决策阈值。

    使用场景与调用位置：
        在 src/evaluate.py 的“加载模型执行二分类评估并输出指标、图像和预测明细”流程中使用。 本文件直接调用位置：main()。

    参数：
        sims: 按样本顺序的余弦相似度数组或列表。
        labels: 一维真实标签序列，0 为不相似、1 为相似。
        threshold: score>=threshold 判为正例；应由 calibration 选择。
        save_path: 目标图片文件路径，通常为 PNG。
        title: 控制台分组标题或图像标题。

    返回值：
        None；创建父目录、保存图像并关闭 figure。

    注意事项：
        独立 BiEncoder 评估后调用；用于诊断重叠区。要求标签分数等长，单类样本可能使另一侧直方图无有效密度。
    """
    sims   = np.array(sims)
    labels = np.array(labels)

    pos_sims = sims[labels == 1]
    neg_sims = sims[labels == 0]
    from explore_data import _get_font
    font = _get_font()

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(pos_sims, bins=50, alpha=0.6, label=f"正样本 (n={len(pos_sims)})",
            color="#2196F3", density=True)
    ax.hist(neg_sims, bins=50, alpha=0.6, label=f"负样本 (n={len(neg_sims)})",
            color="#F44336", density=True)
    ax.axvline(threshold, color="black", linestyle="--", linewidth=1.5,
               label=f"评估阈值 = {threshold:.2f}")
    ax.set_xlabel("余弦相似度", fontproperties=font)
    ax.set_ylabel("密度", fontproperties=font)
    ax.set_title(title, fontproperties=font)
    ax.legend(prop=font)
    fig.tight_layout()

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  图表已保存 → {save_path}")


# ── 独立运行入口 ──────────────────────────────────────────────────────────

def parse_args():
    """解析加载模型执行二分类评估并输出指标、图像和预测明细所需的命令行配置。

    用途：
        解析加载模型执行二分类评估并输出指标、图像和预测明细所需的命令行配置。

    使用场景与调用位置：
        本文件 main() 的启动配置阶段调用；独立运行脚本时首先经过此处。 本文件直接调用位置：main()。

    参数：
        无显式参数；所需配置由内部逻辑、模块常量或命令行提供。

    返回值：
        argparse.Namespace；成员名对应下方 add_argument 定义。

    注意事项：
        由本文件 main 在任何主要工作开始前调用；读取 sys.argv，处理默认值、类型和 choices；--help 或不合法参数通过 argparse 触发 SystemExit，不启动训练或推理。
    """
    parser = argparse.ArgumentParser(description="文本匹配模型评估")
    parser.add_argument("--model_type", required=True, choices=["biencoder", "crossencoder"])
    parser.add_argument("--ckpt",       required=True, type=str, help="checkpoint .pt 文件路径")
    parser.add_argument("--data_dir",   default=str(DATA_DIR), type=str)
    parser.add_argument("--bert_path",  default=DEFAULT_BERT_MODEL, type=str)
    parser.add_argument("--batch_size", default=64, type=int)
    parser.add_argument("--max_length", default=64,  type=int, help="BiEncoder 单句最大长度")
    parser.add_argument("--split",      default="validation", choices=["validation", "test"],
                        help="新协议中 validation=calibration，test=有标签 final test")
    parser.add_argument("--threshold", type=float, default=None,
                        help="固定阈值；final test 必须使用 calibration 阶段保存的阈值")
    parser.add_argument("--tune_threshold", action="store_true",
                        help="仅在 calibration/validation 上显式启用阈值搜索")
    parser.add_argument("--output_dir", default=str(OUTPUT_DIR / "evaluations"))
    parser.add_argument("--bootstrap_samples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if not Path(args.ckpt).is_file():
        candidates = sorted((OUTPUT_DIR).rglob(Path(args.ckpt).name))
        candidates = [path for path in candidates if path.is_file()]
        hint = ""
        if candidates:
            hint = "\n找到同名 checkpoint，请确认训练配置后显式选择：\n" + "\n".join(
                f'  --ckpt "{path}"' for path in candidates
            )
        parser.error(f"checkpoint 文件不存在：{args.ckpt}{hint}")
    return args


def main():
    """组织本脚本的完整流程：加载模型执行二分类评估并输出指标、图像和预测明细。

    用途：
        组织本脚本的完整流程：加载模型执行二分类评估并输出指标、图像和预测明细。

    使用场景与调用位置：
        直接运行此文件时由 if __name__ == "__main__" 入口调用；也可显式调用 main()，此时仍读取进程命令行。

    参数：
        无显式参数；所需配置由内部逻辑、模块常量或命令行提供。

    返回值：
        None；产物及副作用见注意事项，配置由命令行读取。

    注意事项：
        checkpoint 决定模型结构，固定阈值评估是默认；test+tune_threshold 会拒绝执行。data_dir 必须指向有标签的新协议测试集；输出汇总和预测，图像仍写模块级 FIG_DIR。
    """
    args   = parse_args()
    _startup_stage("检测 CUDA 设备")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if __name__ == "__main__":
        faulthandler.cancel_dump_traceback_later()
    print(f"设备: {device}")
    if device.type == "cpu":
        print("当前 Python 环境未启用 CUDA，将使用 CPU，推理可能较慢。", flush=True)
    print(f"加载 checkpoint: {args.ckpt}")

    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    print(f"训练信息: {ckpt.get('args', {})}")

    tokenizer = BertTokenizer.from_pretrained(args.bert_path)
    data_path = Path(args.data_dir) / f"{args.split}.jsonl"

    FIG_DIR.mkdir(parents=True, exist_ok=True)

    if args.model_type == "biencoder":
        from model import build_biencoder
        from dataset import PairDataset
        from torch.utils.data import DataLoader

        saved_args = ckpt.get("args", {})
        model = build_biencoder(
            bert_path=args.bert_path,
            pool=saved_args.get("pool", "mean"),
            num_hidden_layers=saved_args.get("num_hidden_layers"),
        ).to(device)
        model.load_state_dict(ckpt["state_dict"])

        ds     = PairDataset(data_path, tokenizer, args.max_length)
        loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
        print(f"开始评估: {data_path}，共 {len(ds):,} 条 / {len(loader)} 批", flush=True)

        saved_threshold = ckpt.get("threshold", 0.5)
        threshold = args.threshold if args.threshold is not None else saved_threshold
        if args.split == "test" and args.tune_threshold:
            raise ValueError("禁止在 final test 上调阈值；请使用 checkpoint 中的 calibration 阈值")
        metrics = eval_biencoder(
            model, loader, device,
            find_threshold=args.tune_threshold,
            threshold=threshold,
            show_progress=True,
        )
        print(f"\n{'='*50}")
        print(f"BiEncoder 评估结果（{args.split}，{len(ds)} 条）")
        print(f"  评估阈值: {metrics['threshold']:.2f}")
        print(f"  Accuracy: {metrics['accuracy']:.4f}")
        print(f"  F1      : {metrics['f1']:.4f}")
        print(f"  F1(pos) : {metrics['f1_pos']:.4f}")
        print(f"  F1(macro): {metrics['f1_macro']:.4f}")
        print(f"  PR-AUC  : {metrics['pr_auc']:.4f}")
        print(f"  AUC     : {metrics['auc']:.4f}")

        plot_similarity_distribution(
            metrics["similarities"], metrics["labels"], metrics["threshold"],
            save_path=FIG_DIR / f"biencoder_{args.split}_sim_dist.png",
            title=f"BiEncoder 相似度分布（{args.split}）",
        )

        # 打印分类报告
        preds = (np.array(metrics["similarities"]) >= metrics["threshold"]).astype(int)
        print(f"\n{classification_report(metrics['labels'], preds, target_names=['不相似', '相似'])}")
        scores = metrics["similarities"]

    elif args.model_type == "crossencoder":
        from model import build_crossencoder
        from dataset import CrossEncoderDataset
        from torch.utils.data import DataLoader

        saved_args = ckpt.get("args", {})
        model = build_crossencoder(
            bert_path=args.bert_path,
            num_hidden_layers=saved_args.get("num_hidden_layers"),
        ).to(device)
        model.load_state_dict(ckpt["state_dict"])

        ds     = CrossEncoderDataset(data_path, tokenizer, max_length=128)
        loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

        print(f"开始评估: {data_path}，共 {len(ds):,} 条 / {len(loader)} 批", flush=True)
        metrics = eval_crossencoder(model, loader, device, show_progress=True)
        print(f"\n{'='*50}")
        print(f"CrossEncoder 评估结果（{args.split}，{len(ds)} 条）")
        print(f"  Accuracy: {metrics['accuracy']:.4f}")
        print(f"  F1      : {metrics['f1']:.4f}")
        print(f"  F1(pos) : {metrics['f1_pos']:.4f}")
        print(f"  F1(macro): {metrics['f1_macro']:.4f}")
        print(f"  PR-AUC  : {metrics['pr_auc']:.4f}")

        preds = np.argmax(metrics["logits"], axis=1)
        print(f"\n{classification_report(metrics['labels'], preds, target_names=['不相似', '相似'])}")
        scores = metrics["probabilities"]

    print(f"正在计算置信区间（{args.bootstrap_samples} 次重采样）…", flush=True)
    metrics["confidence_intervals"] = bootstrap_confidence_intervals(
        metrics["labels"], preds, scores,
        n_bootstrap=args.bootstrap_samples,
        seed=args.seed,
    )
    metrics["environment"] = environment_snapshot()
    metrics["split"] = args.split
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    # 大数组和逐样本内容单独保存，汇总 JSON 保持便于查看。
    summary = {
        key: value for key, value in metrics.items()
        if key not in {"labels", "similarities", "logits", "probabilities"}
    }
    save_json(summary, output_dir / f"{args.model_type}_{args.split}_metrics.json")
    rows = []
    raw_rows = ds.rows
    for index, (row, label, prediction, score) in enumerate(
        zip(raw_rows, metrics["labels"], preds, scores)
    ):
        rows.append({
            "sample_id": f"{args.split}-{index:05d}",
            "sentence1": row["sentence1"],
            "sentence2": row["sentence2"],
            "label": int(label),
            "prediction": int(prediction),
            "score": float(score),
        })
    write_jsonl(rows, output_dir / f"{args.model_type}_{args.split}_predictions.jsonl")
    print(f"评估完成：{len(rows):,} 条样本", flush=True)
    print(f"指标文件: {(output_dir / f'{args.model_type}_{args.split}_metrics.json').resolve()}")
    print(f"预测文件: {(output_dir / f'{args.model_type}_{args.split}_predictions.jsonl').resolve()}")


if __name__ == "__main__":
    main()
