"""已有 Cosine、Triplet、InBatch、CrossEncoder 的统一评估与报告。

python src/compare_methods.py
python src/compare_methods.py --data_dir data/afqmc --split validation --output_dir outputs/comparison

不重新训练或调阈值；默认 validation 是选模集成绩，不是独立测试成绩。
输出按 split 和模型隔离的指标、预测、图表，以及汇总 JSON/CSV/Markdown。
"""
import argparse
import csv
import faulthandler
import gc
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
if __name__ == "__main__":
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(line_buffering=True)
    print(f"启动四模型对比，加载依赖…\nPython: {sys.executable}", flush=True)
    faulthandler.enable()

# 保持 sklearn 在 torch/matplotlib 之前，避免 Windows Conda 导入栈溢出。
from sklearn.metrics import confusion_matrix
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from transformers import BertTokenizer
from dataset import PairDataset, CrossEncoderDataset
from evaluate import eval_biencoder, eval_crossencoder, plot_similarity_distribution
from experiment_utils import DEFAULT_BERT_MODEL, environment_snapshot, save_json, write_jsonl, sha256_file
from metrics import bootstrap_confidence_intervals
from model import build_biencoder, build_crossencoder

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "afqmc"
CKPT_DIR = ROOT / "outputs" / "checkpoints"
METHODS = [
    {"key": "biencoder_cosine", "label": "BiEncoder\nCosine", "type": "biencoder", "color": "#2196F3", "ckpt": "biencoder_cosine_best.pt"},
    {"key": "biencoder_triplet", "label": "BiEncoder\nTriplet", "type": "biencoder", "color": "#4CAF50", "ckpt": "biencoder_triplet_best.pt"},
    {"key": "biencoder_inbatch", "label": "BiEncoder\nInBatch", "type": "biencoder", "color": "#9C27B0", "ckpt": "biencoder_inbatch_best.pt"},
    {"key": "crossencoder", "label": "CrossEncoder", "type": "crossencoder", "color": "#FF9800", "ckpt": "crossencoder_best.pt"},
]
METRIC_KEYS = ["accuracy", "precision_pos", "recall_pos", "f1_pos", "f1_macro", "f1_weighted", "roc_auc", "pr_auc"]


def parse_args():
    parser = argparse.ArgumentParser(description="已有四模型评估、图表与 Markdown 报告")
    parser.add_argument("--split", default="validation", choices=["validation", "test"])
    parser.add_argument("--batch_size", default=64, type=int)
    parser.add_argument("--data_dir", default=DATA_DIR, type=Path)
    parser.add_argument("--ckpt_dir", default=CKPT_DIR, type=Path)
    parser.add_argument("--bert_path", default=None, help="覆盖 checkpoint 保存的 BERT 路径")
    parser.add_argument("--log_dir", type=Path, help="默认 checkpoint 目录同级 logs")
    parser.add_argument("--output_dir", default=ROOT / "outputs" / "comparison", type=Path)
    parser.add_argument("--bootstrap_samples", default=1000, type=int, help="置信区间采样次数；0 为跳过")
    parser.add_argument("--seed", default=42, type=int, help="仅控制 bootstrap 随机种子")
    parser.add_argument("--allow_missing", action="store_true", help="允许生成部分模型报告")
    args = parser.parse_args()
    if args.batch_size < 1 or args.bootstrap_samples < 0:
        parser.error("batch_size 必须为正数，bootstrap_samples 不能为负数")
    return args


def validate_inputs(args):
    """在加载模型前校验数据与权重，禁止未公开标签参与评估。"""
    with (args.data_dir / f"{args.split}.jsonl").open(encoding="utf-8") as file:
        rows = [json.loads(line) for line in file if line.strip()]
    if not rows:
        raise ValueError("评估集为空")
    for index, row in enumerate(rows):
        if row.get("label") not in (0, 1) or not all(isinstance(row.get(k), str) for k in ("sentence1", "sentence2")):
            raise ValueError(f"第 {index + 1} 条样本必须含句对和 0/1 标签；官方 test 的 -1 标签不能评估")
    missing = [m["key"] for m in METHODS if not (args.ckpt_dir / m["ckpt"]).is_file()]
    if missing and not args.allow_missing:
        raise FileNotFoundError(f"缺少 checkpoint：{', '.join(missing)}。检查 --ckpt_dir；部分对比需显式 --allow_missing")
    if len(missing) == len(METHODS):
        raise FileNotFoundError("没有可用的 checkpoint")
    return rows, missing


def load_and_eval(method, args, device, raw_rows):
    """单模型加载与评估；返回数据而非模型，及时释放显存。"""
    ckpt_path = args.ckpt_dir / method["ckpt"]
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    saved = ckpt.get("args", {})
    bert_path = args.bert_path or saved.get("bert_path") or DEFAULT_BERT_MODEL
    tokenizer = BertTokenizer.from_pretrained(bert_path)
    is_bi = method["type"] == "biencoder"
    max_length = saved.get("max_length", 64 if is_bi else 128)
    kwargs = {"bert_path": bert_path, "num_hidden_layers": saved.get("num_hidden_layers")}
    model = build_biencoder(pool=saved.get("pool", "mean"), **kwargs) if is_bi else build_crossencoder(**kwargs)
    model.load_state_dict(ckpt["state_dict"])
    best_epoch = ckpt.get("epoch")
    threshold = ckpt.get("threshold", 0.5)
    threshold_source = "checkpoint" if "threshold" in ckpt else "fallback_0.5"
    del ckpt
    model.to(device)
    dataset_class = PairDataset if is_bi else CrossEncoderDataset
    ds = dataset_class(args.data_dir / f"{args.split}.jsonl", tokenizer, max_length=max_length)
    if ds.rows != raw_rows:
        raise ValueError("数据加载器更改了样本或顺序，停止对比")
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
    try:
        if is_bi:
            result = eval_biencoder(model, loader, device, find_threshold=False, threshold=threshold, show_progress=True)
            scores = result["similarities"]
            predictions = (np.array(scores) >= threshold).astype(int)
        else:
            result = eval_crossencoder(model, loader, device, show_progress=True)
            scores = result["probabilities"]
            predictions = np.argmax(result["logits"], axis=1)
    finally:
        del model, loader, ds
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
    if args.bootstrap_samples:
        print(f"计算 {method['key']} 的置信区间…", flush=True)
        result["confidence_intervals"] = bootstrap_confidence_intervals(
            result["labels"], predictions, scores if len(set(result["labels"])) == 2 else None,
            n_bootstrap=args.bootstrap_samples, seed=args.seed)
    result.update(method)
    result.update({"checkpoint": str(ckpt_path.resolve()), "training_config": saved,
                   "best_epoch": best_epoch, "eval_max_length": max_length,
                   "eval_bert_path": str(bert_path), "predictions": predictions.tolist(),
                   "threshold_source": threshold_source if is_bi else "argmax"})
    return result


def compact_result(result):
    return {k: v for k, v in result.items() if k not in {"labels", "predictions", "similarities", "probabilities", "logits"}}


def save_model_result(result, rows, output_dir, split):
    model_dir = output_dir / result["key"]
    save_json(compact_result(result), model_dir / "metrics.json")
    scores = result.get("similarities", result.get("probabilities"))
    write_jsonl((dict(sample_id=f"{split}-{i:05d}", **row, prediction=pred, score=float(score))
                 for i, (row, pred, score) in enumerate(zip(rows, result["predictions"], scores))),
                model_dir / "predictions.jsonl")
    if result["type"] == "biencoder":
        plot_similarity_distribution(scores, result["labels"], result["threshold"],
                                     model_dir / "similarity_distribution.png", title=result["key"])


def plot_comparison_bar(results, save_path):
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(results))
    for offset, key, label in [(-0.26, "accuracy", "Accuracy"), (0, "f1_pos", "F1 (positive)"), (0.26, "f1_macro", "Macro-F1")]:
        bars = ax.bar(x + offset, [r[key] for r in results], 0.25, label=label)
        ax.bar_label(bars, fmt="%.3f", padding=3, fontsize=8)
    ax.set_xticks(x, [r["label"] for r in results])
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("Score")
    ax.set_title("Fixed-split model comparison")
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_confusions(results, save_path):
    fig, axes = plt.subplots(1, len(results), figsize=(4 * len(results), 4.5), squeeze=False)
    max_count = max(confusion_matrix(r["labels"], r["predictions"], labels=[0, 1]).max() for r in results)
    for ax, result in zip(axes[0], results):
        matrix = confusion_matrix(result["labels"], result["predictions"], labels=[0, 1])
        ax.imshow(matrix, cmap="Blues", vmin=0, vmax=max_count)
        for (i, j), value in np.ndenumerate(matrix):
            ax.text(j, i, str(value), ha="center", va="center", color="white" if value > max_count / 2 else "black")
        ax.set_xticks([0, 1], ["0", "1"])
        ax.set_yticks([0, 1], ["0", "1"])
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title(result["label"])
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)


def plot_training_curves(results, log_dir, save_path):
    """只使用配置与 checkpoint 一致的日志；缺失或不匹配时记录说明。"""
    notes, drawn = [], False
    epoch_ticks = set()
    fig, ax = plt.subplots(figsize=(9, 5))
    for result in results:
        path = log_dir / f"{result['key']}_log.json"
        if not path.is_file():
            notes.append(f"{result['key']}：无训练日志，未绘制曲线。")
            continue
        with path.open(encoding="utf-8") as file:
            log = json.load(file)
        if not isinstance(log, dict) or log.get("config") != result["training_config"]:
            notes.append(f"{result['key']}：日志配置与 checkpoint 不一致，未绘制曲线。")
            continue
        epochs = [e for e in log.get("epochs", []) if "val_f1_pos" in e]
        if not epochs:
            notes.append(f"{result['key']}：日志无正例 F1，未绘制曲线。")
            continue
        ax.plot([e["epoch"] for e in epochs], [e["val_f1_pos"] for e in epochs], marker="o", label=result["key"], color=result["color"])
        epoch_ticks.update(e["epoch"] for e in epochs)
        drawn = True
    if drawn:
        ax.set_xlabel("Epoch")
        ax.set_xticks(sorted(epoch_ticks))
        ax.set_ylabel("Validation F1 (positive)")
        ax.set_title("Training validation curves (not final-test scores)")
        ax.legend()
        fig.tight_layout()
        fig.savefig(save_path, dpi=150)
    plt.close(fig)
    return drawn, notes


def fmt(value):
    return "—" if value is None else f"{value:.4f}"


def build_report(results, metadata, output_dir, has_curves, notes):
    ordered = sorted(results, key=lambda r: r["f1_pos"], reverse=True)
    lines = ["# 文本匹配四模型基础实验对比", "", "本报告由已有 checkpoint 的真实评估结果生成，不重新训练或搜索阈值。", "",
             f"- 评估数据：`{metadata['data_path']}`，共 {metadata['n_samples']:,} 条。",
             f"- 数据 SHA-256：`{metadata['data_sha256']}`。",
             "- 双塔使用 checkpoint 阈值（缺失时回退 0.5），交叉塔使用 argmax。",
             "- 每个方法仅一个 checkpoint；bootstrap 区间仅反映评估样本波动，不代表多种子训练方差。"]
    if metadata["split"] == "validation":
        lines.append("- **这是验证集/选模集对比，不是独立测试集成绩。**")
    else:
        lines.append("- 当前输入名为 test；是否与训练/选模数据独立仍需核对数据划分记录，本脚本不证明独立性。")
    if metadata["missing_methods"]:
        lines.append(f"- **部分模型报告**：缺失 {', '.join(metadata['missing_methods'])}。")
    lines += ["", "## 指标对比", "", "按正例 F1 排序。AP 对应指标文件中的 pr_auc。", "",
              "| 方法 | Accuracy | P(pos) | R(pos) | F1(pos) | Macro-F1 | Weighted-F1 | ROC-AUC | AP |",
              "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in ordered:
        lines.append(f"| {r['key']} | " + " | ".join(fmt(r.get(k)) for k in METRIC_KEYS) + " |")
    lines += ["", "## 权重与训练配置", "", "| 方法 | 层数 | 训练轮数 | 最佳轮次 | Seed | 评估长度 | 决策规则 |", "|---|---:|---:|---:|---:|---:|---|"]
    for r in results:
        c = r["training_config"]
        decision = f"cosine ≥ {fmt(r.get('threshold'))} ({r['threshold_source']})" if r["type"] == "biencoder" else "argmax"
        lines.append(f"| {r['key']} | {c.get('num_hidden_layers', '—')} | {c.get('epochs', '—')} | {r['best_epoch']} | {c.get('seed', '—')} | {r['eval_max_length']} | {decision} |")
    lines += ["", "完整训练配置、权重路径及实际 BERT 路径保存在各模型 metrics.json 中。配置相同也不代表日志和权重来源已被严格证明。", "", "## 观察与限制", ""]
    best = ordered[0]["f1_pos"]
    winners = [r["key"] for r in ordered if r["f1_pos"] == best]
    lines.append(f"- 本次正例 F1 最高：{', '.join(winners)}（{best:.4f}）。")
    if len(ordered) > 1:
        lines.append(f"- 排名前两项相差 {(ordered[0]['f1_pos'] - ordered[1]['f1_pos']) * 100:.2f} 个百分点；不据此声称统计显著。")
    for r in ordered:
        ci = r.get("confidence_intervals", {}).get("f1_pos")
        if ci:
            lines.append(f"- {r['key']} 正例 F1 的 95% bootstrap 区间：[{ci['low']:.4f}, {ci['high']:.4f}]。")
    lines += ["- 输入样本范围、损失、类别权重、池化和序列长度可能不同，不能将全部差异归因于单一损失或架构。",
              "- 不同损失的数值尺度不同，不直接比较训练 loss 大小；本报告未测量部署延迟或检索效果。", "",
              "## 图表", "", "![指标对比](figures/method_comparison_bar.png)", "", "![混淆矩阵](figures/confusion_matrices.png)"]
    if has_curves:
        lines += ["", "![验证集曲线](figures/validation_f1_curves.png)"]
    for note in notes:
        lines += ["", note]
    lines += ["", "## 结果文件", "", "[汇总 JSON](method_comparison.json) · [汇总 CSV](method_comparison.csv)", ""]
    for r in results:
        key = r["key"]
        lines.append(f"- {key}：[指标]({key}/metrics.json) · [逐样本预测]({key}/predictions.jsonl)")
        if r["type"] == "biencoder":
            lines.append(f"  · [相似度分布]({key}/similarity_distribution.png)")
    path = output_dir / "COMPARISON_REPORT.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def main():
    args = parse_args()
    rows, missing = validate_inputs(args)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = args.output_dir / args.split
    figures = output_dir / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    print(f"设备: {device}；数据: {args.split} / {len(rows):,} 条；输出: {output_dir}", flush=True)
    results = []
    for method in METHODS:
        if method["key"] in missing:
            print(f"跳过缺失模型：{method['key']}")
            continue
        print(f"\n评估 {method['key']}…", flush=True)
        result = load_and_eval(method, args, device, rows)
        save_model_result(result, rows, output_dir, args.split)
        results.append(result)
    data_path = args.data_dir / f"{args.split}.jsonl"
    metadata = {"split": args.split, "data_path": str(data_path.resolve()), "data_sha256": sha256_file(data_path),
                "n_samples": len(rows), "missing_methods": missing, "bootstrap_samples": args.bootstrap_samples,
                "bootstrap_seed": args.seed, "environment": environment_snapshot()}
    save_json({"metadata": metadata, "results": [compact_result(r) for r in results]}, output_dir / "method_comparison.json")
    with (output_dir / "method_comparison.csv").open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["method", *METRIC_KEYS])
        writer.writeheader()
        for r in results:
            writer.writerow({"method": r["key"], **{k: r.get(k) for k in METRIC_KEYS}})
    plot_comparison_bar(results, figures / "method_comparison_bar.png")
    plot_confusions(results, figures / "confusion_matrices.png")
    has_curves, notes = plot_training_curves(results, args.log_dir or args.ckpt_dir.parent / "logs", figures / "validation_f1_curves.png")
    report = build_report(results, metadata, output_dir, has_curves, notes)
    print("\n方法                       Accuracy  F1(pos)  Macro-F1")
    for r in sorted(results, key=lambda r: r["f1_pos"], reverse=True):
        print(f"{r['key']:<26} {r['accuracy']:.4f}    {r['f1_pos']:.4f}   {r['f1_macro']:.4f}")
    print(f"\n对比完成：{len(results)} 个模型。报告：{report.resolve()}", flush=True)


if __name__ == "__main__":
    main()
