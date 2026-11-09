"""
可信基线入口：多数类预测与字符 TF-IDF 句对相似度。

用途与流程位置：
    固定数据协议后运行，建立无需神经网络训练的参考结果。
    TF-IDF 仅在 train 拟合词表，在 calibration 选择正例 F1 阈值，再评估 final test。

输入与输出：
    --data_dir 默认指向 data/experiments/afqmc_seed42。
    --output_dir 保存各方法的 metrics、predictions 及 baseline_summary.json。
    统一计算分类指标和 bootstrap 区间，供 build_experiment_report.py 汇总。

使用方式（在项目根目录执行，Bash）：
    python src/run_baselines.py --data_dir data/experiments/afqmc_seed42 --output_dir outputs/credible_baselines --seed 42

注意事项：
    脚本会直接运行最终评估，基线方案须预先固定，不能根据 final-test 分数反复调整。
    类别不平衡时不要仅报告 Accuracy；同时查看正例 Precision/Recall/F1。

说明核对：2026-09-28；完整流程、前置条件及自定义模型路径见 USAGE_GUIDE.md。
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

# Windows 的 Anaconda 环境可能同时加载 sklearn 与 PyTorch 自带 OpenMP；与原项目
# 其他入口保持一致，避免环境快照阶段重复初始化导致基线任务中断。
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from experiment_utils import environment_snapshot, save_json, write_jsonl
from metrics import (
    bootstrap_confidence_intervals,
    compute_binary_metrics,
    find_best_threshold,
)


ROOT = Path(__file__).parent.parent
DEFAULT_DATA = ROOT / "data" / "experiments" / "afqmc_seed42"
DEFAULT_OUTPUT = ROOT / "outputs" / "credible_baselines"


def load_jsonl(path: Path) -> list[dict]:
    """读取 UTF-8 JSONL 并按文件顺序还原记录。

    用途：
        读取 UTF-8 JSONL 并按文件顺序还原记录。

    使用场景与调用位置：
        在 src/run_baselines.py 的“在固定评估清单上运行多数类和字符 TF-IDF 基线”流程中使用。 本文件直接调用位置：main()。

    参数：
        path: 待读取或写入的文件路径，具体模式见用途与注意事项。

    返回值：
        list[dict]；每个非空行对应一条记录。

    注意事项：
        用于在固定评估清单上运行多数类和字符 TF-IDF 基线阶段的数据读取；忽略空行，不执行字段、标签或重复样本校验；缺失文件和非法 JSON 异常直接上抛，整个文件驻留内存。
    """
    with open(path, encoding="utf-8") as file_obj:
        return [json.loads(line) for line in file_obj if line.strip()]


def pair_scores(vectorizer, rows: list[dict]) -> np.ndarray:
    """计算每个句对的字符 TF-IDF 余弦分数。

    用途：
        计算每个句对的字符 TF-IDF 余弦分数。

    使用场景与调用位置：
        在 src/run_baselines.py 的“在固定评估清单上运行多数类和字符 TF-IDF 基线”流程中使用。 本文件直接调用位置：main()。

    参数：
        vectorizer: 已经在训练语料上 fit 的 TfidfVectorizer。
        rows: 原始记录列表；句对通常包含 sentence1、sentence2、label。

    返回值：
        按输入顺序的 [N] NumPy 分数数组。

    注意事项：
        vectorizer 必须先仅在 train 上 fit，calibration/test 只能 transform；默认 L2 归一化使稀疏逐行点积等于余弦，空词向量得到 0 分。
    """
    left = vectorizer.transform([row["sentence1"] for row in rows])
    right = vectorizer.transform([row["sentence2"] for row in rows])
    # TfidfVectorizer 默认 L2 归一化，逐行点积就是余弦相似度。
    return np.asarray(left.multiply(right).sum(axis=1)).ravel()


def evaluate_and_save(name, rows, predictions, scores, output_dir, seed, extra=None):
    """计算一个基线的指标和区间并保存每条预测。

    用途：
        计算一个基线的指标和区间并保存每条预测。

    使用场景与调用位置：
        在 src/run_baselines.py 的“在固定评估清单上运行多数类和字符 TF-IDF 基线”流程中使用。 本文件直接调用位置：main()。

    参数：
        name: 当前数据划分、模型或输出结果的显示名称。
        rows: 原始记录列表；句对通常包含 sentence1、sentence2、label。
        predictions: 与 labels 等长的一维 0/1 预测序列。
        scores: 与 labels 顺序一致的连续正类分数；None 表示不计算排序型 AUC。
        output_dir: 输出目录，保存图像或实验结果。
        seed: 可复现实验的整数随机种子。
        extra: 可选附加元数据字典，会合并进结果字典。

    返回值：
        含模型名称、环境、指标、区间及 extra 扩展项的字典。

    注意事项：
        main 对多数类和 TF-IDF 分别调用；写 name_metrics.json 和 name_predictions.jsonl；默认执行 1000 次 bootstrap，scores=None 时不计算 AUC。
    """
    labels = np.asarray([int(row["label"]) for row in rows])
    metrics = compute_binary_metrics(labels, predictions, scores)
    metrics["confidence_intervals"] = bootstrap_confidence_intervals(
        labels, predictions, scores, seed=seed
    )
    metrics["model"] = name
    metrics["environment"] = environment_snapshot(include_torch=False)
    if extra:
        metrics.update(extra)

    prediction_rows = []
    for index, (row, prediction) in enumerate(zip(rows, predictions)):
        prediction_rows.append(
            {
                "sample_id": row.get("sample_id", f"row-{index:05d}"),
                "label": int(row["label"]),
                "prediction": int(prediction),
                "score": None if scores is None else float(scores[index]),
                "sentence1": row["sentence1"],
                "sentence2": row["sentence2"],
            }
        )
    save_json(metrics, output_dir / f"{name}_metrics.json")
    write_jsonl(prediction_rows, output_dir / f"{name}_predictions.jsonl")
    return metrics


def parse_args():
    """解析在固定评估清单上运行多数类和字符 TF-IDF 基线所需的命令行配置。

    用途：
        解析在固定评估清单上运行多数类和字符 TF-IDF 基线所需的命令行配置。

    使用场景与调用位置：
        本文件 main() 的启动配置阶段调用；独立运行脚本时首先经过此处。 本文件直接调用位置：main()。

    参数：
        无显式参数；所需配置由内部逻辑、模块常量或命令行提供。

    返回值：
        argparse.Namespace；成员名对应下方 add_argument 定义。

    注意事项：
        由本文件 main 在任何主要工作开始前调用；读取 sys.argv，处理默认值、类型和 choices；--help 或不合法参数通过 argparse 触发 SystemExit，不启动训练或推理。
    """
    parser = argparse.ArgumentParser(description="可信文本匹配基线")
    parser.add_argument("--data_dir", default=str(DEFAULT_DATA))
    parser.add_argument("--output_dir", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min_df", type=int, default=2)
    parser.add_argument("--max_features", type=int, default=200000)
    return parser.parse_args()


def main():
    """组织本脚本的完整流程：在固定评估清单上运行多数类和字符 TF-IDF 基线。

    用途：
        组织本脚本的完整流程：在固定评估清单上运行多数类和字符 TF-IDF 基线。

    使用场景与调用位置：
        直接运行此文件时由 if __name__ == "__main__" 入口调用；也可显式调用 main()，此时仍读取进程命令行。

    参数：
        无显式参数；所需配置由内部逻辑、模块常量或命令行提供。

    返回值：
        None；产物及副作用见注意事项，配置由命令行读取。

    注意事项：
        TF-IDF 仅 fit train，阈值仅用 calibration 正例 F1；两种基线都在 evaluation_manifest 上评估，保存区间、环境、明细和汇总。
    """
    args = parse_args()
    data_dir, output_dir = Path(args.data_dir), Path(args.output_dir)
    train_rows = load_jsonl(data_dir / "train.jsonl")
    calibration_rows = load_jsonl(data_dir / "calibration.jsonl")
    final_rows = load_jsonl(data_dir / "evaluation_manifest.jsonl")

    train_labels = np.asarray([int(row["label"]) for row in train_rows])
    majority_label = int(np.bincount(train_labels).argmax())
    majority_predictions = np.full(len(final_rows), majority_label, dtype=np.int64)
    majority_metrics = evaluate_and_save(
        "majority",
        final_rows,
        majority_predictions,
        None,
        output_dir,
        args.seed,
        {"majority_label": majority_label},
    )

    vectorizer = TfidfVectorizer(
        analyzer="char",
        ngram_range=(2, 4),
        min_df=args.min_df,
        max_features=args.max_features,
        sublinear_tf=True,
        dtype=np.float32,
    )
    # 只在训练句子上 fit，calibration/final test 均不可参与词表学习。
    vectorizer.fit(
        [row[key] for row in train_rows for key in ("sentence1", "sentence2")]
    )
    calibration_scores = pair_scores(vectorizer, calibration_rows)
    calibration_labels = [int(row["label"]) for row in calibration_rows]
    threshold, calibration_f1 = find_best_threshold(
        calibration_scores, calibration_labels, metric="f1_pos"
    )
    final_scores = pair_scores(vectorizer, final_rows)
    final_predictions = (final_scores >= threshold).astype(np.int64)
    tfidf_metrics = evaluate_and_save(
        "char_tfidf",
        final_rows,
        final_predictions,
        final_scores,
        output_dir,
        args.seed,
        {
            "threshold": threshold,
            "threshold_selected_on": "calibration",
            "calibration_f1_pos": calibration_f1,
            "vocabulary_size": len(vectorizer.vocabulary_),
        },
    )

    summary = {"majority": majority_metrics, "char_tfidf": tfidf_metrics}
    save_json(summary, output_dir / "baseline_summary.json")
    print(
        f"majority: acc={majority_metrics['accuracy']:.4f}, "
        f"f1_pos={majority_metrics['f1_pos']:.4f}"
    )
    print(
        f"char_tfidf: acc={tfidf_metrics['accuracy']:.4f}, "
        f"f1_pos={tfidf_metrics['f1_pos']:.4f}, threshold={threshold:.6f}"
    )


if __name__ == "__main__":
    main()
