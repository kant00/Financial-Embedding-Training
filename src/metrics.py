"""
公共二分类指标：统一计分、calibration 阈值搜索与 bootstrap 区间。

用途与调用位置：
    由基线、训练评估、SFT 评估等模块导入，避免不同入口混淆指标口径。
    本模块不启动训练或读取项目数据，没有独立命令行流程。

计分约定：
    分别输出正例、Macro、Weighted F1，并计算适用的分类/排序指标。
    pr_auc 使用 Average Precision（AP），不是手动梯形积分面积。
    阈值搜索使用去重分数的中点与边界，默认优化 f1_pos；同分优先更高阈值。
    bootstrap 在给定预测上重采样估计区间，不等同于重新训练或跨种子波动。

注意事项：
    最终测试必须复用 calibration 决定的阈值，不在 final test 搜索。
    单一类别或缺少连续分数时，部分指标可能不可用，应保留其缺失语义。
    使用流程和指标解释见 USAGE_GUIDE.md。

说明核对：2026-09-28；完整流程、前置条件及自定义模型路径见 USAGE_GUIDE.md。
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
)


def compute_binary_metrics(labels, predictions, scores=None) -> dict:
    """以标签 1 为相似类计算统一二分类指标。

    用途：
        以标签 1 为相似类计算统一二分类指标。

    使用场景与调用位置：
        在 src/metrics.py 的“统一二分类指标、阈值选择、置信区间和多种子汇总”流程中使用。 本文件直接调用位置：bootstrap_confidence_intervals()、find_best_threshold()。 跨文件调用：src/evaluate.py:eval_biencoder()、src/evaluate.py:eval_crossencoder()、src/run_baselines.py:evaluate_and_save()、src_llm/evaluate_sft.py:main()。

    参数：
        labels: 一维真实标签序列，0 为不相似、1 为相似。
        predictions: 与 labels 等长的一维 0/1 预测序列。
        scores: 与 labels 顺序一致的连续正类分数；None 表示不计算排序型 AUC。

    返回值：
        含 accuracy、两类 precision/recall/f1/support、f1_macro、f1_weighted、confusion_matrix 的字典；有 scores 时增加 roc_auc/pr_auc。

    注意事项：
        评估、基线、SFT 和 bootstrap 共用。labels/predictions 必须非空等长且为 0/1，否则 ValueError；f1 始终是 weighted-F1 别名。pr_auc 实际为 average_precision_score 的 AP，不是梯形 PR 积分；单类样本的两种 AUC 返回 None。
    """
    labels = np.asarray(labels, dtype=np.int64)
    predictions = np.asarray(predictions, dtype=np.int64)
    if labels.shape != predictions.shape or labels.size == 0:
        raise ValueError("labels 与 predictions 必须等长且非空")
    if not set(np.unique(labels)).issubset({0, 1}):
        raise ValueError("二分类评估只接受标签 0/1；请勿使用 AFQMC 官方无标签 test")
    if not set(np.unique(predictions)).issubset({0, 1}):
        raise ValueError("predictions 只接受 0/1")

    precision, recall, f1_per_class, support = precision_recall_fscore_support(
        labels,
        predictions,
        labels=[0, 1],
        zero_division=0,
    )
    tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()
    metrics = {
        "n_samples": int(labels.size),
        "accuracy": float(accuracy_score(labels, predictions)),
        "precision_neg": float(precision[0]),
        "recall_neg": float(recall[0]),
        "f1_neg": float(f1_per_class[0]),
        "precision_pos": float(precision[1]),
        "recall_pos": float(recall[1]),
        "f1_pos": float(f1_per_class[1]),
        "f1_macro": float(f1_score(labels, predictions, average="macro", zero_division=0)),
        "f1_weighted": float(
            f1_score(labels, predictions, average="weighted", zero_division=0)
        ),
        "support_neg": int(support[0]),
        "support_pos": int(support[1]),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }
    metrics["f1"] = metrics["f1_weighted"]

    if scores is not None:
        scores = np.asarray(scores, dtype=np.float64)
        if scores.shape != labels.shape:
            raise ValueError("scores 与 labels 必须等长")
        if np.unique(labels).size == 2:
            metrics["roc_auc"] = float(roc_auc_score(labels, scores))
            metrics["pr_auc"] = float(average_precision_score(labels, scores))
        else:
            metrics["roc_auc"] = None
            metrics["pr_auc"] = None
    return metrics


def find_best_threshold(scores, labels, metric: str = "f1_pos") -> tuple[float, float]:
    """在校准样本中寻找目标指标最高的 score>=threshold 决策边界。

    用途：
        在校准样本中寻找目标指标最高的 score>=threshold 决策边界。

    使用场景与调用位置：
        在 src/metrics.py 的“统一二分类指标、阈值选择、置信区间和多种子汇总”流程中使用。 跨文件调用：src/evaluate.py:_find_best_threshold()、src/evaluate.py:eval_biencoder()、src/run_baselines.py:main()。

    参数：
        scores: 与 labels 等长的非空连续分数序列，不能为 None；分数越大越倾向正例。
        labels: 一维真实标签序列，0 为不相似、1 为相似。
        metric: 阈值优化的指标字段名，例如 f1_pos。

    返回值：
        (最佳阈值 float, 对应指标 float)。

    注意事项：
        使用去重分数的中点，以及通过 nextafter 扩展的最小/最大边界；覆盖全判正和全判负，不限固定 [0,1] 网格。同分取更高阈值；默认优化 f1_pos。逐阈值重算指标，规模大时耗时明显；禁止使用 final test 调参。
    """
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    if scores.size == 0 or scores.shape != labels.shape:
        raise ValueError("scores 与 labels 必须等长且非空")

    unique_scores = np.unique(scores)
    lower = np.nextafter(unique_scores[0], -np.inf)
    upper = np.nextafter(unique_scores[-1], np.inf)
    if unique_scores.size == 1:
        candidates = np.array([lower, upper], dtype=np.float64)
    else:
        midpoints = (unique_scores[:-1] + unique_scores[1:]) / 2.0
        candidates = np.concatenate(([lower], midpoints, [upper]))

    best_threshold, best_value = 0.5, -1.0
    for threshold in candidates:
        predictions = (scores >= threshold).astype(np.int64)
        value = compute_binary_metrics(labels, predictions)[metric]
        # 同分时偏向更高阈值，减少误召回，行为稳定且便于解释。
        if value > best_value or (value == best_value and threshold > best_threshold):
            best_threshold, best_value = float(threshold), float(value)
    return best_threshold, best_value


def bootstrap_confidence_intervals(
    labels,
    predictions,
    scores=None,
    n_bootstrap: int = 1000,
    seed: int = 42,
    confidence: float = 0.95,
) -> dict:
    """固定预测后用有放回样本重采样估计指标区间。

    用途：
        固定预测后用有放回样本重采样估计指标区间。

    使用场景与调用位置：
        在 src/metrics.py 的“统一二分类指标、阈值选择、置信区间和多种子汇总”流程中使用。 跨文件调用：src/evaluate.py:main()、src/run_baselines.py:evaluate_and_save()、src_llm/evaluate_sft.py:main()。

    参数：
        labels: 一维真实标签序列，0 为不相似、1 为相似。
        predictions: 与 labels 等长的一维 0/1 预测序列。
        scores: 与 labels 顺序一致的连续正类分数；None 表示不计算排序型 AUC。
        n_bootstrap: bootstrap 重采样次数，必须大于 0。
        seed: 可复现实验的整数随机种子。
        confidence: 置信水平，例如 0.95；应在 0 与 1 之间。

    返回值：
        指标名映射到 low/high/confidence 的字典。

    注意事项：
        基线及最终评估后调用；不重训、不重新选阈值，不涵盖训练随机性。n_bootstrap 需正数，confidence 在 (0,1)；若 scores 的每次重采样都只有单类，AUC 收集为空，quantile 无法计算。
    """
    labels = np.asarray(labels)
    predictions = np.asarray(predictions)
    scores_array = None if scores is None else np.asarray(scores)
    rng = np.random.default_rng(seed)
    collected: dict[str, list[float]] = {
        "accuracy": [],
        "f1_pos": [],
        "f1_macro": [],
        "f1_weighted": [],
    }
    if scores is not None:
        collected.update({"roc_auc": [], "pr_auc": []})

    for _ in range(n_bootstrap):
        indices = rng.integers(0, labels.size, labels.size)
        sampled_scores = None if scores_array is None else scores_array[indices]
        sample_metrics = compute_binary_metrics(
            labels[indices], predictions[indices], sampled_scores
        )
        for key in collected:
            value = sample_metrics.get(key)
            if value is not None:
                collected[key].append(value)

    alpha = (1.0 - confidence) / 2.0
    intervals = {}
    for key, values in collected.items():
        intervals[key] = {
            "low": float(np.quantile(values, alpha)),
            "high": float(np.quantile(values, 1.0 - alpha)),
            "confidence": confidence,
        }
    return intervals


def aggregate_seed_metrics(records: list[dict], keys=None) -> dict:
    """按多个独立训练 seed 汇总指标均值和波动。

    用途：
        按多个独立训练 seed 汇总指标均值和波动。

    使用场景与调用位置：
        在 src/metrics.py 的“统一二分类指标、阈值选择、置信区间和多种子汇总”流程中使用。 跨文件调用：src/run_experiment_suite.py:main()。

    参数：
        records: 每个 seed 对应一个指标字典的列表。
        keys: 要汇总的指标名；省略时采用核心指标集合。

    返回值：
        含 n_seeds 与各指标 mean/std/values 的字典。

    注意事项：
        run_experiment_suite 全部训练评估后调用；跳过缺失或 None 指标，std 使用 ddof=1；单条记录时写 0.0 只是格式约定，不证明稳定性。
    """
    keys = keys or ("accuracy", "f1_pos", "f1_macro", "f1_weighted", "roc_auc", "pr_auc")
    summary = {"n_seeds": len(records)}
    for key in keys:
        values = [record[key] for record in records if record.get(key) is not None]
        if values:
            summary[key] = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
                "values": [float(value) for value in values],
            }
    return summary
