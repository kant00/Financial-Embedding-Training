"""
从真实 JSON 产物生成 Markdown 实验报告，不手工填写指标。

用途与流程位置：
    基线、多种子套件与两阶段检索完成后运行；不负责启动缺失实验。
    读取 baseline_summary.json、multi_seed_summary.json、retrieval_metrics.json。

输入与输出：
    --baseline_dir、--suite_dir、--retrieval_dir 分别指向对应结果目录。
    --output 指定报告文件；分类表展示套件均值，标准差仍在套件 JSON 中。
    某输入缺失时省略相关结果或提示未运行，不会凭空生成分数。

使用方式（在项目根目录执行，Bash）：
    python src/build_experiment_report.py --baseline_dir outputs/credible_baselines --suite_dir outputs/experiment_suite --retrieval_dir outputs/retrieval --output outputs/EXPERIMENT_REPORT.md

注意事项：
    不自动汇总独立 evaluate 目录、多个套件、SFT 或在线 API 结果。
    Hard Triplet 套件需单独生成报告并注明实验身份；报告存在不代表所有实验已完成。
    同名 Markdown 会覆盖，最终验收清单见 USAGE_GUIDE.md 第 9 节。

说明核对：2026-09-28；完整流程、前置条件及自定义模型路径见 USAGE_GUIDE.md。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


ROOT = Path(__file__).parent.parent


def read_json(path):
    """读取已经落盘的实验 JSON。

    用途：
        读取已经落盘的实验 JSON。

    使用场景与调用位置：
        在 src/build_experiment_report.py 的“把基线、多种子和检索结果 JSON 转为 Markdown 报告”流程中使用。 本文件直接调用位置：main()。

    参数：
        path: 待读取或写入的文件路径，具体模式见用途与注意事项。

    返回值：
        json.load 解析得到的 Python 对象。

    注意事项：
        main 读取基线汇总、多种子汇总和检索结果；不验证 schema，格式错误会抛 JSONDecodeError。
    """
    with open(path, encoding="utf-8") as file_obj:
        return json.load(file_obj)


def metric(value):
    """把一个指标格式化成报告单元格。

    用途：
        把一个指标格式化成报告单元格。

    使用场景与调用位置：
        在 src/build_experiment_report.py 的“把基线、多种子和检索结果 JSON 转为 Markdown 报告”流程中使用。 本文件直接调用位置：main()。

    参数：
        value: 待转换、解析或格式化的输入值。

    返回值：
        None 转为破折号，否则为保留四位小数的字符串。

    注意事项：
        main 构造 Markdown 表格时调用；可转 float 的数值才有效，缺失项不会伪造为 0。
    """
    return "—" if value is None else f"{float(value):.4f}"


def parse_args():
    """解析把基线、多种子和检索结果 JSON 转为 Markdown 报告所需的命令行配置。

    用途：
        解析把基线、多种子和检索结果 JSON 转为 Markdown 报告所需的命令行配置。

    使用场景与调用位置：
        本文件 main() 的启动配置阶段调用；独立运行脚本时首先经过此处。 本文件直接调用位置：main()。

    参数：
        无显式参数；所需配置由内部逻辑、模块常量或命令行提供。

    返回值：
        argparse.Namespace；成员名对应下方 add_argument 定义。

    注意事项：
        由本文件 main 在任何主要工作开始前调用；读取 sys.argv，处理默认值、类型和 choices；--help 或不合法参数通过 argparse 触发 SystemExit，不启动训练或推理。
    """
    parser = argparse.ArgumentParser(description="生成可信实验 Markdown 报告")
    parser.add_argument("--baseline_dir", default=str(ROOT / "outputs" / "credible_baselines"))
    parser.add_argument("--suite_dir", default=str(ROOT / "outputs" / "experiment_suite"))
    parser.add_argument("--retrieval_dir", default=str(ROOT / "outputs" / "retrieval_eval"))
    parser.add_argument("--output", default=str(ROOT / "outputs" / "EXPERIMENT_REPORT.md"))
    return parser.parse_args()


def main():
    """组织本脚本的完整流程：把基线、多种子和检索结果 JSON 转为 Markdown 报告。

    用途：
        组织本脚本的完整流程：把基线、多种子和检索结果 JSON 转为 Markdown 报告。

    使用场景与调用位置：
        直接运行此文件时由 if __name__ == "__main__" 入口调用；也可显式调用 main()，此时仍读取进程命令行。

    参数：
        无显式参数；所需配置由内部逻辑、模块常量或命令行提供。

    返回值：
        None；产物及副作用见注意事项，配置由命令行读取。

    注意事项：
        只读取已经存在的结果文件；缺失时显示尚无结果，覆盖输出 Markdown。当前表格展示多种子均值，完整 std 和区间保留在 JSON。
    """
    args = parse_args()
    rows = []
    baseline_summary = Path(args.baseline_dir) / "baseline_summary.json"
    if baseline_summary.exists():
        for name, record in read_json(baseline_summary).items():
            rows.append((name, "single", record))

    suite_summary = Path(args.suite_dir) / "multi_seed_summary.json"
    if suite_summary.exists():
        suite = read_json(suite_summary)
        for name, record in suite.get("summary", {}).items():
            flattened = {
                key: value.get("mean") for key, value in record.items()
                if isinstance(value, dict) and "mean" in value
            }
            rows.append((name, f"{record.get('n_seeds', 0)} seeds mean", flattened))

    lines = [
        "# 可信实验报告",
        "",
        "> 本文件由 `src/build_experiment_report.py` 从 JSON 结果生成，请勿手工修改指标。",
        "",
        "## 二分类结果",
        "",
        "| 方法 | 汇总方式 | Accuracy | F1(pos) | Macro-F1 | Weighted-F1 | ROC-AUC | PR-AUC |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name, aggregation, record in rows:
        lines.append(
            f"| {name} | {aggregation} | {metric(record.get('accuracy'))} | "
            f"{metric(record.get('f1_pos'))} | {metric(record.get('f1_macro'))} | "
            f"{metric(record.get('f1_weighted'))} | {metric(record.get('roc_auc'))} | "
            f"{metric(record.get('pr_auc'))} |"
        )
    if not rows:
        lines.append("| 尚无已运行结果 | — | — | — | — | — | — | — |")

    retrieval_path = Path(args.retrieval_dir) / "retrieval_metrics.json"
    lines.extend(["", "## Recall–Rerank 结果", ""])
    if retrieval_path.exists():
        retrieval = read_json(retrieval_path)
        lines.extend([
            "| 阶段 | MRR | Recall@1 | Recall@5 | Recall@10 | NDCG@10 |",
            "|---|---:|---:|---:|---:|---:|",
        ])
        for stage in ("recall", "rerank"):
            record = retrieval.get(stage)
            if record:
                lines.append(
                    f"| {stage} | {metric(record.get('mrr'))} | "
                    f"{metric(record.get('recall@1'))} | {metric(record.get('recall@5'))} | "
                    f"{metric(record.get('recall@10'))} | {metric(record.get('ndcg@10'))} |"
                )
        lines.extend([
            "",
            f"- 候选规模：{retrieval['protocol']['n_candidates']}",
            f"- 查询规模：{retrieval['protocol']['n_queries']}",
            f"- 检索后端：{retrieval['backend']}",
            f"- 限制：{retrieval['protocol']['relevance_note']}",
        ])
    else:
        lines.append("尚未运行两阶段检索评估。")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as file_obj:
        file_obj.write("\n".join(lines) + "\n")
    print(f"实验报告已生成: {output_path}")


if __name__ == "__main__":
    main()
