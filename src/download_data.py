"""
下载并标准化原始句对数据；本脚本不是可信实验划分入口。

用途与流程位置：
    首次缺少原始数据时使用；已有数据无需重复下载。
    完整入口依次下载 AFQMC（clue/clue 的 afqmc 配置）、LCQMC（C-MTEB/LCQMC）、
    BQ（FinanceMTEB/bq_corpus）；BQ 从源 test 打乱后按约 8:1:1 划分。

输入与输出：
    通过 datasets 访问远程数据或缓存；默认使用 Hugging Face，可通过 HF_ENDPOINT 覆盖。
    写入项目 data 下各数据集目录，字段为 sentence1、sentence2、label。
    AFQMC 原官方 test 标签为 -1，属于无标签数据，不能计算分类指标。
    下载后由 prepare_experiment_data.py 构建 train/calibration/final-test 协议。

使用方式（在项目根目录执行，Bash）：
    python src/download_data.py --dataset afqmc

注意事项：
    不传 --dataset 时下载全部三个数据集；下载会覆盖对应目录中的同名文件。
    需要网络或已有缓存；下载成功不代表已完成泄漏审计或最终评估协议准备。

说明核对：2026-09-28；完整流程、前置条件及自定义模型路径见 USAGE_GUIDE.md。
"""

import argparse
import json
import os
import random

from datasets import load_dataset

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.normpath(os.path.join(SRC_DIR, "..", "data"))

random.seed(42)


# ── 工具函数 ──────────────────────────────────────────────────────────────

def save_jsonl(rows, path):
    """逐行以 UTF-8 保存已经标准化的样本。

    用途：
        逐行以 UTF-8 保存已经标准化的样本。

    使用场景与调用位置：
        在 src/download_data.py 的“下载 AFQMC、LCQMC、BQ 并统一为句对 JSONL”流程中使用。 本文件直接调用位置：download_afqmc()、download_bq()、download_lcqmc()。

    参数：
        rows: 原始记录列表；句对通常包含 sentence1、sentence2、label。
        path: 待读取或写入的文件路径，具体模式见用途与注意事项。

    返回值：
        None；写入 path 指定的 JSONL 文件。

    注意事项：
        覆盖同名文件；父目录由下载入口创建，中文不转义。
    """
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def normalize_row(row):
    """兼容不同数据源的文本和标签字段。

    用途：
        兼容不同数据源的文本和标签字段。

    使用场景与调用位置：
        在 src/download_data.py 的“下载 AFQMC、LCQMC、BQ 并统一为句对 JSONL”流程中使用。 本文件直接调用位置：download_afqmc()、download_bq()、download_lcqmc()。

    参数：
        row: 一个来源数据记录，字段按当前数据源约定。

    返回值：
        含 sentence1、sentence2、label 的字典。

    注意事项：
        按 sentence1/text1/query 和 sentence2/text2/candidate 顺序取文本；label 优先于 score，缺失标签默认 0；不会把 -1 自动修复为有效标签。
    """
    s1 = (row.get("sentence1") or row.get("text1") or row.get("query") or "")
    s2 = (row.get("sentence2") or row.get("text2") or row.get("candidate") or "")
    # score 字段在 C-MTEB / FinanceMTEB 中就是 0/1
    label = int(row.get("label", row.get("score", 0)))
    return {"sentence1": str(s1), "sentence2": str(s2), "label": label}


def print_stats(rows, split, path):
    """打印一次下载的样本数和类别概况。

    用途：
        打印一次下载的样本数和类别概况。

    使用场景与调用位置：
        在 src/download_data.py 的“下载 AFQMC、LCQMC、BQ 并统一为句对 JSONL”流程中使用。 本文件直接调用位置：download_afqmc()、download_bq()、download_lcqmc()。

    参数：
        rows: 原始记录列表；句对通常包含 sentence1、sentence2、label。
        split: 数据划分名，例如 train、validation 或 test。
        path: 待读取或写入的文件路径，具体模式见用途与注意事项。

    返回值：
        None；写入控制台。

    注意事项：
        标签 1 和 0 分别计为正例和负例。官方 test 的 -1 单独计入未标注或其他标签。
    """
    pos = sum(1 for r in rows if r["label"] == 1)
    neg = sum(1 for r in rows if r["label"] == 0)
    other = len(rows) - pos - neg
    print(
        f"  {split:12s}: {len(rows):>7,} 条  正样本 {pos:>6,}  负样本 {neg:>6,}"
        f"  未标注或其他 {other:>6,}  -> {path}"
    )


def preview(path, n=3):
    """展示下载文件前 n 条句对供人工核对。

    用途：
        展示下载文件前 n 条句对供人工核对。

    使用场景与调用位置：
        在 src/download_data.py 的“下载 AFQMC、LCQMC、BQ 并统一为句对 JSONL”流程中使用。 本文件直接调用位置：download_afqmc()、download_bq()、download_lcqmc()。

    参数：
        path: 待读取或写入的文件路径，具体模式见用途与注意事项。
        n: 最多展示的记录条数。

    返回值：
        None；打印句对和标签提示。

    注意事项：
        只读取前 n 行；显示逻辑把非 1 标签显示为不相似，不能替代标签合法性检查。
    """
    print(f"\n  [预览 前 {n} 条]")
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= n:
                break
            r = json.loads(line)
            tag = "✓ 相似" if r["label"] == 1 else "✗ 不相似"
            print(f"    [{tag}] {r['sentence1']!r}  ||  {r['sentence2']!r}")


# ── AFQMC ─────────────────────────────────────────────────────────────────

def download_afqmc():
    """从 clue/clue 的 afqmc 配置获取数据并保存统一字段。

    用途：
        从 clue/clue 的 afqmc 配置获取数据并保存统一字段。

    使用场景与调用位置：
        在 src/download_data.py 的“下载 AFQMC、LCQMC、BQ 并统一为句对 JSONL”流程中使用。 本文件直接调用位置：main()。

    参数：
        无显式参数；所需配置由内部逻辑、模块常量或命令行提供。

    返回值：
        None；生成对应 data 子目录下的 JSONL，并打印统计和预览。

    注意事项：
        需要网络或可用的数据集缓存，会覆盖同名数据文件。保留源 train/validation/test 划分，缺失的 split 会跳过。
    """
    print(f"\n{'='*55}")
    print("下载 AFQMC（clue/clue 的 afqmc 配置）...")
    out_dir = os.path.join(DATA_DIR, "afqmc")
    os.makedirs(out_dir, exist_ok=True)

    ds = load_dataset("clue/clue", "afqmc")

    split_map = {"train": "train", "validation": "validation", "test": "test"}
    for split_out, split_in in split_map.items():
        if split_in not in ds:
            print(f"  [SKIP] {split_in} 不存在")
            continue
        rows = [normalize_row(r) for r in ds[split_in]]
        out_path = os.path.join(out_dir, f"{split_out}.jsonl")
        save_jsonl(rows, out_path)
        print_stats(rows, split_out, out_path)

    preview(os.path.join(out_dir, "train.jsonl"))


# ── LCQMC ─────────────────────────────────────────────────────────────────

def download_lcqmc():
    """从 C-MTEB/LCQMC 获取数据并保存统一字段。

    用途：
        从 C-MTEB/LCQMC 获取数据并保存统一字段。

    使用场景与调用位置：
        在 src/download_data.py 的“下载 AFQMC、LCQMC、BQ 并统一为句对 JSONL”流程中使用。 本文件直接调用位置：main()。

    参数：
        无显式参数；所需配置由内部逻辑、模块常量或命令行提供。

    返回值：
        None；生成对应 data 子目录下的 JSONL，并打印统计和预览。

    注意事项：
        需要网络或可用的数据集缓存，会覆盖同名数据文件。保留源 train/validation/test 划分，缺失的 split 会跳过。
    """
    print(f"\n{'='*55}")
    print("下载 LCQMC（C-MTEB/LCQMC）...")
    out_dir = os.path.join(DATA_DIR, "lcqmc")
    os.makedirs(out_dir, exist_ok=True)

    ds = load_dataset("C-MTEB/LCQMC")

    split_map = {"train": "train", "validation": "validation", "test": "test"}
    for split_out, split_in in split_map.items():
        if split_in not in ds:
            print(f"  [SKIP] {split_in} 不存在")
            continue
        rows = [normalize_row(r) for r in ds[split_in]]
        out_path = os.path.join(out_dir, f"{split_out}.jsonl")
        save_jsonl(rows, out_path)
        print_stats(rows, split_out, out_path)

    preview(os.path.join(out_dir, "train.jsonl"))


# ── BQ Corpus ─────────────────────────────────────────────────────────────

def download_bq():
    """从 FinanceMTEB/bq_corpus 获取数据并保存统一字段。

    用途：
        从 FinanceMTEB/bq_corpus 获取数据并保存统一字段。

    使用场景与调用位置：
        在 src/download_data.py 的“下载 AFQMC、LCQMC、BQ 并统一为句对 JSONL”流程中使用。 本文件直接调用位置：main()。

    参数：
        无显式参数；所需配置由内部逻辑、模块常量或命令行提供。

    返回值：
        None；生成对应 data 子目录下的 JSONL，并打印统计和预览。

    注意事项：
        需要网络或可用的数据集缓存，会覆盖同名数据文件。读取源 test 并打乱后按约 8:1:1 划分，随机状态来自模块级 random.seed(42)。
    """
    print(f"\n{'='*55}")
    print("下载 BQ Corpus（FinanceMTEB/bq_corpus）...")
    print("  注：原始仓库无 train/val split，按 8:1:1 切分")
    out_dir = os.path.join(DATA_DIR, "bq_corpus")
    os.makedirs(out_dir, exist_ok=True)

    ds = load_dataset("FinanceMTEB/bq_corpus", split="test")
    all_rows = [normalize_row(r) for r in ds]
    random.shuffle(all_rows)

    n = len(all_rows)
    n_val = n // 10
    n_test = n // 10
    splits = {
        "train": all_rows[n_val + n_test:],
        "validation": all_rows[:n_val],
        "test": all_rows[n_val: n_val + n_test],
    }
    for split, rows in splits.items():
        out_path = os.path.join(out_dir, f"{split}.jsonl")
        save_jsonl(rows, out_path)
        print_stats(rows, split, out_path)

    preview(os.path.join(out_dir, "train.jsonl"))


# ── 主流程 ────────────────────────────────────────────────────────────────

def main():
    """按选项下载一个数据集或依次下载全部三个数据集。

    用途：
        下载选定数据集并统一为句对 JSONL。

    使用场景与调用位置：
        直接运行此文件时由 if __name__ == "__main__" 入口调用；也可显式调用 main()，此时仍读取进程命令行。

    参数：
        命令行 --dataset 可选 afqmc、lcqmc、bq_corpus 或 all（默认）。

    返回值：
        None；产物及副作用见注意事项，配置由命令行读取。

    注意事项：
        按 AFQMC→LCQMC→BQ 顺序下载并写入原始 data 目录，需要网络，可能覆盖同名文件。
    """
    downloaders = {
        "afqmc": download_afqmc,
        "lcqmc": download_lcqmc,
        "bq_corpus": download_bq,
    }
    parser = argparse.ArgumentParser(description="下载并标准化中文句对数据")
    parser.add_argument("--dataset", choices=("all", *downloaders), default="all")
    args = parser.parse_args()

    print(f"HF_ENDPOINT: {os.getenv('HF_ENDPOINT', 'https://huggingface.co')}")
    print(f"数据保存目录: {DATA_DIR}")

    selected = downloaders if args.dataset == "all" else (args.dataset,)
    for name in selected:
        downloaders[name]()

    print("\n\n全部完成。")


if __name__ == "__main__":
    main()
