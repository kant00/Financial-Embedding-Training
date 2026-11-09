"""
句对数据探索：类别、字符长度、token 长度和句长差分布。

用途与流程位置：
    数据协议生成后、训练前检查数据形态；由人工运行，不修改训练样本。
    长度设置和特征假设应依据 train/calibration，不用最终测试错误指导调参。

输入与输出：
    --data_dir 读取 train/validation/test.jsonl；正式流程传入生成的协议目录。
    --bert_path 指定 BertTokenizer；其分词不是“中文字节对编码”的同义描述。
    token 长度统计只使用部分训练样本，且不等于加入特殊 token 后的完整输入长度。
    --output_dir 保存标签、字符长度、token 长度和长度差 PNG；--skip_token 跳过 token 图。

使用方式（在项目根目录执行，Bash）：
    python src/explore_data.py --data_dir data/experiments/afqmc_seed42 --output_dir outputs/eda

注意事项：
    默认 data_dir 仍是旧 data/afqmc；原官方无标签 test 不适合解释为二分类标签分布。
    分布图只是描述统计，不证明语言现象的因果关系，也不直接保证截断覆盖率。

说明核对：2026-09-28；完整流程、前置条件及自定义模型路径见 USAGE_GUIDE.md。
"""

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse
import json
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np
from transformers import BertTokenizer

from experiment_utils import DEFAULT_BERT_MODEL

# ── 默认路径 ──────────────────────────────────────────────────────────────
ROOT       = Path(__file__).parent.parent
DATA_DIR   = ROOT / "data" / "afqmc"
OUTPUT_DIR = ROOT / "outputs" / "figures"

# 中文字体（matplotlib 默认不支持中文）
_CN_FONT = None
def _get_font():
    """寻找支持中文的系统字体并缓存。

    用途：
        寻找支持中文的系统字体并缓存。

    使用场景与调用位置：
        在 src/explore_data.py 的“检查标签和长度分布，生成数据探索图”流程中使用。 本文件直接调用位置：plot_char_length()、plot_label_distribution()、plot_length_diff()、plot_token_length()。

    参数：
        无显式参数；所需配置由内部逻辑、模块常量或命令行提供。

    返回值：
        matplotlib.font_manager.FontProperties。

    注意事项：
        首次查询后复用全局 _CN_FONT；找不到匹配字体时退回默认字体，可能仍有中文缺字。
    """
    global _CN_FONT
    if _CN_FONT is None:
        # 按字体族优先选择，避免误选仅含扩展汉字的 SimSun-ExtB。
        for family in ("Microsoft YaHei", "SimHei", "SimSun",
                       "Noto Sans CJK SC", "Source Han Sans SC", "PingFang SC"):
            try:
                font_path = fm.findfont(fm.FontProperties(family=family),
                                        fallback_to_default=False)
            except ValueError:
                continue
            _CN_FONT = fm.FontProperties(fname=font_path)
            break
        else:
            _CN_FONT = fm.FontProperties()
    return _CN_FONT


def load_jsonl(path):
    """读取 UTF-8 JSONL 并按文件顺序还原记录。

    用途：
        读取 UTF-8 JSONL 并按文件顺序还原记录。

    使用场景与调用位置：
        在 src/explore_data.py 的“检查标签和长度分布，生成数据探索图”流程中使用。 本文件直接调用位置：main()。

    参数：
        path: 待读取或写入的文件路径，具体模式见用途与注意事项。

    返回值：
        list[dict]；每个非空行对应一条记录。

    注意事项：
        用于检查标签和长度分布，生成数据探索图阶段的数据读取；忽略空行，不执行字段、标签或重复样本校验；缺失文件和非法 JSON 异常直接上抛，整个文件驻留内存。
    """
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


# ── 图 1：标签分布 ────────────────────────────────────────────────────────

def plot_label_distribution(splits_data, output_dir):
    """并排绘制各划分中标签 0/1 的数量。

    用途：
        并排绘制各划分中标签 0/1 的数量。

    使用场景与调用位置：
        在 src/explore_data.py 的“检查标签和长度分布，生成数据探索图”流程中使用。 本文件直接调用位置：main()。

    参数：
        splits_data: 划分名到对应句对记录列表的字典。
        output_dir: 输出目录，保存图像或实验结果。

    返回值：
        None；保存 label_distribution.png。

    注意事项：
        非 0/1 标签单独显示为未公开/未知；比例分母为该 split 全部样本。
        空划分显示无数据，不计算比例；自动创建输出目录。
    """
    if not splits_data:
        raise ValueError("没有可绘制的数据划分")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, len(splits_data), figsize=(10, 4))
    if len(splits_data) == 1:
        axes = [axes]

    fp = _get_font()
    for ax, (split_name, rows) in zip(axes, splits_data.items()):
        labels = [r.get("label") for r in rows]
        cnt = Counter(labels)
        counts = [cnt.get(0, 0), cnt.get(1, 0)]
        categories = ["不相似 (0)", "相似 (1)"]
        colors = ["#F44336", "#2196F3"]
        n_unknown = len(rows) - sum(counts)
        if n_unknown:
            categories.append("未公开/未知")
            counts.append(n_unknown)
            colors.append("#9E9E9E")
        bars = ax.bar(categories, counts, color=colors, width=0.5)
        ax.set_ylim(0, max(max(counts), 1) * 1.25)
        for bar, c in zip(bars, counts):
            if rows:
                ax.annotate(f"{c:,}\n({c/len(rows)*100:.1f}%)",
                            (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                            xytext=(0, 5), textcoords="offset points",
                            ha="center", va="bottom", fontproperties=fp, fontsize=9)
        if not rows:
            ax.text(0.5, 0.5, "无数据", transform=ax.transAxes,
                    ha="center", fontproperties=fp)
        ax.set_title(f"{split_name}（{len(rows):,} 条）", fontproperties=fp)
        ax.set_ylabel("数量", fontproperties=fp)
        ax.tick_params(axis="x", labelsize=9)
        for label in ax.get_xticklabels():
            label.set_fontproperties(fp)
            label.set_fontsize(9)

    fig.suptitle("AFQMC 标签分布（未公开/未知标签单独统计）", fontproperties=fp,
                 fontsize=12, y=1.02)
    fig.tight_layout()
    save_path = output_dir / "label_distribution.png"
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  图表已保存 → {save_path}")


# ── 图 2：句子字符长度分布 ────────────────────────────────────────────────

def plot_char_length(rows, output_dir):
    """比较正负样本两侧句子的字符长度并打印覆盖率。

    用途：
        比较正负样本两侧句子的字符长度并打印覆盖率。

    使用场景与调用位置：
        在 src/explore_data.py 的“检查标签和长度分布，生成数据探索图”流程中使用。 本文件直接调用位置：main()。

    参数：
        rows: 原始记录列表；句对通常包含 sentence1、sentence2、label。
        output_dir: 输出目录，保存图像或实验结果。

    返回值：
        None；保存 char_length_distribution.png。

    注意事项：
        长度单位是 Python 字符数，不能直接等同 tokenizer 长度；输入应非空。
    """
    pos_rows = [r for r in rows if r["label"] == 1]
    neg_rows = [r for r in rows if r["label"] == 0]

    def lens(rs):
        """展开一批句对两侧文本的字符长度。

        用途：
            展开一批句对两侧文本的字符长度。

        使用场景与调用位置：
            在 src/explore_data.py 的“检查标签和长度分布，生成数据探索图”流程中使用。 所属作用域：plot_char_length。

        参数：
            rs: 待统计两侧字符长度的句对记录列表。

        返回值：
            长度为 2×句对数的整数列表。

        注意事项：
            仅供外层 plot_char_length 使用，先拼接所有 sentence1 长度，再拼接 sentence2 长度。
        """
        return [len(r["sentence1"]) for r in rs] + [len(r["sentence2"]) for r in rs]

    pos_lens = lens(pos_rows)
    neg_lens = lens(neg_rows)

    fp = _get_font()
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(pos_lens, bins=40, alpha=0.6, label="正样本（相似）",
            color="#2196F3", density=True)
    ax.hist(neg_lens, bins=40, alpha=0.6, label="负样本（不相似）",
            color="#F44336", density=True)
    ax.axvline(32, color="black", linestyle="--", linewidth=1,
               label="max_length=32")
    ax.axvline(64, color="gray", linestyle="--", linewidth=1,
               label="max_length=64")
    ax.set_xlabel("句子字符长度", fontproperties=fp)
    ax.set_ylabel("密度", fontproperties=fp)
    ax.set_title("正/负样本句子长度分布（train）", fontproperties=fp)
    ax.legend(prop=fp)
    fig.tight_layout()

    save_path = output_dir / "char_length_distribution.png"
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  图表已保存 → {save_path}")

    all_lens = [len(r["sentence1"]) for r in rows] + [len(r["sentence2"]) for r in rows]
    print(f"  字符长度统计（train 全部句子）：")
    print(f"    均值={np.mean(all_lens):.1f}  中位数={np.median(all_lens):.0f}  "
          f"P95={np.percentile(all_lens, 95):.0f}  最长={max(all_lens)}")
    for threshold in [32, 48, 64, 96]:
        cover = sum(1 for l in all_lens if l <= threshold) / len(all_lens) * 100
        print(f"    max_length={threshold:3d} 覆盖率: {cover:.1f}%")


# ── 图 3：Token 数分布（BERT Tokenizer） ─────────────────────────────────

def plot_token_length(rows, tokenizer, output_dir):
    """抽查训练集分词后的单句长度。

    用途：
        抽查训练集分词后的单句长度。

    使用场景与调用位置：
        在 src/explore_data.py 的“检查标签和长度分布，生成数据探索图”流程中使用。 本文件直接调用位置：main()。

    参数：
        rows: 原始记录列表；句对通常包含 sentence1、sentence2、label。
        tokenizer: 与模型权重配套的 tokenizer；负责特殊 token、截断和 padding。
        output_dir: 输出目录，保存图像或实验结果。

    返回值：
        None；保存 token_length_distribution.png 并打印分位数。

    注意事项：
        只计算前 5000 条；tokenize 不包含 CLS/SEP，设定 max_length 时需额外考虑特殊 token。
    """
    print("  计算 Token 长度（需要 tokenize，稍慢...）")
    token_lens = []
    for r in rows[:5000]:  # 取前 5000 条避免太慢
        t1 = len(tokenizer.tokenize(r["sentence1"]))
        t2 = len(tokenizer.tokenize(r["sentence2"]))
        token_lens.extend([t1, t2])

    fp = _get_font()
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(token_lens, bins=40, color="#4CAF50", alpha=0.8, density=True)
    ax.axvline(np.mean(token_lens), color="red", linestyle="-",
               label=f"均值={np.mean(token_lens):.1f}")
    ax.axvline(np.percentile(token_lens, 95), color="orange", linestyle="--",
               label=f"P95={np.percentile(token_lens, 95):.0f}")
    ax.set_xlabel("单句 Token 数（不含 [CLS]/[SEP]）", fontproperties=fp)
    ax.set_ylabel("密度", fontproperties=fp)
    ax.set_title("单句 Token 数分布（train 前 5000 条）", fontproperties=fp)
    ax.legend(prop=fp)
    fig.tight_layout()

    save_path = output_dir / "token_length_distribution.png"
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  图表已保存 → {save_path}")
    print(f"  Token 长度：均值={np.mean(token_lens):.1f}  "
          f"P95={np.percentile(token_lens, 95):.0f}  最长={max(token_lens)}")


# ── 图 4：正/负样本长度差（捷径检测） ────────────────────────────────────

def plot_length_diff(rows, output_dir):
    """比较正负句对的绝对长度差，寻找长度捷径。

    用途：
        比较正负句对的绝对长度差，寻找长度捷径。

    使用场景与调用位置：
        在 src/explore_data.py 的“检查标签和长度分布，生成数据探索图”流程中使用。 本文件直接调用位置：main()。

    参数：
        rows: 原始记录列表；句对通常包含 sentence1、sentence2、label。
        output_dir: 输出目录，保存图像或实验结果。

    返回值：
        None；保存 length_diff_distribution.png 并打印启发式提示。

    注意事项：
        正例均值低于负例均值的 70% 时提示风险；该经验规则不是统计显著性检验，也不能证明不存在数据偏差。
    """
    pos_diffs = [abs(len(r["sentence1"]) - len(r["sentence2"]))
                 for r in rows if r["label"] == 1]
    neg_diffs = [abs(len(r["sentence1"]) - len(r["sentence2"]))
                 for r in rows if r["label"] == 0]

    fp = _get_font()
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(pos_diffs, bins=30, alpha=0.6, label=f"正样本 均值={np.mean(pos_diffs):.1f}",
            color="#2196F3", density=True)
    ax.hist(neg_diffs, bins=30, alpha=0.6, label=f"负样本 均值={np.mean(neg_diffs):.1f}",
            color="#F44336", density=True)
    ax.set_xlabel("|len(s1) - len(s2)| 字符数", fontproperties=fp)
    ax.set_ylabel("密度", fontproperties=fp)
    ax.set_title("正/负样本句子长度差分布（length bias 检测）", fontproperties=fp)
    ax.legend(prop=fp)
    fig.tight_layout()

    save_path = output_dir / "length_diff_distribution.png"
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  图表已保存 → {save_path}")
    print(f"  长度差：正样本均值={np.mean(pos_diffs):.1f}  负样本均值={np.mean(neg_diffs):.1f}")
    if np.mean(pos_diffs) < np.mean(neg_diffs) * 0.7:
        print("  ⚠️  正样本长度差明显更小，存在 length bias 风险")
    else:
        print("  ✓ 正/负样本长度差接近，无明显 length bias")


# ── 控制台统计输出 ────────────────────────────────────────────────────────

def print_stats(name, rows):
    """输出指定划分的类别计数、长度统计和样例。

    用途：
        输出指定划分的类别计数、长度统计和样例。

    使用场景与调用位置：
        在 src/explore_data.py 的“检查标签和长度分布，生成数据探索图”流程中使用。 本文件直接调用位置：main()。

    参数：
        name: 当前数据划分、模型或输出结果的显示名称。
        rows: 原始记录列表；句对通常包含 sentence1、sentence2、label。

    返回值：
        None；输出控制台报告。

    注意事项：
        将非 0/1 标签视为未公开标签；空数据会影响分位数或最大值计算，调用方应先检查。
    """
    labels  = [r["label"] for r in rows]
    cnt     = Counter(labels)
    s1_lens = [len(r["sentence1"]) for r in rows]
    s2_lens = [len(r["sentence2"]) for r in rows]
    all_lens = s1_lens + s2_lens

    print(f"\n{'='*50}")
    print(f"【{name}】共 {len(rows):,} 条")
    print(f"{'='*50}")

    n_pos = cnt.get(1, 0)
    n_neg = cnt.get(0, 0)
    n_unlabeled = sum(v for k, v in cnt.items() if k not in (0, 1))
    if n_unlabeled:
        # AFQMC 竞赛测试集标签为 -1（未公开），不计入正/负统计
        print(f"  标签未公开（CLUE 竞赛格式）: {n_unlabeled:>6,} 条  —— 仅供参考，不用于评估")
    else:
        print(f"  正样本（相似）  : {n_pos:>6,} ({n_pos/len(rows)*100:.1f}%)")
        print(f"  负样本（不相似）: {n_neg:>6,} ({n_neg/len(rows)*100:.1f}%)")
        print(f"  不均衡比 (neg/pos): {n_neg/max(n_pos, 1):.1f}x")
    print(f"  句子字符长度 — 均值={np.mean(all_lens):.1f}  中位数={np.median(all_lens):.0f}  "
          f"P95={np.percentile(all_lens, 95):.0f}  最长={max(all_lens)}")
    print(f"  示例正样本：")
    for r in [r for r in rows if r["label"] == 1][:2]:
        print(f"    ✓  {r['sentence1']!r}  ||  {r['sentence2']!r}")
    print(f"  示例负样本：")
    for r in [r for r in rows if r["label"] == 0][:2]:
        print(f"    ✗  {r['sentence1']!r}  ||  {r['sentence2']!r}")


# ── 主流程 ────────────────────────────────────────────────────────────────

def parse_args():
    """解析检查标签和长度分布，生成数据探索图所需的命令行配置。

    用途：
        解析检查标签和长度分布，生成数据探索图所需的命令行配置。

    使用场景与调用位置：
        本文件 main() 的启动配置阶段调用；独立运行脚本时首先经过此处。 本文件直接调用位置：main()。

    参数：
        无显式参数；所需配置由内部逻辑、模块常量或命令行提供。

    返回值：
        argparse.Namespace；成员名对应下方 add_argument 定义。

    注意事项：
        由本文件 main 在任何主要工作开始前调用；读取 sys.argv，处理默认值、类型和 choices；--help 或不合法参数通过 argparse 触发 SystemExit，不启动训练或推理。
    """
    parser = argparse.ArgumentParser(description="AFQMC 数据集探索")
    parser.add_argument("--data_dir",   default=str(DATA_DIR),   type=Path)
    parser.add_argument("--bert_path",  default=DEFAULT_BERT_MODEL,  type=str)
    parser.add_argument("--output_dir", default=str(OUTPUT_DIR), type=Path)
    parser.add_argument("--skip_token", action="store_true", help="跳过 Token 长度分析（较慢）")
    return parser.parse_args()


def main():
    """组织本脚本的完整流程：检查标签和长度分布，生成数据探索图。

    用途：
        组织本脚本的完整流程：检查标签和长度分布，生成数据探索图。

    使用场景与调用位置：
        直接运行此文件时由 if __name__ == "__main__" 入口调用；也可显式调用 main()，此时仍读取进程命令行。

    参数：
        无显式参数；所需配置由内部逻辑、模块常量或命令行提供。

    返回值：
        None；产物及副作用见注意事项，配置由命令行读取。

    注意事项：
        读取现有划分、打印分布并保存四类图；skip_token 可跳过模型 tokenizer 加载，训练文件为空时提前退出。
    """
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    splits = {}
    for split in ["train", "validation", "test"]:
        path = args.data_dir / f"{split}.jsonl"
        if path.exists():
            splits[split] = load_jsonl(path)

    for name, rows in splits.items():
        print_stats(name, rows)

    train_rows = splits.get("train", [])
    if not train_rows:
        print("train.jsonl 不存在，请先运行 download_data.py")
        return

    print(f"\n{'='*50}")
    print("生成可视化图表...")

    plot_label_distribution(splits, args.output_dir)
    plot_char_length(train_rows, args.output_dir)
    plot_length_diff(train_rows, args.output_dir)

    if not args.skip_token:
        tokenizer = BertTokenizer.from_pretrained(args.bert_path)
        plot_token_length(train_rows, tokenizer, args.output_dir)

    print(f"\n所有图表已保存至 → {args.output_dir}")


if __name__ == "__main__":
    main()
