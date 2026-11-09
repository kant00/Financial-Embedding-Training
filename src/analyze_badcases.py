"""
错误句对分析：定位 FP/FN、阈值附近错误及分数分布。

用途与流程位置：
    在模型训练后分析 calibration，为数据检查与优化提出假设；不是自动优化器。
    支持 Bi-Encoder 和 Cross-Encoder，打印错误分组、语言特征和典型案例并保存图像。

输入与输出：
    --ckpt 指定已训练权重；--data_dir 指向新协议目录，validation 是 calibration 别名。
    --bert_path 指定对应预训练模型；--output_dir 保存错误分布图。
    双塔读取 checkpoint 阈值与单句长度；交叉塔使用 0.5 边界、固定句对长度 128。

使用方式（在项目根目录执行，Bash）：
    python src/analyze_badcases.py --model_type biencoder --ckpt outputs/dev/cosine_seed42/checkpoints/biencoder_cosine_best.pt --data_dir data/experiments/afqmc_seed42 --split validation --output_dir outputs/dev/cosine_seed42/badcases --n_cases 10

注意事项：
    默认数据目录仍是旧 data/afqmc，正式流程必须显式传路径。
    不要用 final test 错误反复调参；输出的优化建议是启发式，不是实测因果结论。
    CosineEmbeddingLoss 增大 margin 会放宽负例约束，不能解释成更严格分离。

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
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import BertTokenizer

from dataset import PairDataset, CrossEncoderDataset, load_jsonl
from experiment_utils import DEFAULT_BERT_MODEL
from model import build_biencoder, build_crossencoder

# ── 默认路径 ──────────────────────────────────────────────────────────────
ROOT      = Path(__file__).parent.parent
DATA_DIR  = ROOT / "data" / "afqmc"
CKPT_DIR  = ROOT / "outputs" / "checkpoints"
FIG_DIR   = ROOT / "outputs" / "figures"


# ── 收集预测结果 ──────────────────────────────────────────────────────────

@torch.no_grad()
def collect_biencoder_preds(model, loader, raw_rows, device, threshold):
    """把模型预测与原始句对对齐以支持错误分析。

    用途：
        把模型预测与原始句对对齐以支持错误分析。

    使用场景与调用位置：
        在 src/analyze_badcases.py 的“采集错误句对，分析 FP/FN 及分数分布”流程中使用。 本文件直接调用位置：main()。

    参数：
        model: 当前操作的模型实例；结构和设备须匹配输入。
        loader: 提供 batch 字典的 DataLoader；评估时应保持样本顺序。
        raw_rows: 与 DataLoader 顺序完全一致的原始句对记录。
        device: torch.device 或兼容设备描述，指定推理/训练位置。
        threshold: score>=threshold 判为正例；应由 calibration 选择。

    返回值：
        每样本一个字典的列表：sentence1、sentence2、label、score、pred。

    注意事项：
        由 main 在对应模型分支调用；score 为余弦相似度。loader 必须 shuffle=False，且顺序/样本量与 raw_rows 完全一致；eval/no_grad，不恢复训练模式。
    """
    model.eval()
    results = []
    idx = 0
    for batch in loader:
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
        sims   = F.cosine_similarity(emb_a, emb_b, dim=-1).cpu().tolist()
        labels = batch["label"].tolist()

        for sim, label in zip(sims, labels):
            row = raw_rows[idx]
            results.append({
                "sentence1": row["sentence1"],
                "sentence2": row["sentence2"],
                "label": label,
                "score": sim,
                "pred":  int(sim >= threshold),
            })
            idx += 1
    return results


@torch.no_grad()
def collect_crossencoder_preds(model, loader, raw_rows, device):
    """把模型预测与原始句对对齐以支持错误分析。

    用途：
        把模型预测与原始句对对齐以支持错误分析。

    使用场景与调用位置：
        在 src/analyze_badcases.py 的“采集错误句对，分析 FP/FN 及分数分布”流程中使用。 本文件直接调用位置：main()。

    参数：
        model: 当前操作的模型实例；结构和设备须匹配输入。
        loader: 提供 batch 字典的 DataLoader；评估时应保持样本顺序。
        raw_rows: 与 DataLoader 顺序完全一致的原始句对记录。
        device: torch.device 或兼容设备描述，指定推理/训练位置。

    返回值：
        每样本一个字典的列表：sentence1、sentence2、label、score、pred。

    注意事项：
        由 main 在对应模型分支调用；score 为softmax 正类概率。loader 必须 shuffle=False，且顺序/样本量与 raw_rows 完全一致；eval/no_grad，不恢复训练模式。
    """
    model.eval()
    results = []
    idx = 0
    for batch in loader:
        logits = model(
            batch["input_ids"].to(device),
            batch["attention_mask"].to(device),
            batch["token_type_ids"].to(device),
        ).cpu()
        probs  = torch.softmax(logits, dim=-1)[:, 1].tolist()
        preds  = logits.argmax(dim=-1).tolist()
        labels = batch["label"].tolist()

        for prob, pred, label in zip(probs, preds, labels):
            row = raw_rows[idx]
            results.append({
                "sentence1": row["sentence1"],
                "sentence2": row["sentence2"],
                "label": label,
                "score": prob,   # P(相似)
                "pred":  pred,
            })
            idx += 1
    return results


# ── Bad Case 分类 ─────────────────────────────────────────────────────────

def split_badcases(results, threshold=0.5):
    """将错误分成假阳性/假阴性以及远阈值/近阈值四组。

    用途：
        将错误分成假阳性/假阴性以及远阈值/近阈值四组。

    使用场景与调用位置：
        在 src/analyze_badcases.py 的“采集错误句对，分析 FP/FN 及分数分布”流程中使用。 本文件直接调用位置：main()。

    参数：
        results: 包含样本标签、预测、分数或方法元数据的结果列表。
        threshold: score>=threshold 判为正例；应由 calibration 选择。

    返回值：
        fp_high、fp_border、fn_high、fn_border 四个列表组成的字典。

    注意事项：
        离阈值绝对差大于 0.15 视为高分差错误；这是启发式分组，不是已校准置信度。FP 按分数降序，FN 按分数升序。
    """
    fp_high, fp_border = [], []
    fn_high, fn_border = [], []

    for r in results:
        if r["pred"] == r["label"]:
            continue
        gap = abs(r["score"] - threshold)
        if r["pred"] == 1 and r["label"] == 0:   # FP
            (fp_high if gap > 0.15 else fp_border).append(r)
        else:                                      # FN
            (fn_high if gap > 0.15 else fn_border).append(r)

    # 匿名排序键从错误记录读取 score：FP 高分优先、FN 低分优先，便于先看
    # 偏离阈值明显的案例。分数距离是启发式，不代表已校准的错误置信概率。
    fp_high.sort(key=lambda x: x["score"], reverse=True)
    fn_high.sort(key=lambda x: x["score"])

    return {
        "fp_high":   fp_high,
        "fp_border": fp_border,
        "fn_high":   fn_high,
        "fn_border": fn_border,
    }


# ── 模式分析 ──────────────────────────────────────────────────────────────

def analyze_patterns(cases, label):
    """打印错误子集的长度差和字符重叠统计。

    用途：
        打印错误子集的长度差和字符重叠统计。

    使用场景与调用位置：
        在 src/analyze_badcases.py 的“采集错误句对，分析 FP/FN 及分数分布”流程中使用。 本文件直接调用位置：main()。

    参数：
        cases: 已筛选并按展示顺序排列的错误样本列表。
        label: 错误子集的显示标签，例如 FP 或 FN。

    返回值：
        None；控制台输出，空列表直接返回。

    注意事项：
        main 分别对 FP/FN 调用；字符 Jaccard 不考虑顺序、重复次数和分词，仅用于定位表层重叠模式。
    """
    if not cases:
        return
    len_diffs  = [abs(len(r["sentence1"]) - len(r["sentence2"])) for r in cases]
    lens_s1    = [len(r["sentence1"]) for r in cases]
    lens_s2    = [len(r["sentence2"]) for r in cases]

    # 字符级 Jaccard 相似度（用 unigram 集合）
    def jaccard(a, b):
        """计算两段文本字符集合的交并比。

        用途：
            计算两段文本字符集合的交并比。

        使用场景与调用位置：
            在 src/analyze_badcases.py 的“采集错误句对，分析 FP/FN 及分数分布”流程中使用。 所属作用域：analyze_patterns。

        参数：
            a: 用于字符集合重叠比较的第一段文本。
            b: 用于字符集合重叠比较的第二段文本。

        返回值：
            [0,1] 浮点数；两者均空时为 0。

        注意事项：
            只在 analyze_patterns 内计算案例字符重叠；用 max(并集大小,1) 避免除零。
        """
        sa, sb = set(a), set(b)
        return len(sa & sb) / max(len(sa | sb), 1)

    jaccards = [jaccard(r["sentence1"], r["sentence2"]) for r in cases]

    print(f"\n  【{label}】共 {len(cases)} 条")
    print(f"    长度差     : 均值={np.mean(len_diffs):.1f}  中位={np.median(len_diffs):.0f}")
    print(f"    s1 长度    : 均值={np.mean(lens_s1):.1f}")
    print(f"    s2 长度    : 均值={np.mean(lens_s2):.1f}")
    print(f"    字符 Jaccard: 均值={np.mean(jaccards):.3f}  "
          f"（1=完全重叠，0=无共同字符）")


def print_cases(cases, title, n=5):
    """按已有顺序打印最多 n 条错误原文及分数。

    用途：
        按已有顺序打印最多 n 条错误原文及分数。

    使用场景与调用位置：
        在 src/analyze_badcases.py 的“采集错误句对，分析 FP/FN 及分数分布”流程中使用。 本文件直接调用位置：main()。

    参数：
        cases: 已筛选并按展示顺序排列的错误样本列表。
        title: 控制台分组标题或图像标题。
        n: 最多展示的记录条数。

    返回值：
        None；输出控制台。

    注意事项：
        main 对每个错误分组调用；不额外排序，不改变输入列表，便于对照高分错误。
    """
    print(f"\n  {title} (展示 {min(n, len(cases))} 条)：")
    for r in cases[:n]:
        score_str = f"score={r['score']:.3f}"
        print(f"    {score_str}  | {r['sentence1']!r}")
        print(f"    {'':>12}  | {r['sentence2']!r}")
        print()


# ── Score 分布可视化 ──────────────────────────────────────────────────────

def plot_score_dist_with_errors(results, threshold, save_path, model_label):
    """同时查看标签分布和正确/错误预测的分数分布。

    用途：
        同时查看标签分布和正确/错误预测的分数分布。

    使用场景与调用位置：
        在 src/analyze_badcases.py 的“采集错误句对，分析 FP/FN 及分数分布”流程中使用。 本文件直接调用位置：main()。

    参数：
        results: 包含样本标签、预测、分数或方法元数据的结果列表。
        threshold: score>=threshold 判为正例；应由 calibration 选择。
        save_path: 目标图片文件路径，通常为 PNG。
        model_label: 用于图像标题的模型名称和配置描述。

    返回值：
        None；保存双子图并创建父目录。

    注意事项：
        main 在错误统计后调用；threshold=None 时不绘制阈值线，model_label 仅用于标题；空子组直方图可能没有有效密度。
    """
    scores  = np.array([r["score"]  for r in results])
    labels  = np.array([r["label"]  for r in results])
    correct = np.array([r["pred"] == r["label"] for r in results])

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # 左图：正/负样本分数分布（同 evaluate.py，基准参考）
    ax = axes[0]
    ax.hist(scores[labels==1], bins=40, alpha=0.6, label="positive", color="#2196F3", density=True)
    ax.hist(scores[labels==0], bins=40, alpha=0.6, label="negative", color="#F44336", density=True)
    if threshold is not None:
        ax.axvline(threshold, color="black", linestyle="--", label=f"threshold={threshold:.2f}")
    ax.set_xlabel("Score (cosine sim / P(similar))")
    ax.set_title(f"[{model_label}] Positive vs Negative Distribution")
    ax.legend(fontsize=8)

    # 右图：正确 vs 错误预测的分数分布
    ax = axes[1]
    ax.hist(scores[correct],  bins=40, alpha=0.6, label="correct", color="#4CAF50", density=True)
    ax.hist(scores[~correct], bins=40, alpha=0.6, label="error",   color="#F44336", density=True)
    if threshold is not None:
        ax.axvline(threshold, color="black", linestyle="--")
    ax.set_xlabel("Score")
    ax.set_title(f"[{model_label}] Correct vs Error Score Distribution")
    ax.legend(fontsize=8)

    fig.tight_layout()
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  图表已保存 → {save_path}")


# ── 优化方向输出 ──────────────────────────────────────────────────────────

def print_optimization_directions(badcases, model_type, fp_jaccard_mean, fn_jaccard_mean):
    """根据 FP/FN 字符重叠均值打印优化建议。

    用途：
        根据 FP/FN 字符重叠均值打印优化建议。

    使用场景与调用位置：
        在 src/analyze_badcases.py 的“采集错误句对，分析 FP/FN 及分数分布”流程中使用。 本文件直接调用位置：main()。

    参数：
        badcases: split_badcases 返回的四类错误字典；当前建议函数未直接使用。
        model_type: 模型类型描述；当前建议函数未直接使用。
        fp_jaccard_mean: 假阳性样本字符 Jaccard 的均值，用于选择建议分支。
        fn_jaccard_mean: 假阴性样本字符 Jaccard 的均值，用于选择建议分支。

    返回值：
        None；输出建议文本。

    注意事项：
        main 完成模式分析后调用；部分建议固定输出，条件分支仅使用两个 Jaccard 均值，badcases/model_type 当前未参与决策。建议不是已验证收益，历史 epoch/增益描述须与升级日志核对。
    """
    print(f"\n{'='*60}")
    print("优化方向建议（基于当前 bad case 分析）")
    print(f"{'='*60}")

    print("""
【1】数据层面
  ├─ 难负样本增强（Hard Negative Mining）
  │    当前负样本是随机采样，FP 案例中很多是"话题相关但语义不同"的句对。
  │    → 用训练好的 BiEncoder 在大规模数据中挖掘相似度高但标签为 0 的对，
  │      加入训练集，提升负例的区分难度。
  │
  ├─ 数据增强（正样本扩充）
  │    AFQMC 正样本只有 10K 条（31%），TripletLoss 三元组因此受限。
  │    → 对正样本做同义改写（换词、调序），扩充正例数量。
  │      可用 LLM API 批量生成改写句。
  │
  └─ 跨数据集迁移
       LCQMC / BQ Corpus（已下载）包含更多样的问句对。
       → 先在 LCQMC（238K 对）上预训练 Sentence-BERT，再 fine-tune 到 AFQMC，
         利用大数据集的语义泛化能力。
""")

    if fp_jaccard_mean > 0.5:
        print("""【2】模型层面（针对 FP：字符重叠高但语义不同）
  ├─ 调小 CosineEmbeddingLoss 的 margin（0.3 → 0.1）
  │    负例损失为 max(0, cosine-margin)；调小 margin 才会要求负例推得更远，
  │    可减少高词汇重叠带来的假阳性，但也要监控 FN 是否增加。
  │
  └─ 引入 Interaction 特征辅助
       BiEncoder 看不到词序交互，而这类 FP 往往需要理解"借呗还款"vs"借呗账单"
       的细粒度区别。可在顶层加一个轻量级 Cross-Attention（保留向量化能力）。
""")
    else:
        print("""【2】模型层面（FP 字符重叠不高，主要是语义理解不足）
  ├─ 增加 BERT 层数（4 → 8 → 12 层）
  │    浅层 BERT 对语义的建模能力有限，更深层能捕捉更细粒度的语义差异。
  │
  └─ 换用金融领域预训练模型
       FinBERT / MacBERT / RoBERTa-Chinese 在金融/客服语料上有更好的初始化，
       AFQMC 中很多错误源于领域术语理解不准确。
""")

    if fn_jaccard_mean < 0.3:
        print("""【3】训练策略层面（针对 FN：词汇重叠低但语义相同的同义句）
  ├─ SimCSE 对比学习预训练
  │    同一句话 dropout 两次得到两个正例，大 batch 内其他句子为负例。
  │    这种方式能让模型学到"用不同词说同一个意思"的不变性。
  │
  └─ 调小 TripletLoss 的 margin（0.3 → 0.1）
       如果正例本身语义就不太相似（同义但换了词），过大的 margin 反而
       要求 sim(a,p) 比 sim(a,n) 高出太多，训练信号消失。
""")

    print("""【4】评估与部署层面
  ├─ 阈值校准
  │    当前阈值在 val 集上网格搜索，但 val 正负比（31:69）和线上分布可能不同。
  │    → 收集真实线上日志，按实际分布重新校准阈值（Platt scaling 等）。
  │
  ├─ 两阶段级联（最实用的工程改进）
  │    BiEncoder（召回 Top-K）→ CrossEncoder（精排 Top-1）
  │    这是 rag_annual_report Reranker 的完整版。
  │    → 可用当前两个 checkpoint 直接组合，无需重新训练。
  │
  └─ 训练更多 epoch + 全量 12 层
       本次演示用 4 层 × 1 epoch 快速验证，完整训练预计提升 5~10 个 F1 点。
       → 建议学生实验：4 层 vs 12 层，1 epoch vs 5 epoch 的 2×2 消融。
""")


# ── 主流程 ────────────────────────────────────────────────────────────────

def parse_args():
    """解析采集错误句对，分析 FP/FN 及分数分布所需的命令行配置。

    用途：
        解析采集错误句对，分析 FP/FN 及分数分布所需的命令行配置。

    使用场景与调用位置：
        本文件 main() 的启动配置阶段调用；独立运行脚本时首先经过此处。 本文件直接调用位置：main()。

    参数：
        无显式参数；所需配置由内部逻辑、模块常量或命令行提供。

    返回值：
        argparse.Namespace；成员名对应下方 add_argument 定义。

    注意事项：
        由本文件 main 在任何主要工作开始前调用；读取 sys.argv，处理默认值、类型和 choices；--help 或不合法参数通过 argparse 触发 SystemExit，不启动训练或推理。
    """
    parser = argparse.ArgumentParser(description="Bad Case 分析与优化方向")
    parser.add_argument("--model_type", default="biencoder",
                        choices=["biencoder", "crossencoder"])
    parser.add_argument("--ckpt", default=None, type=str,
                        help="checkpoint 路径（默认自动选 biencoder_cosine_best.pt）")
    # 新协议将 calibration 映射到 validation；显式传入路径，避免误读旧验证集。
    parser.add_argument("--data_dir", type=Path, default=DATA_DIR,
                        help="包含 validation/test.jsonl 的数据目录")
    parser.add_argument("--bert_path", default=DEFAULT_BERT_MODEL,
                        help="与 checkpoint 对应的 BERT 模型 ID 或本地目录")
    parser.add_argument("--output_dir", type=Path, default=FIG_DIR,
                        help="错误分布图保存目录；不同实验应使用不同目录")
    parser.add_argument("--split",      default="validation",
                        choices=["validation", "test"])
    parser.add_argument("--batch_size", default=64, type=int)
    parser.add_argument("--n_cases",    default=5, type=int,
                        help="每类 bad case 展示条数")
    return parser.parse_args()


def main():
    """组织本脚本的完整流程：采集错误句对，分析 FP/FN 及分数分布。

    用途：
        组织本脚本的完整流程：采集错误句对，分析 FP/FN 及分数分布。

    使用场景与调用位置：
        直接运行此文件时由 if __name__ == "__main__" 入口调用；也可显式调用 main()，此时仍读取进程命令行。

    参数：
        无显式参数；所需配置由内部逻辑、模块常量或命令行提供。

    返回值：
        None；产物及副作用见注意事项，配置由命令行读取。

    注意事项：
        checkpoint 不存在时打印提示并返回；双塔使用已保存阈值，交叉塔用 0.5；输出错误分组、案例和图像。
        data_dir/bert_path/output_dir 可从命令行指定；调试阶段应读取 calibration 的 validation 别名，避免用 final test 指导优化。
    """
    args   = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── 确定 checkpoint 路径 ──────────────────────────────────────────────
    if args.ckpt:
        ckpt_path = Path(args.ckpt)
    elif args.model_type == "biencoder":
        ckpt_path = CKPT_DIR / "biencoder_cosine_best.pt"
    else:
        ckpt_path = CKPT_DIR / "crossencoder_best.pt"

    if not ckpt_path.exists():
        print(f"❌ checkpoint 不存在: {ckpt_path}")
        print("请先运行训练脚本（train_biencoder.py / train_crossencoder.py）")
        return

    print(f"加载 checkpoint: {ckpt_path}")
    ckpt       = torch.load(ckpt_path, map_location=device, weights_only=False)
    saved_args = ckpt.get("args", {})

    # ── 加载原始数据和 DataLoader ─────────────────────────────────────────
    data_path = args.data_dir / f"{args.split}.jsonl"
    raw_rows  = load_jsonl(data_path)
    print(f"数据集: {data_path.name}  共 {len(raw_rows):,} 条")

    tokenizer = BertTokenizer.from_pretrained(args.bert_path)

    # ── 收集预测结果 ──────────────────────────────────────────────────────
    if args.model_type == "biencoder":
        model = build_biencoder(
            bert_path=args.bert_path,
            pool=saved_args.get("pool", "mean"),
            num_hidden_layers=saved_args.get("num_hidden_layers"),
        ).to(device)
        model.load_state_dict(ckpt["state_dict"])
        threshold = ckpt.get("threshold", 0.5)

        ds     = PairDataset(data_path, tokenizer, saved_args.get("max_length", 64))
        loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
        results = collect_biencoder_preds(model, loader, raw_rows, device, threshold)
        model_label = f"BiEncoder(loss={saved_args.get('loss','cosine')},threshold={threshold:.2f})"

    else:
        model = build_crossencoder(
            bert_path=args.bert_path,
            num_hidden_layers=saved_args.get("num_hidden_layers"),
        ).to(device)
        model.load_state_dict(ckpt["state_dict"])
        threshold = 0.5  # P(相似)=0.5 作为分界

        ds     = CrossEncoderDataset(data_path, tokenizer, max_length=128)
        loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
        results = collect_crossencoder_preds(model, loader, raw_rows, device)
        model_label = f"CrossEncoder(threshold=P≥0.5)"

    # ── 整体指标 ──────────────────────────────────────────────────────────
    labels    = [r["label"] for r in results]
    preds     = [r["pred"]  for r in results]
    n_correct = sum(1 for l, p in zip(labels, preds) if l == p)
    accuracy  = n_correct / len(results)
    print(f"\n整体准确率: {accuracy:.4f}  错误数: {len(results)-n_correct}")

    # ── 分类 bad case ─────────────────────────────────────────────────────
    badcases = split_badcases(results, threshold)
    fp_all = badcases["fp_high"] + badcases["fp_border"]
    fn_all = badcases["fn_high"] + badcases["fn_border"]

    print(f"\n{'='*60}")
    print(f"Bad Case 汇总  (共 {len(fp_all)+len(fn_all)} 个错误)")
    print(f"{'─'*60}")
    print(f"  FP 假阳性（预测相似，实际不同）: {len(fp_all):>4} 条")
    print(f"    其中高置信度错误  (Δscore>0.15): {len(badcases['fp_high']):>3} 条")
    print(f"    其中临界错误     (Δscore≤0.15): {len(badcases['fp_border']):>3} 条")
    print(f"  FN 假阴性（预测不同，实际相似）: {len(fn_all):>4} 条")
    print(f"    其中高置信度错误  (Δscore>0.15): {len(badcases['fn_high']):>3} 条")
    print(f"    其中临界错误     (Δscore≤0.15): {len(badcases['fn_border']):>3} 条")

    # ── 模式分析 ──────────────────────────────────────────────────────────
    print(f"\n{'─'*60}")
    print("Bad Case 语言特征分析：")
    analyze_patterns(fp_all,  "FP（假阳性）")
    analyze_patterns(fn_all,  "FN（假阴性）")

    fp_jaccard_mean = (np.mean([
        len(set(r["sentence1"]) & set(r["sentence2"])) /
        max(len(set(r["sentence1"]) | set(r["sentence2"])), 1)
        for r in fp_all
    ]) if fp_all else 0)
    fn_jaccard_mean = (np.mean([
        len(set(r["sentence1"]) & set(r["sentence2"])) /
        max(len(set(r["sentence1"]) | set(r["sentence2"])), 1)
        for r in fn_all
    ]) if fn_all else 0)

    # ── 展示典型案例 ──────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print_cases(badcases["fp_high"],  f"FP 高置信度错误（score最高的{args.n_cases}条）",
                n=args.n_cases)
    print_cases(badcases["fp_border"], f"FP 临界错误（{args.n_cases}条）",
                n=args.n_cases)
    print_cases(badcases["fn_high"],  f"FN 高置信度错误（score最低的{args.n_cases}条）",
                n=args.n_cases)
    print_cases(badcases["fn_border"], f"FN 临界错误（{args.n_cases}条）",
                n=args.n_cases)

    # ── 可视化 ────────────────────────────────────────────────────────────
    args.output_dir.mkdir(parents=True, exist_ok=True)
    fig_path = args.output_dir / f"{args.model_type}_badcase_dist.png"
    plot_score_dist_with_errors(results, threshold, fig_path, model_label)

    # ── 优化方向 ──────────────────────────────────────────────────────────
    print_optimization_directions(badcases, args.model_type,
                                  fp_jaccard_mean, fn_jaccard_mean)


if __name__ == "__main__":
    main()
