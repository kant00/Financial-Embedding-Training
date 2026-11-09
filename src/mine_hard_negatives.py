"""
离线难负样本挖掘：训练双塔检索候选并导出三元组。

用途与流程位置：
    已有开发双塔 checkpoint 后运行；仅从 train 挖掘，为 Hard Triplet 提供输入。
    默认只接受与 anchor 有显式 label=0 关系的召回候选，过滤自身和已知正例。

输入与输出：
    --data_file 指定训练句对；--checkpoint 指定用于挖掘的双塔；--bert_path 须一致。
    --output_file 写 anchor/positive/negative 及分数/来源；同名 .metadata.json 记录数量、
    coverage、来源和实际检索后端，训练前必须检查非空并抽检质量。

使用方式（在项目根目录执行，Bash）：
    python src/mine_hard_negatives.py --data_file data/experiments/afqmc_seed42/train.jsonl --checkpoint outputs/dev/cosine_seed42/checkpoints/biencoder_cosine_best.pt --output_file outputs/hard_negatives/triplets.jsonl --retrieval_depth 100 --backend auto

注意事项：
    --allow_unlabeled_candidates 会引入未人工确认的候选，不保证是真负例。
    显式负例可能不在召回深度内，结果可能为空；多负例下 coverage 不等于去重 anchor 覆盖率。
    auto 在 FAISS 不可导入时回退 NumPy；后者构造完整分数矩阵，降低编码 batch 不消除该开销。
    不要使用 calibration/final test 挖掘后回流训练。

说明核对：2026-09-28；完整流程、前置条件及自定义模型路径见 USAGE_GUIDE.md。
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import torch
from transformers import BertTokenizer

from experiment_utils import DEFAULT_BERT_MODEL, save_json, set_global_seed, write_jsonl
from model import build_biencoder
from retrieval_utils import encode_texts, search_inner_product


ROOT = Path(__file__).parent.parent
DEFAULT_DATA = ROOT / "data" / "experiments" / "afqmc_seed42" / "train.jsonl"
DEFAULT_OUTPUT = ROOT / "outputs" / "hard_negatives" / "triplets.jsonl"


def load_jsonl(path):
    """读取 UTF-8 JSONL 并按文件顺序还原记录。

    用途：
        读取 UTF-8 JSONL 并按文件顺序还原记录。

    使用场景与调用位置：
        在 src/mine_hard_negatives.py 的“使用训练好的双编码器从训练语料挖掘高相似负例”流程中使用。 本文件直接调用位置：main()。

    参数：
        path: 待读取或写入的文件路径，具体模式见用途与注意事项。

    返回值：
        list[dict]；每个非空行对应一条记录。

    注意事项：
        用于使用训练好的双编码器从训练语料挖掘高相似负例阶段的数据读取；忽略空行，不执行字段、标签或重复样本校验；缺失文件和非法 JSON 异常直接上抛，整个文件驻留内存。
    """
    with open(path, encoding="utf-8") as file_obj:
        return [json.loads(line) for line in file_obj if line.strip()]


def normalize_text(text):
    """为负例过滤生成不区分空白和大小写的文本键。

    用途：
        为负例过滤生成不区分空白和大小写的文本键。

    使用场景与调用位置：
        在 src/mine_hard_negatives.py 的“使用训练好的双编码器从训练语料挖掘高相似负例”流程中使用。 本文件直接调用位置：main()。

    参数：
        text: 待规范化或编码的单个文本字符串。

    返回值：
        规范化字符串。

    注意事项：
        main 排除自身、当前正例及已知正例时调用；仅文本规范化，不能识别未标注同义改写。
    """
    return "".join(str(text).strip().lower().split())


def parse_args():
    """解析使用训练好的双编码器从训练语料挖掘高相似负例所需的命令行配置。

    用途：
        解析使用训练好的双编码器从训练语料挖掘高相似负例所需的命令行配置。

    使用场景与调用位置：
        本文件 main() 的启动配置阶段调用；独立运行脚本时首先经过此处。 本文件直接调用位置：main()。

    参数：
        无显式参数；所需配置由内部逻辑、模块常量或命令行提供。

    返回值：
        argparse.Namespace；成员名对应下方 add_argument 定义。

    注意事项：
        由本文件 main 在任何主要工作开始前调用；读取 sys.argv，处理默认值、类型和 choices；--help 或不合法参数通过 argparse 触发 SystemExit，不启动训练或推理。
    """
    parser = argparse.ArgumentParser(description="离线难负样本挖掘")
    parser.add_argument("--data_file", default=str(DEFAULT_DATA))
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--bert_path", default=DEFAULT_BERT_MODEL)
    parser.add_argument("--output_file", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--negatives_per_positive", type=int, default=1)
    parser.add_argument("--retrieval_depth", type=int, default=100)
    parser.add_argument("--allow_unlabeled_candidates", action="store_true")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--max_length", type=int, default=64)
    parser.add_argument("--backend", choices=["auto", "faiss", "numpy"], default="auto")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    """组织本脚本的完整流程：使用训练好的双编码器从训练语料挖掘高相似负例。

    用途：
        组织本脚本的完整流程：使用训练好的双编码器从训练语料挖掘高相似负例。

    使用场景与调用位置：
        直接运行此文件时由 if __name__ == "__main__" 入口调用；也可显式调用 main()，此时仍读取进程命令行。

    参数：
        无显式参数；所需配置由内部逻辑、模块常量或命令行提供。

    返回值：
        None；产物及副作用见注意事项，配置由命令行读取。

    注意事项：
        只应传入训练语料。对正例 anchor 检索 Top-depth，过滤自身和直接已知正例；默认仅显式 label=0，允许弱负例时记录来源。coverage 实际为三元组数/正例数，多负例配置时可大于 1。
    """
    args = parse_args()
    set_global_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows = load_jsonl(args.data_file)

    positive_rows = [row for row in rows if int(row["label"]) == 1]
    positive_by_anchor, explicit_negative_by_anchor = defaultdict(set), defaultdict(set)
    all_texts = set()
    for row in rows:
        left, right = row["sentence1"], row["sentence2"]
        all_texts.update((left, right))
        target = positive_by_anchor if int(row["label"]) == 1 else explicit_negative_by_anchor
        target[left].add(right)
        target[right].add(left)

    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    saved_args = checkpoint.get("args", {})
    tokenizer = BertTokenizer.from_pretrained(args.bert_path)
    model = build_biencoder(
        args.bert_path,
        pool=saved_args.get("pool", "mean"),
        num_hidden_layers=saved_args.get("num_hidden_layers"),
    ).to(device)
    model.load_state_dict(checkpoint["state_dict"])

    corpus = sorted(all_texts)
    corpus_vectors = encode_texts(
        model, tokenizer, corpus, device, args.batch_size, args.max_length
    )
    text_to_index = {text: index for index, text in enumerate(corpus)}
    query_texts = sorted({row["sentence1"] for row in positive_rows})
    query_vectors = corpus_vectors[[text_to_index[text] for text in query_texts]]
    scores, indices, backend_used = search_inner_product(
        query_vectors, corpus_vectors, args.retrieval_depth, args.backend
    )
    retrieved = {
        query: [(corpus[index], float(score)) for index, score in zip(row_indices, row_scores)]
        for query, row_indices, row_scores in zip(query_texts, indices, scores)
    }

    triplets = []
    source_counts = defaultdict(int)
    for row in positive_rows:
        anchor, positive = row["sentence1"], row["sentence2"]
        known_positive_norm = {
            normalize_text(text) for text in positive_by_anchor.get(anchor, set())
        }
        candidates = []
        for candidate, score in retrieved.get(anchor, []):
            candidate_norm = normalize_text(candidate)
            if candidate_norm in {normalize_text(anchor), normalize_text(positive)}:
                continue
            if candidate_norm in known_positive_norm:
                continue
            is_explicit = candidate in explicit_negative_by_anchor.get(anchor, set())
            if not is_explicit and not args.allow_unlabeled_candidates:
                continue
            source = "explicit_label_0" if is_explicit else "unlabeled_retrieval"
            candidates.append((candidate, score, source))

        for negative, score, source in candidates[:args.negatives_per_positive]:
            triplets.append({
                "anchor": anchor,
                "positive": positive,
                "negative": negative,
                "negative_score": score,
                "negative_source": source,
            })
            source_counts[source] += 1

    output_file = Path(args.output_file)
    write_jsonl(triplets, output_file)
    metadata = {
        "data_file": str(Path(args.data_file).resolve()),
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "n_positive_pairs": len(positive_rows),
        "n_triplets": len(triplets),
        "coverage": len(triplets) / max(len(positive_rows), 1),
        "source_counts": dict(source_counts),
        "backend": backend_used,
        "allow_unlabeled_candidates": args.allow_unlabeled_candidates,
        "warning": (
            "unlabeled_retrieval 不是人工确认负例，训练前必须抽样复核。"
            if args.allow_unlabeled_candidates else None
        ),
    }
    save_json(metadata, output_file.with_suffix(".metadata.json"))
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
