"""
两阶段离线检索评估：双塔召回与可选交叉塔精排。

用途与流程位置：
    在模型和检索配置冻结后，用统一 manifest 评估句对派生的检索任务。
    候选库为 sentence2 去重集合，查询及相关性集合来自已标注正例。
    精排仅重排召回列表的前 rerank_top_k，保留剩余候选顺序。

输入与输出：
    需要 --biencoder_checkpoint；--crossencoder_checkpoint 可省略以仅测召回。
    --rerank_top_k 不得大于 --recall_top_k；长度须与训练一致。
    输出 retrieval_metrics.json 与 retrieval_rankings.jsonl，包含 Recall/Hit/NDCG@K、
    截断排名上的 MRR、规模、检索后端及阶段耗时。

使用方式（在项目根目录执行，Bash）：
    python src/evaluate_retrieval.py --manifest data/experiments/afqmc_seed42/evaluation_manifest.jsonl --biencoder_checkpoint outputs/experiment_suite/cosine/seed_42/checkpoints/biencoder_cosine_best.pt --crossencoder_checkpoint outputs/experiment_suite/crossencoder/seed_42/checkpoints/crossencoder_best.pt --output_dir outputs/retrieval --recall_top_k 50 --rerank_top_k 20 --backend auto

注意事项：
    句对标签不是完整检索相关性标注；结果不能直接当作完整基准或线上效果。
    阶段耗时不是 QPS/P95 压测；不保存持久化向量索引。
    NumPy 后端物化完整 query×corpus 矩阵，需评估内存。

说明核对：2026-09-28；完整流程、前置条件及自定义模型路径见 USAGE_GUIDE.md。
"""

from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from transformers import BertTokenizer

from experiment_utils import DEFAULT_BERT_MODEL, environment_snapshot, save_json, write_jsonl
from model import build_biencoder, build_crossencoder
from retrieval_utils import encode_texts, ranking_metrics, search_inner_product


ROOT = Path(__file__).parent.parent
DEFAULT_MANIFEST = ROOT / "data" / "experiments" / "afqmc_seed42" / "evaluation_manifest.jsonl"


def load_jsonl(path):
    """读取 UTF-8 JSONL 并按文件顺序还原记录。

    用途：
        读取 UTF-8 JSONL 并按文件顺序还原记录。

    使用场景与调用位置：
        在 src/evaluate_retrieval.py 的“评估 Bi-Encoder 召回及可选 Cross-Encoder 精排”流程中使用。 本文件直接调用位置：main()。

    参数：
        path: 待读取或写入的文件路径，具体模式见用途与注意事项。

    返回值：
        list[dict]；每个非空行对应一条记录。

    注意事项：
        用于评估 Bi-Encoder 召回及可选 Cross-Encoder 精排阶段的数据读取；忽略空行，不执行字段、标签或重复样本校验；缺失文件和非法 JSON 异常直接上抛，整个文件驻留内存。
    """
    with open(path, encoding="utf-8") as file_obj:
        return [json.loads(line) for line in file_obj if line.strip()]


@torch.no_grad()
def crossencoder_scores(model, tokenizer, query, candidates, device, batch_size, max_length):
    """为一个 query 与一批候选计算交叉编码器匹配分数。

    用途：
        为一个 query 与一批候选计算交叉编码器匹配分数。

    使用场景与调用位置：
        在 src/evaluate_retrieval.py 的“评估 Bi-Encoder 召回及可选 Cross-Encoder 精排”流程中使用。 本文件直接调用位置：main()。

    参数：
        model: 当前操作的模型实例；结构和设备须匹配输入。
        tokenizer: 与模型权重配套的 tokenizer；负责特殊 token、截断和 padding。
        query: 单个查询文本。
        candidates: 与 query 配对打分的候选文本列表。
        device: torch.device 或兼容设备描述，指定推理/训练位置。
        batch_size: 每次 DataLoader 取样或模型推理的样本数，应为正整数。
        max_length: 编码序列长度上限，含特殊 token；单句/句对语义见注意事项。

    返回值：
        与 candidates 顺序一致的正类概率列表；空候选返回 []。

    注意事项：
        main 精排召回列表前 rerank_top_k 时调用；联合 tokenize 并动态 padding，总长受 max_length 限制；eval/no_grad，不对候选排序，外层根据返回分数排序。
    """
    model.eval()
    probabilities = []
    for start in range(0, len(candidates), batch_size):
        batch_candidates = candidates[start:start + batch_size]
        encoded = tokenizer(
            [query] * len(batch_candidates),
            batch_candidates,
            max_length=max_length,
            truncation=True,
            padding=True,
            return_tensors="pt",
        )
        encoded = {key: value.to(device) for key, value in encoded.items()}
        if "token_type_ids" not in encoded:
            encoded["token_type_ids"] = torch.zeros_like(encoded["input_ids"])
        logits = model(
            encoded["input_ids"], encoded["attention_mask"], encoded["token_type_ids"]
        )
        probabilities.extend(torch.softmax(logits, dim=-1)[:, 1].cpu().tolist())
    return probabilities


def parse_cutoffs(value):
    """解析排序评估所需的多个 K 值。

    用途：
        解析排序评估所需的多个 K 值。

    使用场景与调用位置：
        在 src/evaluate_retrieval.py 的“评估 Bi-Encoder 召回及可选 Cross-Encoder 精排”流程中使用。

    参数：
        value: 待转换、解析或格式化的输入值。

    返回值：
        去重、排序后仅保留正整数的 tuple。

    注意事项：
        parse_args 用作 --cutoffs 的 type；非法整数触发 ValueError，空内容或全非正数不会自动提供默认有效 K。
    """
    return tuple(sorted({int(item) for item in value.split(",") if int(item) > 0}))


def parse_args():
    """解析评估 Bi-Encoder 召回及可选 Cross-Encoder 精排所需的命令行配置。

    用途：
        解析评估 Bi-Encoder 召回及可选 Cross-Encoder 精排所需的命令行配置。

    使用场景与调用位置：
        本文件 main() 的启动配置阶段调用；独立运行脚本时首先经过此处。 本文件直接调用位置：main()。

    参数：
        无显式参数；所需配置由内部逻辑、模块常量或命令行提供。

    返回值：
        argparse.Namespace；成员名对应下方 add_argument 定义。

    注意事项：
        由本文件 main 在任何主要工作开始前调用；读取 sys.argv，处理默认值、类型和 choices；--help 或不合法参数通过 argparse 触发 SystemExit，不启动训练或推理。
    """
    parser = argparse.ArgumentParser(description="Recall-Rerank 检索评估")
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--biencoder_checkpoint", required=True)
    parser.add_argument("--crossencoder_checkpoint", default=None)
    parser.add_argument("--bert_path", default=DEFAULT_BERT_MODEL)
    parser.add_argument("--output_dir", default=str(ROOT / "outputs" / "retrieval_eval"))
    parser.add_argument("--recall_top_k", type=int, default=50)
    parser.add_argument("--rerank_top_k", type=int, default=20)
    parser.add_argument("--cutoffs", type=parse_cutoffs, default=(1, 5, 10, 50))
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--max_length", type=int, default=64)
    parser.add_argument("--cross_max_length", type=int, default=128)
    parser.add_argument("--backend", choices=["auto", "faiss", "numpy"], default="auto")
    return parser.parse_args()


def main():
    """组织本脚本的完整流程：评估 Bi-Encoder 召回及可选 Cross-Encoder 精排。

    用途：
        组织本脚本的完整流程：评估 Bi-Encoder 召回及可选 Cross-Encoder 精排。

    使用场景与调用位置：
        直接运行此文件时由 if __name__ == "__main__" 入口调用；也可显式调用 main()，此时仍读取进程命令行。

    参数：
        无显式参数；所需配置由内部逻辑、模块常量或命令行提供。

    返回值：
        None；产物及副作用见注意事项，配置由命令行读取。

    注意事项：
        从 manifest 的 sentence2 建候选库，仅评估有正例的 query；未标注相关项未知，属于 proxy benchmark。精排只改前 R 个，保留尾部顺序。耗时是整批平均统计，不等同在线 p95。
    """
    args = parse_args()
    if args.rerank_top_k > args.recall_top_k:
        raise ValueError("rerank_top_k 不能大于 recall_top_k")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows = load_jsonl(args.manifest)

    relevant_by_query = defaultdict(set)
    corpus = []
    seen_candidates = set()
    for row in rows:
        candidate = row["sentence2"]
        if candidate not in seen_candidates:
            seen_candidates.add(candidate)
            corpus.append(candidate)
        if int(row["label"]) == 1:
            relevant_by_query[row["sentence1"]].add(candidate)
    queries = sorted(relevant_by_query)

    tokenizer = BertTokenizer.from_pretrained(args.bert_path)
    bi_checkpoint = torch.load(
        args.biencoder_checkpoint, map_location=device, weights_only=False
    )
    bi_args = bi_checkpoint.get("args", {})
    biencoder = build_biencoder(
        args.bert_path,
        pool=bi_args.get("pool", "mean"),
        num_hidden_layers=bi_args.get("num_hidden_layers"),
    ).to(device)
    biencoder.load_state_dict(bi_checkpoint["state_dict"])

    encode_start = time.perf_counter()
    corpus_vectors = encode_texts(
        biencoder, tokenizer, corpus, device, args.batch_size, args.max_length
    )
    query_vectors = encode_texts(
        biencoder, tokenizer, queries, device, args.batch_size, args.max_length
    )
    encode_seconds = time.perf_counter() - encode_start
    search_start = time.perf_counter()
    scores, indices, backend_used = search_inner_product(
        query_vectors, corpus_vectors, args.recall_top_k, args.backend
    )
    search_seconds = time.perf_counter() - search_start

    recall_rankings = {
        query: [corpus[index] for index in row_indices]
        for query, row_indices in zip(queries, indices)
    }
    recall_metrics = ranking_metrics(recall_rankings, relevant_by_query, args.cutoffs)

    crossencoder = None
    rerank_rankings = {}
    rerank_seconds = 0.0
    if args.crossencoder_checkpoint:
        cross_checkpoint = torch.load(
            args.crossencoder_checkpoint, map_location=device, weights_only=False
        )
        cross_args = cross_checkpoint.get("args", {})
        crossencoder = build_crossencoder(
            args.bert_path,
            num_hidden_layers=cross_args.get("num_hidden_layers"),
        ).to(device)
        crossencoder.load_state_dict(cross_checkpoint["state_dict"])

        rerank_start = time.perf_counter()
        for query in queries:
            recall_candidates = recall_rankings[query]
            head = recall_candidates[:args.rerank_top_k]
            tail = recall_candidates[args.rerank_top_k:]
            cross_scores = crossencoder_scores(
                crossencoder, tokenizer, query, head, device,
                args.batch_size, args.cross_max_length,
            )
            # sorted 的匿名键读取 (candidate, score) 的第二项；按正类概率降序，
            # 仅影响召回头部的精排顺序，不扩展候选集合。
            sorted_head = [
                candidate for candidate, _ in sorted(
                    zip(head, cross_scores), key=lambda item: item[1], reverse=True
                )
            ]
            # 只精排 head，tail 保持召回阶段顺序，符合线上常见实现。
            rerank_rankings[query] = sorted_head + tail
        rerank_seconds = time.perf_counter() - rerank_start

    rerank_metrics = (
        ranking_metrics(rerank_rankings, relevant_by_query, args.cutoffs)
        if rerank_rankings else None
    )
    result = {
        "protocol": {
            "manifest": str(Path(args.manifest).resolve()),
            "n_queries": len(queries),
            "n_candidates": len(corpus),
            "relevance_note": (
                "只把清单中 label=1 的句对视为相关；未标注候选可能包含潜在相关项，"
                "因此这是受控 proxy retrieval benchmark。"
            ),
        },
        "recall": recall_metrics,
        "rerank": rerank_metrics,
        "latency": {
            "encode_total_seconds": encode_seconds,
            "search_total_seconds": search_seconds,
            "search_ms_per_query": search_seconds * 1000 / max(len(queries), 1),
            "rerank_total_seconds": rerank_seconds,
            "rerank_ms_per_query": rerank_seconds * 1000 / max(len(queries), 1),
        },
        "backend": backend_used,
        "config": vars(args),
        "environment": environment_snapshot(),
    }
    output_dir = Path(args.output_dir)
    save_json(result, output_dir / "retrieval_metrics.json")
    details = []
    for query in queries:
        details.append({
            "query": query,
            "relevant": sorted(relevant_by_query[query]),
            "recall_ranking": recall_rankings[query],
            "rerank_ranking": rerank_rankings.get(query),
        })
    write_jsonl(details, output_dir / "retrieval_rankings.jsonl")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
