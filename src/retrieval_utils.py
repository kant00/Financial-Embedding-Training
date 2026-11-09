"""
检索公共实现：文本编码、精确内积搜索与排序指标。

用途与调用位置：
    mine_hard_negatives.py 与 evaluate_retrieval.py 导入本模块，无独立命令行入口。
    encode_texts 批量调用双塔编码，得到句向量；模型由调用者构建并加载权重。
    search_inner_product 在 FAISS IndexFlatIP 或 NumPy 上执行精确搜索。
    ranking_metrics 计算 Recall/Hit/NDCG@K 和所给截断排名上的 MRR。

输入与输出：
    搜索输入为同维 query/corpus 向量；L2 归一化后内积才等价余弦。
    返回分数、候选索引和实际后端；指标依赖查询对应的相关文档集合。

注意事项：
    auto 仅在 FAISS ImportError 时回退，其他运行错误不会被静默吞掉。
    NumPy 构造完整 query×corpus 矩阵，不是分块或近似搜索。
    这里没有线上服务或索引持久化；完整操作见 USAGE_GUIDE.md 第 6、8 节。

说明核对：2026-09-28；完整流程、前置条件及自定义模型路径见 USAGE_GUIDE.md。
"""

from __future__ import annotations

import math
from collections import defaultdict

import numpy as np
import torch


@torch.no_grad()
def encode_texts(model, tokenizer, texts, device, batch_size=128, max_length=64):
    """以动态 padding 批量计算归一化句向量。

    用途：
        以动态 padding 批量计算归一化句向量。

    使用场景与调用位置：
        在 src/retrieval_utils.py 的“为难负样本挖掘和两阶段评估提供编码、搜索与排序指标”流程中使用。 跨文件调用：src/evaluate_retrieval.py:main()、src/mine_hard_negatives.py:main()。

    参数：
        model: 当前操作的模型实例；结构和设备须匹配输入。
        tokenizer: 与模型权重配套的 tokenizer；负责特殊 token、截断和 padding。
        texts: 按输出顺序排列的文本字符串列表。
        device: torch.device 或兼容设备描述，指定推理/训练位置。
        batch_size: 每次 DataLoader 取样或模型推理的样本数，应为正整数。
        max_length: 编码序列长度上限，含特殊 token；单句/句对语义见注意事项。

    返回值：
        连续 float32 NumPy 矩阵 [N,H]；空文本列表返回 [0,0]。

    注意事项：
        难负样本和 Recall–Rerank 主流程调用；模型切换 eval/no_grad，在 device 推理再移到 CPU。缺 token_type_ids 时补零；向量归一化由 model.encode 保证。
    """
    model.eval()
    matrices = []
    for start in range(0, len(texts), batch_size):
        batch_texts = texts[start:start + batch_size]
        encoded = tokenizer(
            batch_texts,
            max_length=max_length,
            truncation=True,
            padding=True,
            return_tensors="pt",
        )
        encoded = {key: value.to(device) for key, value in encoded.items()}
        if "token_type_ids" not in encoded:
            encoded["token_type_ids"] = torch.zeros_like(encoded["input_ids"])
        vectors = model.encode(**encoded).cpu().float().numpy()
        matrices.append(vectors)
    if not matrices:
        return np.empty((0, 0), dtype=np.float32)
    return np.ascontiguousarray(np.concatenate(matrices, axis=0), dtype=np.float32)


def search_inner_product(query_vectors, corpus_vectors, top_k, backend="auto"):
    """为每个 query 找到内积最高的 top_k 个候选。

    用途：
        为每个 query 找到内积最高的 top_k 个候选。

    使用场景与调用位置：
        在 src/retrieval_utils.py 的“为难负样本挖掘和两阶段评估提供编码、搜索与排序指标”流程中使用。 跨文件调用：src/evaluate_retrieval.py:main()、src/mine_hard_negatives.py:main()。

    参数：
        query_vectors: [Q,H] float32 查询向量矩阵。
        corpus_vectors: [N,H] float32 候选向量矩阵，与查询维度一致。
        top_k: 每个 query 保留的候选数，应大于 0。
        backend: 检索后端 auto/faiss/numpy。

    返回值：
        (scores [Q,K], indices [Q,K], backend_name)，索引指向原 corpus 行。

    注意事项：
        挖掘和检索评估调用；输入应同维且归一化，才等价余弦。K 限制为候选数；auto 在 FAISS ImportError 时回退 NumPy，其他错误不吞掉；NumPy 物化 [Q,N] 全矩阵，大规模需控制内存。
    """
    top_k = min(int(top_k), len(corpus_vectors))
    if top_k <= 0:
        raise ValueError("corpus 不能为空且 top_k 必须为正数")

    if backend in {"auto", "faiss"}:
        try:
            import faiss

            index = faiss.IndexFlatIP(corpus_vectors.shape[1])
            index.add(np.ascontiguousarray(corpus_vectors, dtype=np.float32))
            scores, indices = index.search(
                np.ascontiguousarray(query_vectors, dtype=np.float32), top_k
            )
            return scores, indices, "faiss.IndexFlatIP"
        except ImportError:
            if backend == "faiss":
                raise

    similarities = np.asarray(query_vectors) @ np.asarray(corpus_vectors).T
    partition = np.argpartition(-similarities, kth=top_k - 1, axis=1)[:, :top_k]
    partial_scores = np.take_along_axis(similarities, partition, axis=1)
    order = np.argsort(-partial_scores, axis=1)
    indices = np.take_along_axis(partition, order, axis=1)
    scores = np.take_along_axis(similarities, indices, axis=1)
    return scores, indices, "numpy_exact"


def ranking_metrics(rankings, relevant_by_query, cutoffs=(1, 5, 10, 50)):
    """根据排序列表与相关答案集合评估检索质量。

    用途：
        根据排序列表与相关答案集合评估检索质量。

    使用场景与调用位置：
        在 src/retrieval_utils.py 的“为难负样本挖掘和两阶段评估提供编码、搜索与排序指标”流程中使用。 跨文件调用：src/evaluate_retrieval.py:main()。

    参数：
        rankings: query 到按优先级排序的候选标识列表的映射。
        relevant_by_query: query 到相关候选标识集合的映射。
        cutoffs: 需要报告 Recall/NDCG 等指标的正整数 K 序列。

    返回值：
        含 n_queries、mrr、recall@k、hit_rate@k、ndcg@k 的字典。

    注意事项：
        evaluate_retrieval 对召回与重排结果各调用一次；无相关标注的 query 跳过，排名要求候选不重复；MRR 仅对传入截断列表计算，漏出列表的相关项记 0。所有 query 都无相关项时均值为 NaN。
    """
    if not rankings:
        raise ValueError("rankings 不能为空")
    cutoffs = sorted(set(int(k) for k in cutoffs if int(k) > 0))
    recall_values = defaultdict(list)
    hit_values = defaultdict(list)
    ndcg_values = defaultdict(list)
    reciprocal_ranks = []

    for query, ranking in rankings.items():
        relevant = set(relevant_by_query.get(query, set()))
        if not relevant:
            continue
        first_relevant_rank = None
        for rank, candidate in enumerate(ranking, start=1):
            if candidate in relevant:
                first_relevant_rank = rank
                break
        reciprocal_ranks.append(0.0 if first_relevant_rank is None else 1.0 / first_relevant_rank)

        for cutoff in cutoffs:
            top_candidates = ranking[:cutoff]
            hits = [1 if candidate in relevant else 0 for candidate in top_candidates]
            recall_values[cutoff].append(sum(hits) / len(relevant))
            hit_values[cutoff].append(float(any(hits)))
            dcg = sum(hit / math.log2(rank + 1) for rank, hit in enumerate(hits, start=1))
            ideal_hits = min(len(relevant), cutoff)
            ideal_dcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))
            ndcg_values[cutoff].append(dcg / ideal_dcg if ideal_dcg else 0.0)

    result = {"n_queries": len(reciprocal_ranks), "mrr": float(np.mean(reciprocal_ranks))}
    for cutoff in cutoffs:
        result[f"recall@{cutoff}"] = float(np.mean(recall_values[cutoff]))
        result[f"hit_rate@{cutoff}"] = float(np.mean(hit_values[cutoff]))
        result[f"ndcg@{cutoff}"] = float(np.mean(ndcg_values[cutoff]))
    return result
