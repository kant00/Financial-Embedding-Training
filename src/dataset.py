"""
数据集与 DataLoader：把句对或三元组转换为 BERT 训练输入。

用途与调用位置：
    由双塔/交叉塔训练、评估及错误分析模块导入；没有独立命令行流程。
    PairDataset：读取 sentence1/sentence2/label，用于 cosine 训练和双塔评估。
    PositivePairDataset：筛选正例句对，供 in-batch InfoNCE 构造 batch。
    TripletDataset：从正例构建 anchor/positive，优先选显式负例，否则从全局池采样。
    PrecomputedTripletDataset：读取 anchor/positive/negative，供 Hard Triplet 使用。
    CrossEncoderDataset：两句联合分词，生成交叉塔输入。

输入与输出：
    build_pair_loaders、build_crossencoder_loaders 读取 train/validation 别名。
    build_triplet_loader、build_inbatch_loader 提供对应目标的训练 DataLoader。
    分词生成 input_ids、attention_mask、token_type_ids；具体 batch 结构见各类说明。

注意事项：
    随机三元组在初始化时构建，epoch 间不刷新；全局随机候选不保证是真负例。
    数据文件有正例不代表需要额外人工标注才能训练 InfoNCE，但 batch 构造决定负例池。
    长度、padding、标签转换须与训练/评估保持一致；正式数据协议见 USAGE_GUIDE.md。

说明核对：2026-09-28；完整流程、前置条件及自定义模型路径见 USAGE_GUIDE.md。
"""

import json
import random
from collections import defaultdict
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset

random.seed(42)


# ── 工具函数 ──────────────────────────────────────────────────────────────

def load_jsonl(path):
    """读取 UTF-8 JSONL 并按文件顺序还原记录。

    用途：
        读取 UTF-8 JSONL 并按文件顺序还原记录。

    使用场景与调用位置：
        在 src/dataset.py 的“将 JSONL 句对转为训练与评估 DataLoader 可消费的张量”流程中使用。 本文件直接调用位置：CrossEncoderDataset.__init__()、PairDataset.__init__()、PrecomputedTripletDataset.__init__()、TripletDataset.__init__()。 跨文件调用：src/analyze_badcases.py:main()。

    参数：
        path: 待读取或写入的文件路径，具体模式见用途与注意事项。

    返回值：
        list[dict]；每个非空行对应一条记录。

    注意事项：
        用于将 JSONL 句对转为训练与评估 DataLoader 可消费的张量阶段的数据读取；忽略空行，不执行字段、标签或重复样本校验；缺失文件和非法 JSON 异常直接上抛，整个文件驻留内存。
    """
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def encode_single(tokenizer, text, max_length):
    """把单个句子编码为可直接输入 BERT 的字段。

    用途：
        把单个句子编码为可直接输入 BERT 的字段。

    使用场景与调用位置：
        在 src/dataset.py 的“将 JSONL 句对转为训练与评估 DataLoader 可消费的张量”流程中使用。 本文件直接调用位置：PairDataset.__getitem__()、PrecomputedTripletDataset.__getitem__()、TripletDataset.__getitem__()。

    参数：
        tokenizer: 与模型权重配套的 tokenizer；负责特殊 token、截断和 padding。
        text: 待规范化或编码的单个文本字符串。
        max_length: 编码序列长度上限，含特殊 token；单句/句对语义见注意事项。

    返回值：
        input_ids、attention_mask、token_type_ids 字典；每项为 [max_length] CPU 整数张量。

    注意事项：
        使用固定长度 padding 和 truncation；要求 tokenizer 返回 token_type_ids，因此不能无条件替换为任意模型 tokenizer。
    """
    enc = tokenizer(
        text,
        max_length=max_length,
        truncation=True,
        padding="max_length",
        return_tensors="pt",
    )
    return {
        "input_ids":      enc["input_ids"].squeeze(0),
        "attention_mask": enc["attention_mask"].squeeze(0),
        "token_type_ids": enc["token_type_ids"].squeeze(0),
    }


# ── PairDataset ───────────────────────────────────────────────────────────

class PairDataset(Dataset):
    """为 Cosine 训练和二分类评估提供独立编码的句对。

    用途：
        为 Cosine 训练和二分类评估提供独立编码的句对。

    使用场景与调用位置：
        在 src/dataset.py 的“将 JSONL 句对转为训练与评估 DataLoader 可消费的张量”流程中使用。 本文件直接调用位置：build_inbatch_loader()、build_pair_loaders()、build_triplet_loader()。 跨文件调用：src/analyze_badcases.py:main()、src/compare_methods.py:load_and_eval()、src/evaluate.py:main()。

    参数：
        构造参数见本类 __init__；实例成员与样本/模型结构见注意事项。

    返回值：
        Dataset 对象；由 DataLoader 通过长度和索引协议读取。

    注意事项：
        build_pair_loaders、评估和 bad case 脚本。self.rows 保存原始句对；每次索引才 tokenize，固定长度 padding；0/1 标签在训练器中转换为 Cosine 所需的 -1/+1。
    """

    def __init__(self, data_path, tokenizer, max_length=64):
        """初始化数据记录、tokenizer 和截断长度。

        用途：
            初始化数据记录、tokenizer 和截断长度。

        使用场景与调用位置：
            调用 PairDataset(...) 创建实例时自动执行；在训练或评估加载模型/数据的准备阶段使用。

        参数：
            data_path: 包含样本记录的 UTF-8 JSONL 文件路径。
            tokenizer: 与模型权重配套的 tokenizer；负责特殊 token、截断和 padding。
            max_length: 编码序列长度上限，含特殊 token；单句/句对语义见注意事项。

        返回值：
            None；修改当前实例的数据成员。

        注意事项：
            self.rows 保存原始句对；每次索引才 tokenize，固定长度 padding；0/1 标签在训练器中转换为 Cosine 所需的 -1/+1。
        """
        self.rows      = load_jsonl(data_path)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        """提供可索引训练/评估样本总数。

        用途：
            提供可索引训练/评估样本总数。

        使用场景与调用位置：
            通过 dataset[idx] / len(dataset) 或 DataLoader 的采样和组批协议隐式调用，所属类为 PairDataset。

        参数：
            无显式业务参数；实例方法通过 self 读取内部状态，入口函数从命令行获取配置。

        返回值：
            非负整数；句对数量。

        注意事项：
            DataLoader 的采样器、len(dataset) 和批次数计算隐式使用此方法；不执行分词。
        """
        return len(self.rows)

    def __getitem__(self, idx):
        """按索引取出一个样本并编码为 CPU 张量。

        用途：
            按索引取出一个样本并编码为 CPU 张量。

        使用场景与调用位置：
            通过 dataset[idx] / len(dataset) 或 DataLoader 的采样和组批协议隐式调用，所属类为 PairDataset。

        参数：
            idx: 样本在当前数据集中的整数索引。

        返回值：
            字典：两侧 _a/_b 编码 [L] 张量、标量 label，以及原文 text_a/text_b。

        注意事项：
            原文用于 InfoNCE 重复文本屏蔽；DataLoader 叠加张量为 [B,L]，将原文汇集为列表。 索引越界会沿用列表的 IndexError。
        """
        r = self.rows[idx]
        enc_a = encode_single(self.tokenizer, r["sentence1"], self.max_length)
        enc_b = encode_single(self.tokenizer, r["sentence2"], self.max_length)
        return {
            "input_ids_a":      enc_a["input_ids"],
            "attention_mask_a": enc_a["attention_mask"],
            "token_type_ids_a": enc_a["token_type_ids"],
            "input_ids_b":      enc_b["input_ids"],
            "attention_mask_b": enc_b["attention_mask"],
            "token_type_ids_b": enc_b["token_type_ids"],
            "label": torch.tensor(r["label"], dtype=torch.long),
            # 保留原文供 in-batch false-negative 去重；普通训练器会安全忽略。
            "text_a": r["sentence1"],
            "text_b": r["sentence2"],
        }


class PositivePairDataset(PairDataset):
    """筛选 label=1 句对供 in-batch InfoNCE 训练。

    用途：
        筛选 label=1 句对供 in-batch InfoNCE 训练。

    使用场景与调用位置：
        在 src/dataset.py 的“将 JSONL 句对转为训练与评估 DataLoader 可消费的张量”流程中使用。 本文件直接调用位置：build_inbatch_loader()。

    参数：
        构造参数见本类 __init__；实例成员与样本/模型结构见注意事项。

    返回值：
        Dataset 对象；由 DataLoader 通过长度和索引协议读取。

    注意事项：
        build_inbatch_loader。继承 PairDataset 的长度和取样逻辑；无正例时构造阶段抛出 ValueError。
    """

    def __init__(self, data_path, tokenizer, max_length=64):
        """初始化数据记录、tokenizer 和截断长度。

        用途：
            初始化数据记录、tokenizer 和截断长度。

        使用场景与调用位置：
            调用 PositivePairDataset(...) 创建实例时自动执行；在训练或评估加载模型/数据的准备阶段使用。

        参数：
            data_path: 包含样本记录的 UTF-8 JSONL 文件路径。
            tokenizer: 与模型权重配套的 tokenizer；负责特殊 token、截断和 padding。
            max_length: 编码序列长度上限，含特殊 token；单句/句对语义见注意事项。

        返回值：
            None；修改当前实例的数据成员。

        注意事项：
            先调用 PairDataset 初始化，再过滤正例；没有正例会抛出 ValueError。
        """
        super().__init__(data_path, tokenizer, max_length)
        self.rows = [row for row in self.rows if int(row["label"]) == 1]
        if not self.rows:
            raise ValueError(f"{data_path} 中没有正例，无法进行 in-batch 对比学习")


# ── TripletDataset ────────────────────────────────────────────────────────

class TripletDataset(Dataset):
    """从正例对与显式/随机负例构建静态三元组。

    用途：
        从正例对与显式/随机负例构建静态三元组。

    使用场景与调用位置：
        在 src/dataset.py 的“将 JSONL 句对转为训练与评估 DataLoader 可消费的张量”流程中使用。 本文件直接调用位置：build_triplet_loader()。

    参数：
        构造参数见本类 __init__；实例成员与样本/模型结构见注意事项。

    返回值：
        Dataset 对象；由 DataLoader 通过长度和索引协议读取。

    注意事项：
        build_triplet_loader 未指定 hard_negative_file 时。self.triplets 在初始化时生成，epoch 间不刷新；全局随机候选并非一定是真负例。
    """

    def __init__(self, data_path, tokenizer, max_length=64):
        """初始化数据记录、tokenizer 和截断长度。

        用途：
            初始化数据记录、tokenizer 和截断长度。

        使用场景与调用位置：
            调用 TripletDataset(...) 创建实例时自动执行；在训练或评估加载模型/数据的准备阶段使用。

        参数：
            data_path: 包含样本记录的 UTF-8 JSONL 文件路径。
            tokenizer: 与模型权重配套的 tokenizer；负责特殊 token、截断和 padding。
            max_length: 编码序列长度上限，含特殊 token；单句/句对语义见注意事项。

        返回值：
            None；修改当前实例的数据成员。

        注意事项：
            立即读取文件并调用 _build_triplets；负采样依赖 Python random 状态。
        """
        self.tokenizer  = tokenizer
        self.max_length = max_length
        self.triplets   = self._build_triplets(load_jsonl(data_path))

    def _build_triplets(self, rows):
        # 为每个句子建立"其负样本列表"索引
        """为每个正例挑选一个负例，生成离线静态三元组。

        用途：
            为每个正例挑选一个负例，生成离线静态三元组。

        使用场景与调用位置：
            在 src/dataset.py 的“将 JSONL 句对转为训练与评估 DataLoader 可消费的张量”流程中使用。 所属作用域：TripletDataset。

        参数：
            rows: 原始记录列表；句对通常包含 sentence1、sentence2、label。

        返回值：
            (anchor, positive, negative) 元组列表。

        注意事项：
            优先取同 anchor 显式负例，否则从所有句子中随机取非 anchor/positive 的文本；不排除全部潜在同义句；候选不足时 while 循环可能无法结束。
        """
        neg_by_sent = defaultdict(list)
        all_sents   = set()
        for r in rows:
            all_sents.add(r["sentence1"])
            all_sents.add(r["sentence2"])
            if r["label"] == 0:
                neg_by_sent[r["sentence1"]].append(r["sentence2"])
                neg_by_sent[r["sentence2"]].append(r["sentence1"])

        global_pool = list(all_sents)

        triplets = []
        for r in rows:
            if r["label"] != 1:
                continue
            anchor   = r["sentence1"]
            positive = r["sentence2"]

            negs = neg_by_sent.get(anchor, [])
            if negs:
                negative = random.choice(negs)
            else:
                # anchor 没有自身的负样本 → 从全局随机选
                negative = anchor
                while negative in (anchor, positive):
                    negative = random.choice(global_pool)

            triplets.append((anchor, positive, negative))

        print(f"  TripletDataset: 构建 {len(triplets):,} 个三元组")
        return triplets

    def __len__(self):
        """提供可索引训练/评估样本总数。

        用途：
            提供可索引训练/评估样本总数。

        使用场景与调用位置：
            通过 dataset[idx] / len(dataset) 或 DataLoader 的采样和组批协议隐式调用，所属类为 TripletDataset。

        参数：
            无显式业务参数；实例方法通过 self 读取内部状态，入口函数从命令行获取配置。

        返回值：
            非负整数；三元组数量。

        注意事项：
            DataLoader 的采样器、len(dataset) 和批次数计算隐式使用此方法；不执行分词。
        """
        return len(self.triplets)

    def __getitem__(self, idx):
        """按索引取出一个样本并编码为 CPU 张量。

        用途：
            按索引取出一个样本并编码为 CPU 张量。

        使用场景与调用位置：
            通过 dataset[idx] / len(dataset) 或 DataLoader 的采样和组批协议隐式调用，所属类为 TripletDataset。

        参数：
            idx: 样本在当前数据集中的整数索引。

        返回值：
            字典：input_ids/attention_mask/token_type_ids 各含 _a、_p、_n 后缀，共 9 个 [L] 整数张量。

        注意事项：
            a/p/n 分别为 anchor/positive/negative；不返回分类标签，Triplet loss 由三者相对位置定义。 索引越界会沿用列表的 IndexError。
        """
        anchor, positive, negative = self.triplets[idx]
        enc_a = encode_single(self.tokenizer, anchor,   self.max_length)
        enc_p = encode_single(self.tokenizer, positive, self.max_length)
        enc_n = encode_single(self.tokenizer, negative, self.max_length)
        return {
            "input_ids_a":      enc_a["input_ids"],
            "attention_mask_a": enc_a["attention_mask"],
            "token_type_ids_a": enc_a["token_type_ids"],
            "input_ids_p":      enc_p["input_ids"],
            "attention_mask_p": enc_p["attention_mask"],
            "token_type_ids_p": enc_p["token_type_ids"],
            "input_ids_n":      enc_n["input_ids"],
            "attention_mask_n": enc_n["attention_mask"],
            "token_type_ids_n": enc_n["token_type_ids"],
        }


class PrecomputedTripletDataset(Dataset):
    """读取离线挖掘后的三元组并独立编码三侧文本。

    用途：
        读取离线挖掘后的三元组并独立编码三侧文本。

    使用场景与调用位置：
        在 src/dataset.py 的“将 JSONL 句对转为训练与评估 DataLoader 可消费的张量”流程中使用。 本文件直接调用位置：build_triplet_loader()。

    参数：
        构造参数见本类 __init__；实例成员与样本/模型结构见注意事项。

    返回值：
        Dataset 对象；由 DataLoader 通过长度和索引协议读取。

    注意事项：
        build_triplet_loader 指定 hard_negative_file 时。记录需含 anchor/positive/negative；当前只对第一行检查字段；分数和来源元数据不参与 loss。
    """

    def __init__(self, data_path, tokenizer, max_length=64):
        """初始化数据记录、tokenizer 和截断长度。

        用途：
            初始化数据记录、tokenizer 和截断长度。

        使用场景与调用位置：
            调用 PrecomputedTripletDataset(...) 创建实例时自动执行；在训练或评估加载模型/数据的准备阶段使用。

        参数：
            data_path: 包含样本记录的 UTF-8 JSONL 文件路径。
            tokenizer: 与模型权重配套的 tokenizer；负责特殊 token、截断和 padding。
            max_length: 编码序列长度上限，含特殊 token；单句/句对语义见注意事项。

        返回值：
            None；修改当前实例的数据成员。

        注意事项：
            记录需含 anchor/positive/negative；当前只对第一行检查字段；分数和来源元数据不参与 loss。
        """
        self.rows = load_jsonl(data_path)
        self.tokenizer = tokenizer
        self.max_length = max_length
        required = {"anchor", "positive", "negative"}
        if self.rows and not required.issubset(self.rows[0]):
            raise ValueError(f"三元组文件缺少字段: {sorted(required)}")

    def __len__(self):
        """提供可索引训练/评估样本总数。

        用途：
            提供可索引训练/评估样本总数。

        使用场景与调用位置：
            通过 dataset[idx] / len(dataset) 或 DataLoader 的采样和组批协议隐式调用，所属类为 PrecomputedTripletDataset。

        参数：
            无显式业务参数；实例方法通过 self 读取内部状态，入口函数从命令行获取配置。

        返回值：
            非负整数；三元组数量。

        注意事项：
            DataLoader 的采样器、len(dataset) 和批次数计算隐式使用此方法；不执行分词。
        """
        return len(self.rows)

    def __getitem__(self, idx):
        """按索引取出一个样本并编码为 CPU 张量。

        用途：
            按索引取出一个样本并编码为 CPU 张量。

        使用场景与调用位置：
            通过 dataset[idx] / len(dataset) 或 DataLoader 的采样和组批协议隐式调用，所属类为 PrecomputedTripletDataset。

        参数：
            idx: 样本在当前数据集中的整数索引。

        返回值：
            字典：input_ids/attention_mask/token_type_ids 各含 _a、_p、_n 后缀，共 9 个 [L] 整数张量。

        注意事项：
            a/p/n 分别为 anchor/positive/negative；不返回分类标签，Triplet loss 由三者相对位置定义。 索引越界会沿用列表的 IndexError。
        """
        row = self.rows[idx]
        enc_a = encode_single(self.tokenizer, row["anchor"], self.max_length)
        enc_p = encode_single(self.tokenizer, row["positive"], self.max_length)
        enc_n = encode_single(self.tokenizer, row["negative"], self.max_length)
        return {
            "input_ids_a": enc_a["input_ids"],
            "attention_mask_a": enc_a["attention_mask"],
            "token_type_ids_a": enc_a["token_type_ids"],
            "input_ids_p": enc_p["input_ids"],
            "attention_mask_p": enc_p["attention_mask"],
            "token_type_ids_p": enc_p["token_type_ids"],
            "input_ids_n": enc_n["input_ids"],
            "attention_mask_n": enc_n["attention_mask"],
            "token_type_ids_n": enc_n["token_type_ids"],
        }


# ── CrossEncoderDataset ───────────────────────────────────────────────────

class CrossEncoderDataset(Dataset):
    """将两句拼接为 BERT 单条输入序列。

    用途：
        将两句拼接为 BERT 单条输入序列。

    使用场景与调用位置：
        在 src/dataset.py 的“将 JSONL 句对转为训练与评估 DataLoader 可消费的张量”流程中使用。 本文件直接调用位置：build_crossencoder_loaders()。 跨文件调用：src/analyze_badcases.py:main()、src/compare_methods.py:load_and_eval()、src/evaluate.py:main()。

    参数：
        构造参数见本类 __init__；实例成员与样本/模型结构见注意事项。

    返回值：
        Dataset 对象；由 DataLoader 通过长度和索引协议读取。

    注意事项：
        build_crossencoder_loaders、Cross-Encoder 评估及错误分析。token_type_ids 区分句子 A/B；max_length 是整个句对的总长度，而非每句长度。
    """

    def __init__(self, data_path, tokenizer, max_length=128):
        """初始化数据记录、tokenizer 和截断长度。

        用途：
            初始化数据记录、tokenizer 和截断长度。

        使用场景与调用位置：
            调用 CrossEncoderDataset(...) 创建实例时自动执行；在训练或评估加载模型/数据的准备阶段使用。

        参数：
            data_path: 包含样本记录的 UTF-8 JSONL 文件路径。
            tokenizer: 与模型权重配套的 tokenizer；负责特殊 token、截断和 padding。
            max_length: 编码序列长度上限，含特殊 token；单句/句对语义见注意事项。

        返回值：
            None；修改当前实例的数据成员。

        注意事项：
            token_type_ids 区分句子 A/B；max_length 是整个句对的总长度，而非每句长度。
        """
        self.rows       = load_jsonl(data_path)
        self.tokenizer  = tokenizer
        self.max_length = max_length

    def __len__(self):
        """提供可索引训练/评估样本总数。

        用途：
            提供可索引训练/评估样本总数。

        使用场景与调用位置：
            通过 dataset[idx] / len(dataset) 或 DataLoader 的采样和组批协议隐式调用，所属类为 CrossEncoderDataset。

        参数：
            无显式业务参数；实例方法通过 self 读取内部状态，入口函数从命令行获取配置。

        返回值：
            非负整数；句对数量。

        注意事项：
            DataLoader 的采样器、len(dataset) 和批次数计算隐式使用此方法；不执行分词。
        """
        return len(self.rows)

    def __getitem__(self, idx):
        """按索引取出一个样本并编码为 CPU 张量。

        用途：
            按索引取出一个样本并编码为 CPU 张量。

        使用场景与调用位置：
            通过 dataset[idx] / len(dataset) 或 DataLoader 的采样和组批协议隐式调用，所属类为 CrossEncoderDataset。

        参数：
            idx: 样本在当前数据集中的整数索引。

        返回值：
            字典：input_ids、attention_mask、token_type_ids 三个 [L] 张量及标量 label。

        注意事项：
            两句联合分词，包含 CLS/SEP；采用固定总长 padding 和截断，DataLoader 产生 [B,L] 输入。 索引越界会沿用列表的 IndexError。
        """
        r = self.rows[idx]
        enc = self.tokenizer(
            r["sentence1"],
            r["sentence2"],
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        return {
            "input_ids":      enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "token_type_ids": enc["token_type_ids"].squeeze(0),
            "label": torch.tensor(r["label"], dtype=torch.long),
        }


# ── DataLoader 工厂函数 ───────────────────────────────────────────────────

def build_pair_loaders(data_dir, tokenizer, max_length=64, batch_size=32):
    """构建相应训练方式的 DataLoader。

    用途：
        构建相应训练方式的 DataLoader。

    使用场景与调用位置：
        在 src/dataset.py 的“将 JSONL 句对转为训练与评估 DataLoader 可消费的张量”流程中使用。 跨文件调用：src/train_biencoder.py:main()。

    参数：
        data_dir: 数据目录，包含调用流程所需的 train/validation/test 等 JSONL。
        tokenizer: 与模型权重配套的 tokenizer；负责特殊 token、截断和 padding。
        max_length: 编码序列长度上限，含特殊 token；单句/句对语义见注意事项。
        batch_size: 每次 DataLoader 取样或模型推理的样本数，应为正整数。

    返回值：
        (train_loader, val_loader, test_loader)。

    注意事项：
        PairDataset；训练打乱，评估保持顺序。 num_workers=0；validation/test 的语义由传入目录决定，新协议分别为 calibration/final test。原始官方 test 为 -1，禁止用于有监督评分。
    """
    data_dir = Path(data_dir)
    train_ds = PairDataset(data_dir / "train.jsonl",      tokenizer, max_length)
    val_ds   = PairDataset(data_dir / "validation.jsonl", tokenizer, max_length)
    test_ds  = PairDataset(data_dir / "test.jsonl",       tokenizer, max_length)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, num_workers=0)

    print(f"  train : {len(train_ds):>7,} 条, {len(train_loader):>5} batch")
    print(f"  val   : {len(val_ds):>7,} 条, {len(val_loader):>5} batch")
    print(f"  test  : {len(test_ds):>7,} 条, {len(test_loader):>5} batch  (AFQMC test 无正样本，仅供参考)")
    return train_loader, val_loader, test_loader


def build_triplet_loader(
    data_dir, tokenizer, max_length=64, batch_size=32, hard_negative_file=None
):
    """构建相应训练方式的 DataLoader。

    用途：
        构建相应训练方式的 DataLoader。

    使用场景与调用位置：
        在 src/dataset.py 的“将 JSONL 句对转为训练与评估 DataLoader 可消费的张量”流程中使用。 跨文件调用：src/train_biencoder.py:main()。

    参数：
        data_dir: 数据目录，包含调用流程所需的 train/validation/test 等 JSONL。
        tokenizer: 与模型权重配套的 tokenizer；负责特殊 token、截断和 padding。
        max_length: 编码序列长度上限，含特殊 token；单句/句对语义见注意事项。
        batch_size: 每次 DataLoader 取样或模型推理的样本数，应为正整数。
        hard_negative_file: 可选离线三元组 JSONL 路径；None 表示使用随机三元组。

    返回值：
        (train_loader, val_loader)。

    注意事项：
        hard_negative_file 为真时读取预计算三元组，否则使用静态随机三元组；验证仍使用 PairDataset。 num_workers=0；validation/test 的语义由传入目录决定，新协议分别为 calibration/final test。原始官方 test 为 -1，禁止用于有监督评分。
    """
    data_dir = Path(data_dir)
    train_ds = (
        PrecomputedTripletDataset(hard_negative_file, tokenizer, max_length)
        if hard_negative_file
        else TripletDataset(data_dir / "train.jsonl", tokenizer, max_length)
    )
    val_ds   = PairDataset(data_dir / "validation.jsonl", tokenizer, max_length)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=0)

    print(f"  triplet train : {len(train_ds):>7,} 三元组, {len(train_loader):>5} batch")
    print(f"  val (pair)    : {len(val_ds):>7,} 对,     {len(val_loader):>5} batch")
    return train_loader, val_loader


def build_inbatch_loader(data_dir, tokenizer, max_length=64, batch_size=32):
    """构建相应训练方式的 DataLoader。

    用途：
        构建相应训练方式的 DataLoader。

    使用场景与调用位置：
        在 src/dataset.py 的“将 JSONL 句对转为训练与评估 DataLoader 可消费的张量”流程中使用。 跨文件调用：src/train_biencoder.py:main()。

    参数：
        data_dir: 数据目录，包含调用流程所需的 train/validation/test 等 JSONL。
        tokenizer: 与模型权重配套的 tokenizer；负责特殊 token、截断和 padding。
        max_length: 编码序列长度上限，含特殊 token；单句/句对语义见注意事项。
        batch_size: 每次 DataLoader 取样或模型推理的样本数，应为正整数。

    返回值：
        (train_loader, val_loader)。

    注意事项：
        训练只含正例，验证包含全部标签；drop_last=False，最后可能出现只有一个样本的 batch，此时没有有效 batch 内负例。 num_workers=0；validation/test 的语义由传入目录决定，新协议分别为 calibration/final test。原始官方 test 为 -1，禁止用于有监督评分。
    """
    data_dir = Path(data_dir)
    train_ds = PositivePairDataset(data_dir / "train.jsonl", tokenizer, max_length)
    val_ds = PairDataset(data_dir / "validation.jsonl", tokenizer, max_length)
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=False
    )
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    print(f"  in-batch train : {len(train_ds):>7,} 正例对, {len(train_loader):>5} batch")
    print(f"  val (pair)     : {len(val_ds):>7,} 对,   {len(val_loader):>5} batch")
    return train_loader, val_loader


def build_crossencoder_loaders(data_dir, tokenizer, max_length=128, batch_size=32):
    """构建相应训练方式的 DataLoader。

    用途：
        构建相应训练方式的 DataLoader。

    使用场景与调用位置：
        在 src/dataset.py 的“将 JSONL 句对转为训练与评估 DataLoader 可消费的张量”流程中使用。 跨文件调用：src/train_crossencoder.py:main()。

    参数：
        data_dir: 数据目录，包含调用流程所需的 train/validation/test 等 JSONL。
        tokenizer: 与模型权重配套的 tokenizer；负责特殊 token、截断和 padding。
        max_length: 编码序列长度上限，含特殊 token；单句/句对语义见注意事项。
        batch_size: 每次 DataLoader 取样或模型推理的样本数，应为正整数。

    返回值：
        (train_loader, val_loader, test_loader)。

    注意事项：
        使用联合编码 CrossEncoderDataset；max_length 约束两句总长。 num_workers=0；validation/test 的语义由传入目录决定，新协议分别为 calibration/final test。原始官方 test 为 -1，禁止用于有监督评分。
    """
    data_dir = Path(data_dir)
    train_ds = CrossEncoderDataset(data_dir / "train.jsonl",      tokenizer, max_length)
    val_ds   = CrossEncoderDataset(data_dir / "validation.jsonl", tokenizer, max_length)
    test_ds  = CrossEncoderDataset(data_dir / "test.jsonl",       tokenizer, max_length)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, num_workers=0)

    print(f"  train : {len(train_ds):>7,} 条, {len(train_loader):>5} batch")
    print(f"  val   : {len(val_ds):>7,} 条, {len(val_loader):>5} batch")
    print(f"  test  : {len(test_ds):>7,} 条, {len(test_loader):>5} batch")
    return train_loader, val_loader, test_loader
