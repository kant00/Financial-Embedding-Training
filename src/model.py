"""
模型定义：共享骨干双塔与联合编码交叉塔。

用途与调用位置：
    由训练、评估、难负挖掘和检索模块导入，无独立训练或预测命令。
    build_biencoder / build_crossencoder 负责构造模型并打印参数规模。

模型接口：
    BiEncoder.encode 接收单句 batch，池化后输出 L2 归一化向量。
    BiEncoder.forward 接收两侧 batch，返回两个向量集合；余弦/点积由调用方计算。
    CrossEncoder.forward 接收联合句对 batch，返回 [B, 2] logits，不直接返回概率。
    需要概率时由调用方 softmax；分类可取 argmax，精排可使用正例概率。

配置与限制：
    双塔支持 cls/mean/max 池化；num_hidden_layers 控制加载的骨干层数。
    缩短层数可以改变计算量，但不保证 4 层耗时恰为 12 层的三分之一。
    工厂函数只构造模型；训练后的能力来自另行加载的 checkpoint state_dict。
    推理需 model.eval() 并关闭梯度，模型与输入须在同一设备。
    根目录可运行的句对推理示例见 USAGE_GUIDE.md 第 5.4 节。

说明核对：2026-09-28；完整流程、前置条件及自定义模型路径见 USAGE_GUIDE.md。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers
from transformers import BertConfig, BertModel


# ── BiEncoder ─────────────────────────────────────────────────────────────

class BiEncoder(nn.Module):
    """将两句话映射到共享语义向量空间。

    用途：
        将两句话映射到共享语义向量空间。

    使用场景与调用位置：
        在 src/model.py 的“定义共享 BERT 双编码器和句对交互分类器”流程中使用。 本文件直接调用位置：build_biencoder()。

    参数：
        构造参数见本类 __init__；实例成员与样本/模型结构见注意事项。

    返回值：
        torch.nn.Module；encode 返回单侧向量，forward 返回两侧向量。

    注意事项：
        由 build_biencoder 创建；用于三种双塔损失、二分类评估、语料编码和召回。共享 BERT→pooling→Dropout→L2 normalize；可预计算候选向量。
    """

    def __init__(self, bert_path, pool="mean", dropout=0.1, num_hidden_layers=None):
        """加载 BERT 预训练骨干并建立池化和 Dropout 配置。

        用途：
            加载 BERT 预训练骨干并建立池化和 Dropout 配置。

        使用场景与调用位置：
            调用 BiEncoder(...) 创建实例时自动执行；在训练或评估加载模型/数据的准备阶段使用。

        参数：
            bert_path: BERT 本地预训练目录或库支持的模型标识。
            pool: 句向量池化方式，支持 cls、mean、max。
            dropout: 训练时随机丢弃表示维度的概率，eval 时关闭。
            num_hidden_layers: 加载的 Transformer 层数；None 保留原模型配置。

        返回值：
            None；注册可训练的子模块。

        注意事项：
            num_hidden_layers 指定仅加载前 N 层，None 使用预训练配置；需要可读取的模型配置和权重。pool 只接受 cls/mean/max。
        """
        super().__init__()
        assert pool in ("cls", "mean", "max"), f"pool 须为 cls/mean/max，收到: {pool}"

        config = BertConfig.from_pretrained(bert_path)
        if num_hidden_layers is not None:
            config.num_hidden_layers = num_hidden_layers

        _prev = transformers.logging.get_verbosity()
        transformers.logging.set_verbosity_error()
        self.bert = BertModel.from_pretrained(bert_path, config=config)
        transformers.logging.set_verbosity(_prev)

        self.pool    = pool
        self.dropout = nn.Dropout(dropout)

    def encode(self, input_ids, attention_mask, token_type_ids):
        """独立编码一个 batch 的句子。

        用途：
            独立编码一个 batch 的句子。

        使用场景与调用位置：
            在 src/model.py 的“定义共享 BERT 双编码器和句对交互分类器”流程中使用。 所属作用域：BiEncoder。

        参数：
            input_ids: [B,L] token ID 整数张量，设备须与模型一致。
            attention_mask: [B,L] 有效 token 掩码，1 表示参与计算，0 表示 padding。
            token_type_ids: [B,L] 句段编号；单句通常全 0，句对区分 A/B。

        返回值：
            [B,H] 浮点句向量，与模型同设备，按最后一维 L2 归一化。

        注意事项：
            训练器、forward、离线编码和挖掘调用；不会自行切换 eval/no_grad，推理调用方负责关闭 dropout 和梯度。归一化向量点积等于余弦相似度。
        """
        out = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            return_dict=True,
        )
        vec = self._pool(out.last_hidden_state, attention_mask)  # [B, H]
        vec = self.dropout(vec)
        return F.normalize(vec, p=2, dim=-1)

    def forward(self, batch_a, batch_b):
        """用同一个编码器分别编码句对两侧。

        用途：
            用同一个编码器分别编码句对两侧。

        使用场景与调用位置：
            在 src/model.py 的“定义共享 BERT 双编码器和句对交互分类器”流程中使用。 所属作用域：BiEncoder。

        参数：
            batch_a: 句子 A 的编码字典，含 [B,L] input_ids/attention_mask/token_type_ids。
            batch_b: 句子 B 的编码字典，字段与 batch_a 相同。

        返回值：
            (emb_a, emb_b)，均为 [B,H] 张量。

        注意事项：
            通过 model(batch_a,batch_b) 由 PyTorch 调用；不计算相似度或 loss，保持编码与训练目标解耦。
        """
        emb_a = self.encode(**batch_a)
        emb_b = self.encode(**batch_b)
        return emb_a, emb_b

    def _pool(self, last_hidden, attention_mask):
        """把 token 隐状态压缩成一个句子向量。

        用途：
            把 token 隐状态压缩成一个句子向量。

        使用场景与调用位置：
            在 src/model.py 的“定义共享 BERT 双编码器和句对交互分类器”流程中使用。 所属作用域：BiEncoder。

        参数：
            last_hidden: [B,L,H] 的 BERT 最后一层 token 表示。
            attention_mask: [B,L] 有效 token 掩码，1 表示参与计算，0 表示 padding。

        返回值：
            [B,H] 未归一化句向量。

        注意事项：
            只在 encode 内调用；CLS 取第 0 token，mean 用 mask 加权平均并保护分母，max 用极小值屏蔽 padding。mean/max 未额外排除 CLS/SEP。
        """
        if self.pool == "cls":
            return last_hidden[:, 0, :]

        mask = attention_mask.unsqueeze(-1).float()  # [B, L, 1]

        if self.pool == "mean":
            sum_h = (last_hidden * mask).sum(dim=1)
            count = mask.sum(dim=1).clamp(min=1e-9)
            return sum_h / count

        if self.pool == "max":
            masked = last_hidden + (1 - mask) * (-1e9)
            return masked.max(dim=1).values


# ── CrossEncoder ──────────────────────────────────────────────────────────

class CrossEncoder(nn.Module):
    """建模两句 token 的联合交互并预测匹配标签。

    用途：
        建模两句 token 的联合交互并预测匹配标签。

    使用场景与调用位置：
        在 src/model.py 的“定义共享 BERT 双编码器和句对交互分类器”流程中使用。 本文件直接调用位置：build_crossencoder()。

    参数：
        构造参数见本类 __init__；实例成员与样本/模型结构见注意事项。

    返回值：
        torch.nn.Module；forward 输出 [B,2] logits。

    注意事项：
        由 build_crossencoder 创建；用于交叉熵训练、二分类评估和 Top-K 精排。候选无法独立预编码，查询与每个候选均需联合计算。
    """

    def __init__(self, bert_path, dropout=0.1, num_hidden_layers=None):
        """加载 BERT 预训练骨干并建立Dropout 与二分类线性层。

        用途：
            加载 BERT 预训练骨干并建立Dropout 与二分类线性层。

        使用场景与调用位置：
            调用 CrossEncoder(...) 创建实例时自动执行；在训练或评估加载模型/数据的准备阶段使用。

        参数：
            bert_path: BERT 本地预训练目录或库支持的模型标识。
            dropout: 训练时随机丢弃表示维度的概率，eval 时关闭。
            num_hidden_layers: 加载的 Transformer 层数；None 保留原模型配置。

        返回值：
            None；注册可训练的子模块。

        注意事项：
            num_hidden_layers 指定仅加载前 N 层，None 使用预训练配置；需要可读取的模型配置和权重。分类头由当前进程随机初始化，加载 checkpoint 后才恢复已训练权重。
        """
        super().__init__()

        config = BertConfig.from_pretrained(bert_path)
        if num_hidden_layers is not None:
            config.num_hidden_layers = num_hidden_layers

        _prev = transformers.logging.get_verbosity()
        transformers.logging.set_verbosity_error()
        self.bert = BertModel.from_pretrained(bert_path, config=config)
        transformers.logging.set_verbosity(_prev)

        hidden_size  = self.bert.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, 2)

    def forward(self, input_ids, attention_mask, token_type_ids):
        """对联合句对输入执行 BERT 和分类头。

        用途：
            对联合句对输入执行 BERT 和分类头。

        使用场景与调用位置：
            在 src/model.py 的“定义共享 BERT 双编码器和句对交互分类器”流程中使用。 所属作用域：CrossEncoder。

        参数：
            input_ids: [B,L] token ID 整数张量，设备须与模型一致。
            attention_mask: [B,L] 有效 token 掩码，1 表示参与计算，0 表示 padding。
            token_type_ids: [B,L] 句段编号；单句通常全 0，句对区分 A/B。

        返回值：
            [B,2] 未归一化 logits，第 1 列对应相似类。

        注意事项：
            通过 model(...) 在训练、评估与精排调用；不在内部做 softmax，因为 CrossEntropyLoss 接受原始 logits。推理排序时在外部取正类概率。
        """
        out = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            return_dict=True,
        )
        cls_vec = out.last_hidden_state[:, 0, :]  # [B, H]
        cls_vec = self.dropout(cls_vec)
        return self.classifier(cls_vec)            # [B, 2]


# ── 工厂函数 ──────────────────────────────────────────────────────────────

def build_biencoder(bert_path, pool="mean", dropout=0.1, num_hidden_layers=None):
    """创建 BiEncoder 并打印参数规模。

    用途：
        创建 BiEncoder 并打印参数规模。

    使用场景与调用位置：
        在 src/model.py 的“定义共享 BERT 双编码器和句对交互分类器”流程中使用。 跨文件调用：src/analyze_badcases.py:main()、src/compare_methods.py:load_and_eval()、src/evaluate.py:main()、src/evaluate_retrieval.py:main()、src/mine_hard_negatives.py:main()、src/train_biencoder.py:main()。

    参数：
        bert_path: BERT 本地预训练目录或库支持的模型标识。
        pool: 句向量池化方式，支持 cls、mean、max。
        dropout: 训练时随机丢弃表示维度的概率，eval 时关闭。
        num_hidden_layers: 加载的 Transformer 层数；None 保留原模型配置。

    返回值：
        BiEncoder 实例；初始设备由 PyTorch 默认设置决定。

    注意事项：
        训练、评估、挖掘和检索入口共用此工厂；加载预训练骨干不等于加载任务 checkpoint，调用方另行 load_state_dict 并 .to(device)。
    """
    model = BiEncoder(bert_path, pool=pool, dropout=dropout,
                      num_hidden_layers=num_hidden_layers)
    _print_param_info(model, f"BiEncoder (pool={pool}, layers={num_hidden_layers or 12})")
    return model


def build_crossencoder(bert_path, dropout=0.1, num_hidden_layers=None):
    """创建 CrossEncoder 并打印参数规模。

    用途：
        创建 CrossEncoder 并打印参数规模。

    使用场景与调用位置：
        在 src/model.py 的“定义共享 BERT 双编码器和句对交互分类器”流程中使用。 跨文件调用：src/analyze_badcases.py:main()、src/compare_methods.py:load_and_eval()、src/evaluate.py:main()、src/evaluate_retrieval.py:main()、src/train_crossencoder.py:main()。

    参数：
        bert_path: BERT 本地预训练目录或库支持的模型标识。
        dropout: 训练时随机丢弃表示维度的概率，eval 时关闭。
        num_hidden_layers: 加载的 Transformer 层数；None 保留原模型配置。

    返回值：
        CrossEncoder 实例；初始设备由 PyTorch 默认设置决定。

    注意事项：
        训练、评估、挖掘和检索入口共用此工厂；加载预训练骨干不等于加载任务 checkpoint，调用方另行 load_state_dict 并 .to(device)。
    """
    model = CrossEncoder(bert_path, dropout=dropout,
                         num_hidden_layers=num_hidden_layers)
    _print_param_info(model, f"CrossEncoder (layers={num_hidden_layers or 12})")
    return model


def _print_param_info(model, name):
    """区分模型总参数与 BERT 骨干参数的数量。

    用途：
        区分模型总参数与 BERT 骨干参数的数量。

    使用场景与调用位置：
        在 src/model.py 的“定义共享 BERT 双编码器和句对交互分类器”流程中使用。 本文件直接调用位置：build_biencoder()、build_crossencoder()。

    参数：
        model: 当前操作的模型实例；结构和设备须匹配输入。
        name: 当前数据划分、模型或输出结果的显示名称。

    返回值：
        None；按百万参数单位打印统计。

    注意事项：
        由两个 build 工厂调用；统计全部 parameters 的 numel，不等同于只统计 requires_grad=True。
    """
    total = sum(p.numel() for p in model.parameters()) / 1e6
    bert  = sum(p.numel() for p in model.bert.parameters()) / 1e6
    print(f"模型: {name}")
    print(f"参数量: {total:.1f}M  (BERT 骨干: {bert:.1f}M)")
