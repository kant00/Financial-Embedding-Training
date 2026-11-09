# 中文文本匹配实验与复现

给定两句中文，判断它们是否表达相同的意图。项目从多数类和字符 TF‑IDF 基线入手，比较 BERT 双塔的 Cosine、Triplet、In-batch 对比学习与 Cross-Encoder，并提供难负样本挖掘、双塔召回与交叉编码器精排的实验脚本。LoRA 指令微调和在线 API 对比是可选分支。

## 快速开始

使用 Python 3.10。从项目根目录运行 `python -m venv .venv`，然后激活环境。Windows PowerShell 使用 `.\.venv\Scripts\Activate.ps1`，Windows Git Bash 使用 `source .venv/Scripts/activate`，Linux 或 macOS 使用 `source .venv/bin/activate`。

下载 AFQMC 需要网络，后续步骤只用 CPU，无需安装 PyTorch 或下载 BERT 权重。

```bash
python -m pip install -r requirements-baseline.txt
python src/download_data.py --dataset afqmc
python src/prepare_experiment_data.py
python src/run_baselines.py
python src/build_experiment_report.py
```

打开 `outputs/EXPERIMENT_REPORT.md` 查看指标。数据划分、源文件哈希和句子重叠审计记录在 `data/experiments/afqmc_seed42/split_metadata.json`。如果已经按上游条款取得 AFQMC，可将统一为 `sentence1`、`sentence2`、`label` 三个字段的 `train.jsonl` 和 `validation.jsonl` 放入 `data/afqmc/`，再从划分步骤开始。

## 实验路线

| 想做的实验 | 入口 |
| --- | --- |
| 训练 BERT 双塔或交叉编码器 | 安装 `requirements.txt`，从[使用指南第 5 节](USAGE_GUIDE.md#5-学习调试先跑通一个双塔)开始 |
| 冻结配置后跑多种子分类实验 | [使用指南第 7 节](USAGE_GUIDE.md#7-冻结配置后多模型多种子最终评估) |
| 评估召回与精排 | [使用指南第 8 节](USAGE_GUIDE.md#8-两阶段检索bi-encoder-召回--cross-encoder-精排) |
| 用 LoRA 或在线 API 探索生成式匹配 | [使用指南第 10 节](USAGE_GUIDE.md#10-可选-llm-分支不阻塞-bert-主线) |

BERT 默认使用 [`google-bert/bert-base-chinese`](https://huggingface.co/google-bert/bert-base-chinese)，可用 `--bert_path` 指定本地模型目录。LoRA 分支的 `--model_path` 接收本地指令模型目录。完整命令和参数见[使用指南](USAGE_GUIDE.md)。

## 已有结果

字符 TF‑IDF 基线的正例 F1 为 **0.4778**。历史四模型比较中，Cross-Encoder 的正例 F1 为 **0.6220**，来自 AFQMC 原始验证集的一次训练。固定协议也将这份验证集用作 `final_test`。结果、数据划分及局限见[实验结果](RESULTS.md)。

## 数据和许可证

[AFQMC（CLUE）](https://huggingface.co/datasets/clue/clue/tree/main/afqmc)是本文快速开始使用的数据集。[LCQMC](https://huggingface.co/datasets/C-MTEB/LCQMC)和 [BQ Corpus](https://huggingface.co/datasets/FinanceMTEB/bq_corpus)可通过下载脚本单独获取。获取和使用第三方数据或模型时，请遵循相应上游条款并引用原始来源。

本仓库源码和随附文档采用 [MIT 许可证](LICENSE)。第三方数据、预训练模型和外部服务适用各自条款。数据、模型权重、逐样本预测及运行日志在本地生成或取得。
