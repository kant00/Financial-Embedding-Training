# 文本匹配项目：从数据到实验报告

命令使用 Bash（Windows Git Bash、Linux 或 WSL），均从项目根目录执行。只想运行 CPU 基线时，从 [README](README.md#快速开始) 开始。

本文记录中文文本匹配的离线实验流程。公开汇总和评估范围见[实验结果](RESULTS.md)。

## 0. 先选路线

- **第一次学习**：第 1～4 节 → 第 5 节单模型训练和 calibration 错误分析。先理解一个 batch 和一个 checkpoint，不必一次跑完所有模型。
- **完整 BERT 实验**：第 1～5 节 → 可选第 6 节难负样本准备 → 第 7 节正式多模型评估 → 第 8 节召回精排 → 第 9 节报告与验收。
- **LLM 扩展**：完成主线后再读第 10 节；它不是运行 BERT 主线的前置条件。

主线：原始句对 → 固定数据划分 → 基线 → calibration 调试 → 冻结方案 → 多种子训练/最终评估 → Recall–Rerank → 报告。

本项目目前没有训练断点续跑、在线 API 服务、持久化向量索引、完整性能压测。遇到中断，不要假定脚本会自动跳过已完成实验。

## 1. 安装依赖和准备模型

先在项目根目录创建并激活 Python 3.10 环境。第 2 节的数据准备和第 4 节的 CPU 基线只需 `requirements-baseline.txt`。其余模型实验使用 `requirements.txt`，其中已包含基线依赖。

```bash
python --version
python -m pip install -r requirements.txt
```

已经安装依赖就跳过安装。使用已激活的 Python 环境；有报错先解决，不要继续下一步。

BERT 默认使用 Hugging Face 上的 `google-bert/bert-base-chinese`，首次加载需要网络或可用缓存。使用本地模型时，给对应脚本传入 `--bert_path`。本地目录需包含模型配置、分词器和权重。以下命令以 `models/bert-base-chinese` 为例。

```bash
python src/explore_data.py --data_dir data/experiments/afqmc_seed42 --bert_path models/bert-base-chinese
```

第 2 节生成数据后即可运行该示例。训练和评估使用同一份 BERT。`models/` 用于存放本地权重，由 `.gitignore` 排除。

需要确认是否可用 GPU 时运行：

```bash
python -c "import torch; print(torch.cuda.is_available())"
```

输出 False 时脚本使用 CPU，完整训练可能很慢。安装 requirements 不保证得到 GPU 版 PyTorch。

为固定 Python 哈希随机性，可在训练前执行一次：

```bash
export PYTHONHASHSEED=42
```

下文保留必要的数据和输出路径，防止误读旧数据或覆盖其他模型。重做实验时换一个输出目录，并同步后续命令中的 checkpoint/报告路径。长命令用 `\` 换行，反斜杠后不要加空格。

## 2. 准备原始数据与固定实验协议

### 2.1 原始数据

已有 `data/afqmc/train.jsonl` 和 `validation.jsonl` 就无需下载。每行格式：

```json
{"sentence1":"如何修改手机号","sentence2":"怎样更换绑定手机","label":1}
```

标签 1 表示匹配，0 表示不匹配。AFQMC **原始官方 test 的 -1 是无标签标记，不能用于计算准确率**。

主线只使用 AFQMC。下面的命令从 Hugging Face 获取 AFQMC，并覆盖对应目录中的同名文件。`--dataset lcqmc` 和 `--dataset bq_corpus` 分别获取另外两个数据集。省略 `--dataset` 会下载全部三个数据集。可用 `HF_ENDPOINT` 配置下载端点，使用数据前请查阅 [README 中的上游条款说明](README.md#数据和许可证)。

```bash
python src/download_data.py --dataset afqmc
```

### 2.2 生成协议（只在开始实验前运行一次）

```bash
python src/prepare_experiment_data.py
cat data/experiments/afqmc_seed42/split_metadata.json
```

| 文件 | 用途 | 本地原始 AFQMC 对应条数 |
|---|---|---:|
| train.jsonl | 参数训练、挖掘难负样本 | 30,901 |
| calibration.jsonl / validation.jsonl | 同一份数据的两个文件名；选模型、阈值、超参数 | 3,433 |
| final_test.jsonl / test.jsonl | 同一份数据的两个文件名；冻结方案后最终评估 | 4,316 |
| evaluation_manifest.jsonl | final test 加 sample_id，供检索和 SFT 使用 | 4,316 |
| split_metadata.json | 源文件哈希、标签统计、句子重叠审计 | — |

条数以实际元数据为准。`final_test` 来自 **AFQMC 原始验证集**，这批句对曾用于早期选模。要验证泛化，需要另取未参与选模的留出集。官方测试集的标签为 `-1`。

默认按标签分层划分，可能有共享句子。若需要按共享句子隔离 train/calibration，可在开始实验前改用严格划分：

```bash
python src/prepare_experiment_data.py --strict_group_split --output_dir data/experiments/afqmc_group_seed42
```

选择此分支后，把后续命令中的 `data/experiments/afqmc_seed42` 全部换成 `data/experiments/afqmc_group_seed42`，并使用独立输出目录，不能混合两种协议的结果。

这是**替代分支**，不必两种都运行。strict 只保证 train/calibration 隔离，**不保证 train/final test 无共享句子**，仍需阅读泄漏审计。

## 3. 理解数据（不训练）

```bash
python src/explore_data.py --data_dir data/experiments/afqmc_seed42 --output_dir outputs/eda
```

检查类别比例、字符长度、token 长度和句长差分布。token 图只抽取部分样本，不能直接当作全量截断率。没有分词器时可加 `--skip_token`，但正式训练仍需要完整 BERT。

主线双塔使用单句 `max_length=64`，交叉塔使用句对 `max_length=128`。如要改长度，依据训练/calibration 数据决定，不要根据 final test 错误反复调参。当前独立交叉塔评估和错误分析固定长度 128，本指南保持训练一致；自定义长度需同步修改实现，不能只改训练命令。

## 4. 跑无需神经训练的基线

```bash
python src/run_baselines.py \
  --data_dir data/experiments/afqmc_seed42 \
  --output_dir outputs/credible_baselines \
  --seed 42
```

输出多数类和字符 TF-IDF 的 metrics、predictions，以及 `baseline_summary.json`。词表只在 train 建立，阈值只在 calibration 选择；脚本会自动评估 final test。这个分支是预先固定的参考基线，不要据其 final-test 分数反复修改方案。

读结果时区分 `f1_pos`、`f1_macro`、`f1_weighted`。兼容字段 `f1` 指 weighted-F1，不是正例 F1。`pr_auc` 当前实现为 Average Precision（AP）。类别不平衡时，不能只用 Accuracy 证明模型有效。

## 5. 学习调试：先跑通一个双塔

### 5.1 Cosine 训练

```bash
python src/train_biencoder.py \
  --data_dir data/experiments/afqmc_seed42 \
  --output_dir outputs/dev/cosine_seed42 \
  --loss cosine
```

初次只验证流程可改 `--epochs 1 --batch_size 8 --num_hidden_layers 1`，仍然读取全量数据，不是快速小样本开关；其结果不能冒充正式 4 层实验。显存不足优先降 batch；`--grad_accum` 可累积梯度，但对 InfoNCE **不会增加 batch 内负样本数**。当前尾部不足一个累积组仍按完整累积步数缩放，精确对照实验需注意。

默认训练 4 层 BERT、3 轮、batch_size=32。训练每轮读取 calibration，按正例 F1 保存最佳权重和阈值。先确认 `outputs/dev/cosine_seed42/checkpoints/biencoder_cosine_best.pt` 已生成，再进入评估。日志位于 `logs/biencoder_cosine_log.json`，是包含配置、环境和逐轮记录的对象，不是旧版列表。

### 5.2 calibration 评估与错误分析

下面的 `outputs/dev/cosine_seed42/...` 路径要求先完成第 5.1 节的训练。如果使用默认参数训练，checkpoint 位于 `outputs/checkpoints/`；对默认原始 AFQMC 验证集评估可在项目根目录运行：

```bash
python src/evaluate.py --model_type biencoder --ckpt outputs/checkpoints/biencoder_cosine_best.pt
```

该命令使用原始 validation，不等同于新协议 calibration/final test；评估数据须与训练时的数据协议一致。checkpoint 路径不存在时，脚本会提示找到的同名文件，需核对后选择。

Windows 启动故障：当前 `nlp` Conda 环境曾在先导入 matplotlib/PyTorch、后导入 scikit-learn 时出现 `Windows fatal exception: stack overflow`。`evaluate.py` 已调整为先加载 scikit-learn，再加载 PyTorch 和绘图库；不要恢复旧导入顺序。启动提示会显示依赖加载阶段，验证应同时覆盖激活 Conda 后的终端环境。

```bash
python src/evaluate.py \
  --model_type biencoder \
  --ckpt outputs/dev/cosine_seed42/checkpoints/biencoder_cosine_best.pt \
  --data_dir data/experiments/afqmc_seed42 \
  --split validation \
  --max_length 64 \
  --output_dir outputs/dev/cosine_seed42/calibration
python src/analyze_badcases.py \
  --model_type biencoder \
  --ckpt outputs/dev/cosine_seed42/checkpoints/biencoder_cosine_best.pt \
  --data_dir data/experiments/afqmc_seed42 \
  --split validation \
  --output_dir outputs/dev/cosine_seed42/badcases \
  --n_cases 10
```

关注 FP（错误判相似）、FN（漏判相似）、否定词/实体/数字变化和阈值附近案例。错误分析的优化建议是启发式提示，不是已验证结论；CosineEmbeddingLoss **增大 margin 会放宽负例约束**。

评估默认读取 checkpoint 的固定阈值。只有 calibration 可以加 `--tune_threshold`，该操作不会自动重写 checkpoint；若采用新阈值，必须记录并在后续显式使用同一 `--threshold`。final test 禁止调阈值。阈值搜索基于分数边界和中点，默认优化正例 F1，不是固定 [0,1] 的 101 点网格。

`evaluate.py` 的 JSON/JSONL 按指定 output_dir 保存；双塔分布图目前仍写入全局 `outputs/figures`，同 split 会覆盖，需图像时及时另存。

### 5.3 换训练目标时怎么读代码

| 方法 | 训练数据如何使用 | 入口与检查点文件名 |
|---|---|---|
| Cosine | 原始 0/1 句对 | train_biencoder.py --loss cosine；biencoder_cosine_best.pt |
| 随机 Triplet | 正例为 anchor/positive，数据集初始化时采样负例，epoch 间不刷新 | train_biencoder.py --loss triplet；biencoder_triplet_best.pt |
| in-batch InfoNCE | 仅正例句对；同 batch 其他样本作候选负例 | train_biencoder.py --loss inbatch；biencoder_inbatch_best.pt |
| Hard Triplet | 第 6 节生成 anchor/positive/negative JSONL | 同 triplet，额外传 --hard_negative_file |
| Cross-Encoder | 两句联合编码，分类损失；默认 balanced 类别权重 | train_crossencoder.py；crossencoder_best.pt |

InfoNCE 不要求重新人工标注一种数据格式，但要求构造正例 batch，且存在 false negative 风险；batch_size 应大于 1。随机 Triplet 的负例不等于人工确认不匹配。看代码可按 `dataset.py → model.py → train_biencoder.py → metrics.py → evaluate.py`，跟踪一个 batch 的字段和张量形状。

### 5.4 用训练后的模型判断自己输入的句对

这一节是可选的代码示例。完成第 5.1 节后，将下面代码保存为项目根目录的 `predict_demo.py`，修改两句文本后运行。它不修改权重，也不是线上服务。

```python
import sys
import torch
from transformers import BertTokenizer
sys.path.insert(0, "src")  # 从项目根目录导入现有模型实现。
from model import build_biencoder

checkpoint_path = "outputs/dev/cosine_seed42/checkpoints/biencoder_cosine_best.pt"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 只加载自己训练或可信来源的 checkpoint；不要加载来源不明的 pickle 权重。
checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
config = checkpoint["args"]
bert_path = config["bert_path"]  # 若移动了本地模型，请换成它的新路径。
tokenizer = BertTokenizer.from_pretrained(bert_path)
model = build_biencoder(bert_path, pool=config.get("pool", "mean"), num_hidden_layers=config.get("num_hidden_layers"))
model.load_state_dict(checkpoint["state_dict"])
model.to(device).eval()  # 关闭 dropout；仅 no_grad 并不会自动切换到评估模式。
inputs = tokenizer(["如何修改绑定手机号", "怎样更换绑定手机"], padding=True, truncation=True, max_length=config.get("max_length", 64), return_tensors="pt")
inputs = {name: value.to(device) for name, value in inputs.items()}
with torch.no_grad():
    vectors = model.encode(**inputs)  # encode 已做 L2 归一化，点积等于余弦相似度。
    score = (vectors[0] * vectors[1]).sum().item()
threshold = float(checkpoint["threshold"])  # 复用 calibration 阈值，不根据演示句对调阈值。
print({"score": score, "threshold": threshold, "prediction": int(score >= threshold)})
```

```bash
python predict_demo.py
```

prediction=1 表示匹配，0 表示不匹配。余弦分数不是概率，演示句对也不保证一定预测正确。

## 6. 扩展：挖掘 Hard Negative 并检查质量

此步骤使用第 5 节已训练的双塔，仅从 **train** 挖掘；不得混入 calibration/final test。只做基础主线可跳过整节。

```bash
python src/mine_hard_negatives.py \
  --data_file data/experiments/afqmc_seed42/train.jsonl \
  --checkpoint outputs/dev/cosine_seed42/checkpoints/biencoder_cosine_best.pt \
  --output_file outputs/hard_negatives/triplets.jsonl \
  --negatives_per_positive 1 \
  --retrieval_depth 100 \
  --batch_size 64 \
  --max_length 64 \
  --backend auto \
  --seed 42
head -n 10 outputs/hard_negatives/triplets.jsonl
```

默认只保留检索到的、与 anchor 存在显式 label=0 关系的负例，因此覆盖率可能很低，甚至为零。打开 `outputs/hard_negatives/triplets.metadata.json`，检查 `n_triplets`、`coverage`、`source_counts`；`n_triplets` 必须大于 0 才能训练，并应随机人工抽查，不能只看前十条。

若为空，可在 train 上增大 retrieval_depth 后重试；不要直接把未标注候选当真负例。`--allow_unlabeled_candidates` 是带误负例风险的实验开关，启用前应确定人工复核方案。多负例时 coverage 是三元组数/正例数，可能大于 1，不是去重 anchor 覆盖率。

`auto` 优先尝试 FAISS，不可用时回退 NumPy。FAISS 是可选依赖，不包含在 requirements；需在所选环境中安装兼容版本。NumPy 检索会构造完整 query×corpus 分数矩阵：全量挖掘可能需要数 GB 甚至更多内存，降低编码 batch_size **不能**消除该矩阵开销。内存不足应停止，准备可用 FAISS/足够内存后重试；当前未实现分块检索。

如需在正式评估前调试 Hard Triplet，执行：

```bash
python src/train_biencoder.py \
  --data_dir data/experiments/afqmc_seed42 \
  --output_dir outputs/dev/hard_triplet_seed42 \
  --loss triplet \
  --hard_negative_file outputs/hard_negatives/triplets.jsonl
python src/evaluate.py \
  --model_type biencoder \
  --ckpt outputs/dev/hard_triplet_seed42/checkpoints/biencoder_triplet_best.pt \
  --data_dir data/experiments/afqmc_seed42 \
  --split validation \
  --output_dir outputs/dev/hard_triplet_seed42/calibration
```

Hard Triplet 训练从预训练 BERT 重新初始化，并非接着挖掘用 checkpoint 续训。与随机 Triplet 的区别还包括训练样本数量/分布，应在报告中记录，不能把分数差全部归因于“负例更难”。

## 7. 冻结配置后：多模型、多种子最终评估

进入本节前，确定数据协议、模型集合、层数、epoch、batch 等参数；此后不要根据 final-test 排名回头挑超参数。自动套件会在每个训练结束后立即评估 test。

### 7.1 先预演，再正式运行

```bash
python src/run_experiment_suite.py \
  --data_dir data/experiments/afqmc_seed42 \
  --output_dir outputs/experiment_suite \
  --models cosine,triplet,inbatch,crossencoder \
  --seeds 13,42,2026 \
  --dry_run
```

检查打印出的 12 次训练、12 次评估的路径。`--dry_run` 不训练，也不会产生指标。默认每组训练 4 层 BERT、3 轮、batch_size=32。确认资源充足后：

```bash
python src/run_experiment_suite.py \
  --data_dir data/experiments/afqmc_seed42 \
  --output_dir outputs/experiment_suite \
  --models cosine,triplet,inbatch,crossencoder \
  --seeds 13,42,2026
```

学习演示可将两个命令都改为 `--seeds 42 --epochs 1`，但须明确“单种子/单轮”，不能当作三种子结果。套件不会复用第 5 节开发 checkpoint，而是重新训练。

本套件只暴露命令中这些训练选项和 hard_negative_file，不支持随意附加 `--pool`、`--temperature`、`--max_length`、`--grad_accum`；内部使用各训练器默认值。如开发阶段调整了这些参数，必须先同步套件实现或用独立脚本执行相同配置，不能假定套件沿用开发配置。

每种方法的产物位于 `outputs/experiment_suite/方法/seed_种子/`：

- `checkpoints/`：最佳模型；`logs/`：训练配置、环境、逐轮指标。
- `evaluation/biencoder_test_metrics.json` 或 `crossencoder_test_metrics.json`，以及对应 predictions.jsonl。
- 全部完成后根目录生成 `multi_seed_summary.json`，包含单次结果和 mean/std。

中途失败时已完成文件通常保留，但总汇总可能尚未生成。套件没有自动断点恢复/跳过功能；重跑同目录会覆盖同名产物。不要在相同目录分别跑多个模型子集后，以为最后的 summary 会自动合并。

### 7.2 可选：Hard Triplet 三种子对照

先完成第 6 节并冻结 `outputs/hard_negatives/triplets.jsonl`。使用独立目录，避免覆盖随机 Triplet：

```bash
python src/run_experiment_suite.py \
  --data_dir data/experiments/afqmc_seed42 \
  --output_dir outputs/hard_suite \
  --models triplet \
  --seeds 13,42,2026 \
  --hard_negative_file outputs/hard_negatives/triplets.jsonl
```

这里三个种子共用一份固定教师挖掘的数据；统计的是训练随机性的变化，不是完整“挖掘+训练”随机性的变化。套件内方法名仍是 triplet，应通过独立目录和实验说明区分 Hard Triplet。

### 7.3 独立模型如何评估 final test

仅当没有使用套件、需要评估自己单独训练的冻结模型时使用；套件已评估的不必重复运行：

```bash
python src/evaluate.py \
  --model_type biencoder \
  --ckpt outputs/dev/cosine_seed42/checkpoints/biencoder_cosine_best.pt \
  --data_dir data/experiments/afqmc_seed42 \
  --split test \
  --max_length 64 \
  --output_dir outputs/dev/cosine_seed42/final_test
```

交叉塔改为 `--model_type crossencoder` 并传对应 checkpoint。每个模型使用独立 output_dir，避免同名 metrics 被覆盖。独立评估结果不会自动进入第 9 节套件报告。

## 8. 两阶段检索：Bi-Encoder 召回 + Cross-Encoder 精排

下面预先选择 cosine seed42 和 crossencoder seed42，**不是根据 final-test 排名选出的最优种子**。依赖第 7.1 节已完成的两个 checkpoint；缺少文件时先完成训练：

```bash
python src/evaluate_retrieval.py \
  --manifest data/experiments/afqmc_seed42/evaluation_manifest.jsonl \
  --biencoder_checkpoint outputs/experiment_suite/cosine/seed_42/checkpoints/biencoder_cosine_best.pt \
  --crossencoder_checkpoint outputs/experiment_suite/crossencoder/seed_42/checkpoints/crossencoder_best.pt \
  --output_dir outputs/retrieval \
  --recall_top_k 50 \
  --rerank_top_k 20 \
  --cutoffs 1,5,10,50 \
  --batch_size 64 \
  --max_length 64 \
  --cross_max_length 128 \
  --backend auto
```

只测召回可省略 crossencoder_checkpoint。rerank_top_k 必须不大于 recall_top_k；本例重排前 20，后 30 保持召回次序，不会凭空找回召回阶段遗漏的文档。

产物：`retrieval_metrics.json`（recall/rerank 指标、规模、后端、阶段耗时）和 `retrieval_rankings.jsonl`（逐查询排名）。读取 Recall@K、MRR、NDCG@K，同时注明 MRR 基于截断排名。耗时是本次离线阶段统计，不是线上 QPS、P95/P99。

候选库由 manifest 的 sentence2 去重构成，查询/相关集合来自已标注正例。这是从句对数据派生的检索实验，相关性标注并不完备，不能冒充完整人工标注检索基准或线上效果。脚本不保存可部署的向量索引。NumPy 内存限制同第 6 节。

## 9. 生成报告与验收

```bash
python src/build_experiment_report.py \
  --baseline_dir outputs/credible_baselines \
  --suite_dir outputs/experiment_suite \
  --retrieval_dir outputs/retrieval \
  --output outputs/EXPERIMENT_REPORT.md
cat outputs/EXPERIMENT_REPORT.md
python -m pip freeze > outputs/environment-freeze.txt
cp data/experiments/afqmc_seed42/split_metadata.json outputs/split_metadata.json
```

报告只读取指定目录里的真实 JSON。某一步没跑，它会缺行或显示未运行；**生成 Markdown 成功不等于完成实验**。表格展示多种子均值，标准差仍需阅读 `multi_seed_summary.json`，不要遗漏波动性。

若完成 Hard Triplet，用同一报告脚本、`--suite_dir outputs/hard_suite`、`--output outputs/HARD_TRIPLET_REPORT.md` 再生成单独报告；当前不会自动将两个套件合并，标题/说明须注明 Hard Triplet，不能把两行 triplet 混淆。SFT/API 结果也不会自动合并。

完成检查：

- [ ] 保存数据协议/源哈希，确认所有命令使用同一数据目录。
- [ ] 基线结果存在；每个预定方法/种子都存在 checkpoint、训练日志、最终指标及预测明细。
- [ ] `multi_seed_summary.json` 中种子数符合预定方案；没有把单轮/单种子 smoke run 写成正式实验。
- [ ] 难负样本分支（如启用）有 metadata、抽检说明、独立套件结果。
- [ ] 检索结果同时包含召回与精排指标，记录后端、候选规模与硬件。
- [ ] 最终报告不是空表；报告数字来自本轮输出，未拼接旧协议结果。
- [ ] 保存实际命令、环境和实验局限，报告中的数字应能追溯到本轮产物。

BERT 预测明细当前生成 `test-00000` 一类 ID，manifest 使用 `afqmc-final-00000`；SFT 明细也未保留统一 ID。虽然按本文全量流程样本来源一致，逐样本对比仍须校验顺序及 sentence1/sentence2/label，不能直接假设 sample_id 可联表。

## 10. 可选 LLM 分支（不阻塞 BERT 主线）

### 10.1 本地 Qwen LoRA SFT

先按上游条款下载完整的指令模型，例如 [Qwen2.5-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct)，将权重、分词器和聊天模板保存到 `models/qwen-0.5b` 或你指定的本地目录。`--model_path` 接收本地目录。训练和 LoRA 评估须使用相同的基座版本。

```bash
python src_llm/train_sft.py \
  --model_path models/qwen-0.5b \
  --data_dir data/experiments/afqmc_seed42 \
  --output_dir outputs/sft \
  --num_train 5000 \
  --epochs 3 \
  --batch_size 4 \
  --grad_accum 4 \
  --max_length 256 \
  --seed 42
```

LoRA 输出 `outputs/sft/sft_adapter`，日志在 `outputs/sft/logs/train_sft.json`。`--num_train -1` 仍会做类别平衡下采样，不代表使用全部 train。验证使用 calibration 的前 500 条，按验证 loss 选最佳模型。

当前训练为 float32，不能按“0.5B+LoRA”简单保证某显存一定够。`max_length=256` 是本示例配置，不保证所有样本答案都保留；实现先拼接后截断，需检查回答 token 没被截掉。`--full_ft` 是额外的全参数微调分支，输出 `sft_full_ckpt`，资源需求更高，不必为跑通主线执行。

正式评估仅在方案冻结后运行。先确认 `data/experiments/afqmc_seed42/evaluation_manifest.jsonl` 和 `outputs/sft/sft_adapter` 存在；manifest 缺失会让脚本回退到其他数据：

```bash
python src_llm/evaluate_sft.py \
  --model_path models/qwen-0.5b \
  --ckpt_dir outputs/sft/sft_adapter \
  --data_dir data/experiments/afqmc_seed42 \
  --manifest data/experiments/afqmc_seed42/evaluation_manifest.jsonl \
  --num_samples -1 \
  --seed 42
cp outputs/logs/sft_results.json outputs/sft/sft_results.json
cp outputs/logs/sft_predictions.jsonl outputs/sft/sft_predictions.jsonl
```

评估器没有 output_dir 参数，固定写全局 `outputs/logs`，所以每次运行后立即复制归档。`--demo` 只抽 5 条；`--num_samples 200` 也是子集，均不能与全量 BERT 指标直接比较。

主指标将解析失败按错误计入，`valid_only` 仅供诊断，必须同时报告 parse_fail_rate。用真实标签反转处理失败只是评估计分规则，不是可部署的预测回退策略。控制台耗时文字含“GPU”，实际设备仍以运行环境为准。

### 10.2 在线 API 试验

这是独立探索分支，会发送句对文本给外部服务并产生费用；仅使用允许外发的数据。先在自己的服务账户确认可用模型、权限和计费。不在本文保存 API Key：

密钥仅在本机终端设置，不要写入项目文件、截图或提交到仓库；下面是占位符：

```bash
export DASHSCOPE_API_KEY="你的 API Key"
python src_llm/llm_compare.py --data_dir data/experiments/afqmc_seed42 --split validation --num_samples 100
unset DASHSCOPE_API_KEY
```

直接输入密钥可能留在 shell 历史中；也可以在运行环境中预先配置环境变量，只执行中间的 Python 命令。

输出固定到 `outputs/logs/llm_compare_results.json`，同名覆盖。此脚本按类别抽样且采用 valid-only 统计，不读取统一 manifest；仅作为 calibration 上的提示词/格式探索，**不能直接纳入同测试集正式排行榜**。100 条也不保证有统计代表性。

## 11. 常见卡点与恢复

| 现象 | 检查/处理 |
|---|---|
| 找不到 Python 或装完包仍报缺包 | 先激活正确 Python 环境；用 `python -m pip` 安装 |
| `OMP: Error #15` | 当前环境可能混装 OpenMP；优先修复/隔离依赖。部分脚本已有临时绕过，不等于环境健康 |
| CUDA=False | 当前 PyTorch/驱动不支持 GPU 或无可用设备；先修复环境，不是调 batch 就能启用 CUDA |
| 模型加载失败/权重尺寸不匹配 | 核对完整权重、tokenizer、层数和训练使用的 bert_path；不要强行忽略加载错误 |
| 没有最佳 checkpoint | 检查训练是否退出、验证 F1 是否一直为 0；当前保存条件可能在这种情况下不写最佳文件 |
| 原始 test 全是 -1 | 指错 data_dir；正式流程必须指向生成的协议目录 |
| 验证阶段看似停住 | 阈值搜索逐候选计算指标，本身可能较慢；检查进程活动与日志，不要立即重复启动 |
| 挖掘为空/内存不足 | 先检查 metadata、检索深度及 NumPy 全矩阵开销，参见第 6 节 |
| 最终报告缺少模型 | 查看对应 suite summary 是否生成；独立 evaluate 目录不会被报告自动扫描 |
| 重跑后旧结果不见 | 同名文件会覆盖；每轮换输出目录，并保留旧实验目录 |

遇到 OMP 冲突，只有理解这是临时绕过并接受风险时，才在当前终端设置 `export KMP_DUPLICATE_LIB_OK=TRUE` 重试；它不是正确性保证。不要把这种环境的结果无条件称为严格可复现。

本地已经具备原始 AFQMC 验证集及四个训练 checkpoint 时，可运行基础对比：

```bash
python src/compare_methods.py --split validation --output_dir outputs/comparison
```

脚本统一评估 `outputs/checkpoints` 中的 Cosine、Triplet、InBatch 和 CrossEncoder，输出 `outputs/comparison/validation/COMPARISON_REPORT.md`、汇总 JSON/CSV、指标柱状图、混淆矩阵及有匹配日志的验证 F1 曲线。每个模型有独立的 `metrics.json`、`predictions.jsonl` 和双塔相似度图，不覆盖独立 evaluate 的图表。按 checkpoint 恢复层数、池化、序列长度和双塔阈值，不重新训练或调阈值。

支持 `--data_dir`、`--ckpt_dir`、`--bert_path`、`--log_dir` 和 `--batch_size`。默认计算 1000 次 bootstrap 区间，`--bootstrap_samples 0` 可跳过。缺失模型默认报错；确需部分对比时加 `--allow_missing`，报告会注明缺失方法。官方 test 的 -1 标签会在加载模型前被拒绝。

这份报告每个方法仅一个 checkpoint，默认使用曾用于选模的 AFQMC 原始验证集。多种子实验使用第 7 节的套件和 `build_experiment_report.py`。公开汇总与协议差异见[实验结果](RESULTS.md)。

