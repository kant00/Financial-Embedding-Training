"""
在线 LLM 句对分类探索（DashScope 兼容接口，非正式统一评测入口）。

用途与流程位置：
    可选的提示词/回答格式试验；调用外部服务，不参与 BERT 训练。
    提示模型输出“是”或“否”，解析结果后计算分类指标。

输入与输出：
    通过 DASHSCOPE_API_KEY 环境变量读取密钥；安全设置和清理见 USAGE_GUIDE.md 第 10.2 节。
    --data_dir 与 --split 决定来源，--num_samples 控制按类别抽取的样本量。
    不读取统一 manifest；异常/无法解析的结果不进入 valid-only 主统计。
    结果固定写 outputs/logs/llm_compare_results.json，同名覆盖。

使用方式（在项目根目录执行，Bash）：
    python src_llm/llm_compare.py --data_dir data/experiments/afqmc_seed42 --split validation --num_samples 100 --model qwen-plus

注意事项：
    调用会外发句对并产生费用，先确认数据许可、账户权限、模型可用性与计费。
    100 条不保证有代表性；valid-only 可能乐观，不能直接与全量 BERT/SFT 指标比较。
    默认 data_dir 仍是旧目录，主线探索应显式传入 calibration 的 validation 别名。
    不保证 LLM 精度、泛化、成本或速度优于其他方法。

说明核对：2026-09-28；完整流程、前置条件及自定义模型路径见 USAGE_GUIDE.md。
"""

import argparse
import json
import os
import random
import time
from pathlib import Path

random.seed(42)

ROOT     = Path(__file__).parent.parent
DATA_DIR = ROOT / "data" / "afqmc"

DASHSCOPE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"

PROMPT_TEMPLATE = """请判断以下两个问题是否表达相同的意思。只回答"是"或"否"，不要有任何其他内容。

问题1：{s1}
问题2：{s2}

回答："""


# ── 单次 LLM 调用 ─────────────────────────────────────────────────────────

def call_llm(client, s1, s2, model):
    """通过兼容 OpenAI 的客户端发送一对句子的零样本请求。

    用途：
        通过兼容 OpenAI 的客户端发送一对句子的零样本请求。

    使用场景与调用位置：
        在 src_llm/llm_compare.py 的“调用远程语言模型 API 进行零样本文本匹配实验”流程中使用。 本文件直接调用位置：evaluate_llm()。

    参数：
        client: 已经配置凭据和服务地址的 API 客户端。
        s1: 句子 A 的文本。
        s2: 句子 B 的文本。
        model: 远程服务接受的模型名称字符串，例如 qwen-plus。

    返回值：
        (pred, raw)：pred 为 0/1/-1；raw 为答案或异常文本。

    注意事项：
        evaluate_llm 对每条样本调用；temperature=0、max_tokens=5，会产生远程 API 请求及可能的费用。当前先检查是再检查否，包含两者时可能误判；异常统一返回 -1。
    """
    prompt = PROMPT_TEMPLATE.format(s1=s1, s2=s2)
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=5,
            temperature=0.0,
        )
        answer = resp.choices[0].message.content.strip()
        # 解析：含"是"→1，含"否"→0，其他→-1（解析失败）
        if "是" in answer:
            return 1, answer
        elif "否" in answer:
            return 0, answer
        else:
            return -1, answer  # 解析失败
    except Exception as e:
        return -1, str(e)


# ── 批量评估 ─────────────────────────────────────────────────────────────

def evaluate_llm(samples, client, model, sleep_sec=0.2):
    """串行调用 API 收集预测并统计解析/调用失败。

    用途：
        串行调用 API 收集预测并统计解析/调用失败。

    使用场景与调用位置：
        在 src_llm/llm_compare.py 的“调用远程语言模型 API 进行零样本文本匹配实验”流程中使用。 本文件直接调用位置：main()。

    参数：
        samples: 本次 API 评估使用的已标注句对列表。
        client: 已经配置凭据和服务地址的 API 客户端。
        model: 远程服务接受的模型名称字符串，例如 qwen-plus。
        sleep_sec: 相邻 API 请求后的等待秒数，应非负。

    返回值：
        (results, parse_fail)，每条含 sentence1/sentence2/label/pred/raw。

    注意事项：
        main 采样后调用；每 10 条打印有效预测 accuracy，每次调用后 sleep_sec 等待。无显式自定义重试，SDK 行为由客户端决定。
    """
    results = []
    parse_fail = 0

    for i, r in enumerate(samples):
        pred, raw = call_llm(client, r["sentence1"], r["sentence2"], model)
        if pred == -1:
            parse_fail += 1
        results.append({
            "sentence1": r["sentence1"],
            "sentence2": r["sentence2"],
            "label":     r["label"],
            "pred":      pred,
            "raw":       raw,
        })
        if (i + 1) % 10 == 0:
            done = [x for x in results if x["pred"] != -1]
            if done:
                acc_so_far = sum(1 for x in done if x["pred"] == x["label"]) / len(done)
                print(f"  [{i+1}/{len(samples)}] 当前准确率（有效预测）: {acc_so_far:.3f}  "
                      f"解析失败: {parse_fail}")
        time.sleep(sleep_sec)

    return results, parse_fail


# ── 统计指标 ──────────────────────────────────────────────────────────────

def compute_metrics(results):
    """为旧 API 对比入口计算有效预测上的二分类指标。

    用途：
        为旧 API 对比入口计算有效预测上的二分类指标。

    使用场景与调用位置：
        在 src_llm/llm_compare.py 的“调用远程语言模型 API 进行零样本文本匹配实验”流程中使用。 本文件直接调用位置：main()。

    参数：
        results: 包含样本标签、预测、分数或方法元数据的结果列表。

    返回值：
        accuracy、precision_pos、recall_pos、f1_pos、n_valid、n_fail；全部失败时返回简化字段。

    注意事项：
        main 在 API 批量完成后调用；当前过滤 pred=-1，得到的是 valid-only 结果，与 evaluate_sft 的失败按错误计入主指标不同；使用时必须说明覆盖率。
    """
    valid = [r for r in results if r["pred"] != -1]
    if not valid:
        return {"accuracy": 0.0, "f1_pos": 0.0, "n_valid": 0, "n_fail": len(results)}

    labels = [r["label"] for r in valid]
    preds  = [r["pred"]  for r in valid]

    tp = sum(1 for l, p in zip(labels, preds) if l == 1 and p == 1)
    fp = sum(1 for l, p in zip(labels, preds) if l == 0 and p == 1)
    fn = sum(1 for l, p in zip(labels, preds) if l == 1 and p == 0)
    acc = sum(1 for l, p in zip(labels, preds) if l == p) / len(valid)
    prec = tp / max(tp + fp, 1)
    rec  = tp / max(tp + fn, 1)
    f1   = 2 * prec * rec / max(prec + rec, 1e-9)

    return {
        "accuracy": acc,
        "precision_pos": prec,
        "recall_pos":    rec,
        "f1_pos":        f1,
        "n_valid":       len(valid),
        "n_fail":        len(results) - len(valid),
    }


# ── 主流程 ────────────────────────────────────────────────────────────────

def parse_args():
    """解析调用远程语言模型 API 进行零样本文本匹配实验所需的命令行配置。

    用途：
        解析调用远程语言模型 API 进行零样本文本匹配实验所需的命令行配置。

    使用场景与调用位置：
        本文件 main() 的启动配置阶段调用；独立运行脚本时首先经过此处。 本文件直接调用位置：main()。

    参数：
        无显式参数；所需配置由内部逻辑、模块常量或命令行提供。

    返回值：
        argparse.Namespace；成员名对应下方 add_argument 定义。

    注意事项：
        由本文件 main 在任何主要工作开始前调用；读取 sys.argv，处理默认值、类型和 choices；--help 或不合法参数通过 argparse 触发 SystemExit，不启动训练或推理。
    """
    parser = argparse.ArgumentParser(description="LLM zero-shot 文本匹配评估")
    parser.add_argument("--data_dir",    default=str(DATA_DIR), type=str)
    parser.add_argument("--split",       default="validation",
                        choices=["validation", "test"])
    parser.add_argument("--num_samples", default=100, type=int,
                        help="评估样本数（默认 100，全集太贵；AFQMC val 共 4316 条）")
    parser.add_argument("--model",       default="qwen-plus", type=str)
    parser.add_argument("--sleep_sec",   default=0.2, type=float,
                        help="每次调用后等待时间（秒），避免触发限流")
    return parser.parse_args()


def main():
    """组织本脚本的完整流程：调用远程语言模型 API 进行零样本文本匹配实验。

    用途：
        组织本脚本的完整流程：调用远程语言模型 API 进行零样本文本匹配实验。

    使用场景与调用位置：
        直接运行此文件时由 if __name__ == "__main__" 入口调用；也可显式调用 main()，此时仍读取进程命令行。

    参数：
        无显式参数；所需配置由内部逻辑、模块常量或命令行提供。

    返回值：
        None；产物及副作用见注意事项，配置由命令行读取。

    注意事项：
        读取 DASHSCOPE_API_KEY，无 key 提前返回；从所选 split 抽约 1/3 正例并调用远程 API。当前未接入固定 manifest，且失败排除后评分，不能直接当作新协议统一结果。
    """
    args = parse_args()

    api_key = os.getenv("DASHSCOPE_API_KEY")
    if not api_key:
        print("❌ 未设置 DASHSCOPE_API_KEY 环境变量")
        print("   请运行：export DASHSCOPE_API_KEY='sk-xxx'")
        return

    # 读取数据
    data_path = Path(args.data_dir) / f"{args.split}.jsonl"
    rows = []
    with open(data_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))

    # 随机采样（保持正负比例）
    pos_rows = [r for r in rows if r["label"] == 1]
    neg_rows = [r for r in rows if r["label"] == 0]
    n_pos = min(args.num_samples // 3, len(pos_rows))   # 约 1/3 正样本
    n_neg = args.num_samples - n_pos
    samples = random.sample(pos_rows, n_pos) + random.sample(neg_rows, min(n_neg, len(neg_rows)))
    random.shuffle(samples)

    print(f"数据集: {data_path.name}  样本数: {len(samples)}")
    print(f"  正样本: {n_pos}  负样本: {len(samples) - n_pos}")
    print(f"模型: {args.model}")
    print(f"\nPrompt 示例：")
    ex = samples[0]
    print(PROMPT_TEMPLATE.format(s1=ex["sentence1"], s2=ex["sentence2"]))
    print("─" * 50)

    from openai import OpenAI
    client = OpenAI(api_key=api_key, base_url=DASHSCOPE_URL)

    print(f"\n开始评估（共 {len(samples)} 条，预计 {len(samples) * (0.5 + args.sleep_sec):.0f}s）...")
    results, parse_fail = evaluate_llm(samples, client, args.model, args.sleep_sec)

    metrics = compute_metrics(results)

    print(f"\n{'='*55}")
    print(f"LLM 评估结果（{args.model}，{len(samples)} 条样本）")
    print(f"  准确率 (Accuracy)  : {metrics['accuracy']:.4f}")
    print(f"  正例精确率         : {metrics['precision_pos']:.4f}")
    print(f"  正例召回率         : {metrics['recall_pos']:.4f}")
    print(f"  正例 F1            : {metrics['f1_pos']:.4f}")
    print(f"  有效预测数         : {metrics['n_valid']}")
    print(f"  解析失败数         : {metrics['n_fail']}")

    print(f"\n{'─'*55}")
    print("对比参考（来自 BiEncoder/CrossEncoder 训练结果，val 集全量）：")
    print("  指标            | BiEncoder | CrossEncoder | LLM zero-shot")
    print("  Accuracy        |  (见训练日志)  |  (见训练日志)  | "
          f"{metrics['accuracy']:.4f} ({len(samples)} 样本)")
    print("  推理速度        |   毫秒级       |   秒级         |  秒级+网络延迟")
    print("  可检索（向量）  |    ✓           |    ✗           |   ✗")
    print("  需要训练        |    ✓           |    ✓           |   ✗")

    # 保存失败案例
    fail_cases = [r for r in results if r["pred"] != r["label"]][:10]
    if fail_cases:
        print(f"\n前 {len(fail_cases)} 条预测错误样本：")
        for r in fail_cases:
            label_str = "相似" if r["label"] == 1 else "不相似"
            pred_str  = "相似" if r["pred"] == 1 else ("不相似" if r["pred"] == 0 else "解析失败")
            print(f"  [真:{label_str} | 预:{pred_str}]")
            print(f"    {r['sentence1']!r}  ||  {r['sentence2']!r}")

    # 保存结果到文件，供 evaluate_sft.py 读取比较
    log_dir = ROOT / "outputs" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    out_path = log_dir / "llm_compare_results.json"
    save_data = {
        "model": args.model,
        "n_samples": len(samples),
        "metrics": {k: float(v) if isinstance(v, float) else v
                    for k, v in metrics.items()},
        "results": results,
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(save_data, f, ensure_ascii=False, indent=2)
    print(f"\n结果已保存 → {out_path}")


if __name__ == "__main__":
    main()
