"""
构建固定 train/calibration/final-test 协议及可追溯元数据。

用途与流程位置：
    原始 AFQMC 下载后、任何模型训练前运行；不改写 source_dir 内的原始数据。
    原 train 拆为 train + calibration，原 validation 用作有标签 final test。
    原官方 test 标签为 -1，不进入指标计算。

输入与输出：
    默认分层划分，seed=42、calibration_ratio=0.1；--strict_group_split 按共享句子分组。
    写入 train、calibration/validation 别名、final_test/test 别名、evaluation_manifest。
    split_metadata.json 记录源哈希、样本/标签数量及跨集合句子重叠。

使用方式（在项目根目录执行，Bash）：
    python src/prepare_experiment_data.py

注意事项：
    strict 只隔离 train/calibration，不保证 train/final test 没有共享句子。
    若旧实验已用官方 validation 调参，不能宣称新 final test 从未被接触。
    输出同名文件会覆盖；实验开始后冻结协议，不要边训练边重新划分。

说明核对：2026-09-28；完整流程、前置条件及自定义模型路径见 USAGE_GUIDE.md。
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

from experiment_utils import save_json, sha256_file, write_jsonl


ROOT = Path(__file__).parent.parent
DEFAULT_SOURCE = ROOT / "data" / "afqmc"
DEFAULT_OUTPUT = ROOT / "data" / "experiments" / "afqmc_seed42"


def load_jsonl(path: Path) -> list[dict]:
    """读取 UTF-8 JSONL 并按文件顺序还原记录。

    用途：
        读取 UTF-8 JSONL 并按文件顺序还原记录。

    使用场景与调用位置：
        在 src/prepare_experiment_data.py 的“生成 train/calibration/final-test 数据协议”流程中使用。 本文件直接调用位置：main()。

    参数：
        path: 待读取或写入的文件路径，具体模式见用途与注意事项。

    返回值：
        list[dict]；每个非空行对应一条记录。

    注意事项：
        用于生成 train/calibration/final-test 数据协议阶段的数据读取；忽略空行，不执行字段、标签或重复样本校验；缺失文件和非法 JSON 异常直接上抛，整个文件驻留内存。
    """
    with open(path, encoding="utf-8") as file_obj:
        return [json.loads(line) for line in file_obj if line.strip()]


def normalize_text(text: str) -> str:
    """给泄漏检查提供统一文本键。

    用途：
        给泄漏检查提供统一文本键。

    使用场景与调用位置：
        在 src/prepare_experiment_data.py 的“生成 train/calibration/final-test 数据协议”流程中使用。 本文件直接调用位置：grouped_split()、sentence_overlap()。

    参数：
        text: 待规范化或编码的单个文本字符串。

    返回值：
        去除全部空白并转小写的字符串。

    注意事项：
        grouped_split 和 sentence_overlap 调用；不做同义词归并，空白/大小写不同文本会被视为同一键。
    """
    return "".join(str(text).strip().lower().split())


def stratified_split(rows: list[dict], calibration_ratio: float, seed: int):
    """按 0/1 类别分别随机切分训练和校准样本。

    用途：
        按 0/1 类别分别随机切分训练和校准样本。

    使用场景与调用位置：
        在 src/prepare_experiment_data.py 的“生成 train/calibration/final-test 数据协议”流程中使用。

    参数：
        rows: 原始记录列表；句对通常包含 sentence1、sentence2、label。
        calibration_ratio: 希望划给校准集的比例，通常为 0.1。
        seed: 可复现实验的整数随机种子。

    返回值：
        (train_rows, calibration_rows) 两个列表。

    注意事项：
        main 默认选择；局部 random.Random(seed) 不改变全局 RNG；每类至少分配 1 条校准样本，小数据可能使训练集缺类。该模式不保证句子隔离。
    """
    rng = random.Random(seed)
    by_label = defaultdict(list)
    for row in rows:
        by_label[int(row["label"])].append(row)

    train_rows, calibration_rows = [], []
    for label_rows in by_label.values():
        rng.shuffle(label_rows)
        n_calibration = max(1, round(len(label_rows) * calibration_ratio))
        calibration_rows.extend(label_rows[:n_calibration])
        train_rows.extend(label_rows[n_calibration:])
    rng.shuffle(train_rows)
    rng.shuffle(calibration_rows)
    return train_rows, calibration_rows


class UnionFind:
    """维护句子图连通分量以支持分组划分。

    用途：
        维护句子图连通分量以支持分组划分。

    使用场景与调用位置：
        在 src/prepare_experiment_data.py 的“生成 train/calibration/final-test 数据协议”流程中使用。 本文件直接调用位置：grouped_split()。

    参数：
        构造参数见本类 __init__；实例成员与样本/模型结构见注意事项。

    返回值：
        带 parent 字典的并查集实例。

    注意事项：
        grouped_split 内部创建；路径压缩加速查询，无按秩合并；即使 label=0 的句对也连边，因为目标是隔离共享文本而不是推断同义关系。
    """

    def __init__(self):
        """初始化空的节点父指针表。

        用途：
            初始化空的节点父指针表。

        使用场景与调用位置：
            调用 UnionFind(...) 创建实例时自动执行；在训练或评估加载模型/数据的准备阶段使用。

        参数：
            无显式业务参数；实例方法通过 self 读取内部状态，入口函数从命令行获取配置。

        返回值：
            None；self.parent 置为空字典。

        注意事项：
            构造 UnionFind() 时自动调用；节点在 find 中按需添加。
        """
        self.parent: dict[str, str] = {}

    def find(self, item: str) -> str:
        """查找节点的根并压缩路径。

        用途：
            查找节点的根并压缩路径。

        使用场景与调用位置：
            在 src/prepare_experiment_data.py 的“生成 train/calibration/final-test 数据协议”流程中使用。 所属作用域：UnionFind。

        参数：
            item: 并查集节点的文本标识，通常是规范化句子。

        返回值：
            表示根节点的字符串。

        注意事项：
            union 和 grouped_split 查询所属分量时调用；未见节点自动作为自身根加入；递归实现可能受极端链长和 Python 递归上限影响。
        """
        self.parent.setdefault(item, item)
        if self.parent[item] != item:
            self.parent[item] = self.find(self.parent[item])
        return self.parent[item]

    def union(self, left: str, right: str) -> None:
        """把两个句子所属连通分量合并。

        用途：
            把两个句子所属连通分量合并。

        使用场景与调用位置：
            在 src/prepare_experiment_data.py 的“生成 train/calibration/final-test 数据协议”流程中使用。 所属作用域：UnionFind。

        参数：
            left: 待合并的左侧句子节点。
            right: 待合并的右侧句子节点。

        返回值：
            None；修改 parent 中一个根的父指针。

        注意事项：
            grouped_split 遍历句对时调用；两者本已同根则不改变结构。
        """
        left_root, right_root = self.find(left), self.find(right)
        if left_root != right_root:
            self.parent[right_root] = left_root


def grouped_split(rows: list[dict], calibration_ratio: float, seed: int):
    """按共享句子连通分量进行不可拆分的 train/calibration 划分。

    用途：
        按共享句子连通分量进行不可拆分的 train/calibration 划分。

    使用场景与调用位置：
        在 src/prepare_experiment_data.py 的“生成 train/calibration/final-test 数据协议”流程中使用。

    参数：
        rows: 原始记录列表；句对通常包含 sentence1、sentence2、label。
        calibration_ratio: 希望划给校准集的比例，通常为 0.1。
        seed: 可复现实验的整数随机种子。

    返回值：
        (train_rows, calibration_rows) 两个列表。

    注意事项：
        main 启用 strict_group_split 时选择；先打乱再按分量大小排序，贪心接近目标规模，不保证类别比例。只隔离 train/calibration，不能据此声称与官方 final test 也零交叉。
    """
    union_find = UnionFind()
    for row in rows:
        union_find.union(normalize_text(row["sentence1"]), normalize_text(row["sentence2"]))

    components = defaultdict(list)
    for row in rows:
        root = union_find.find(normalize_text(row["sentence1"]))
        components[root].append(row)

    rng = random.Random(seed)
    groups = list(components.values())
    rng.shuffle(groups)
    # 大分量优先可减少贪心划分对目标规模的过冲。
    groups.sort(key=len, reverse=True)
    target = round(len(rows) * calibration_ratio)
    calibration_rows, train_rows = [], []
    for group in groups:
        current_gap = abs(target - len(calibration_rows))
        added_gap = abs(target - (len(calibration_rows) + len(group)))
        if added_gap <= current_gap:
            calibration_rows.extend(group)
        else:
            train_rows.extend(group)
    rng.shuffle(train_rows)
    rng.shuffle(calibration_rows)
    return train_rows, calibration_rows


def split_stats(rows: list[dict]) -> dict:
    """统计某个划分的总数及标签数。

    用途：
        统计某个划分的总数及标签数。

    使用场景与调用位置：
        在 src/prepare_experiment_data.py 的“生成 train/calibration/final-test 数据协议”流程中使用。 本文件直接调用位置：main()。

    参数：
        rows: 原始记录列表；句对通常包含 sentence1、sentence2、label。

    返回值：
        total 和 label_counts 字典；标签键为字符串。

    注意事项：
        main 生成 split_metadata 时调用；计数所有出现的整数标签，不自动过滤 -1。
    """
    counts = Counter(int(row["label"]) for row in rows)
    return {"total": len(rows), "label_counts": {str(k): v for k, v in sorted(counts.items())}}


def sentence_overlap(left_rows: list[dict], right_rows: list[dict]) -> int:
    """检查右侧句对中至少一句是否出现在左侧句子集合。

    用途：
        检查右侧句对中至少一句是否出现在左侧句子集合。

    使用场景与调用位置：
        在 src/prepare_experiment_data.py 的“生成 train/calibration/final-test 数据协议”流程中使用。 本文件直接调用位置：main()。

    参数：
        left_rows: 作为参考句子集合的左侧数据列表。
        right_rows: 需要检查交叉情况的右侧句对列表。

    返回值：
        满足条件的右侧句对条数 int。

    注意事项：
        main 记录划分交叉风险；不是唯一重复句子的数量，也不是重复句对数量；右侧两句均重复只计该条一次。
    """
    left_sentences = {
        normalize_text(row[key]) for row in left_rows for key in ("sentence1", "sentence2")
    }
    return sum(
        1
        for row in right_rows
        if normalize_text(row["sentence1"]) in left_sentences
        or normalize_text(row["sentence2"]) in left_sentences
    )


def parse_args():
    """解析生成 train/calibration/final-test 数据协议所需的命令行配置。

    用途：
        解析生成 train/calibration/final-test 数据协议所需的命令行配置。

    使用场景与调用位置：
        本文件 main() 的启动配置阶段调用；独立运行脚本时首先经过此处。 本文件直接调用位置：main()。

    参数：
        无显式参数；所需配置由内部逻辑、模块常量或命令行提供。

    返回值：
        argparse.Namespace；成员名对应下方 add_argument 定义。

    注意事项：
        由本文件 main 在任何主要工作开始前调用；读取 sys.argv，处理默认值、类型和 choices；--help 或不合法参数通过 argparse 触发 SystemExit，不启动训练或推理。
    """
    parser = argparse.ArgumentParser(description="生成可信文本匹配实验划分")
    parser.add_argument("--source_dir", default=str(DEFAULT_SOURCE))
    parser.add_argument("--output_dir", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--calibration_ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--strict_group_split", action="store_true")
    return parser.parse_args()


def main():
    """组织本脚本的完整流程：生成 train/calibration/final-test 数据协议。

    用途：
        组织本脚本的完整流程：生成 train/calibration/final-test 数据协议。

    使用场景与调用位置：
        直接运行此文件时由 if __name__ == "__main__" 入口调用；也可显式调用 main()，此时仍读取进程命令行。

    参数：
        无显式参数；所需配置由内部逻辑、模块常量或命令行提供。

    返回值：
        None；产物及副作用见注意事项，配置由命令行读取。

    注意事项：
        保持官方 validation 作为 final test，从原 train 分出 calibration；strict_group_split 只隔离 train/calibration。写入多个别名文件、manifest 和哈希元数据，同名文件覆盖。
    """
    args = parse_args()
    source_dir, output_dir = Path(args.source_dir), Path(args.output_dir)
    train_source = source_dir / "train.jsonl"
    validation_source = source_dir / "validation.jsonl"
    source_rows = load_jsonl(train_source)
    final_test_rows = load_jsonl(validation_source)

    splitter = grouped_split if args.strict_group_split else stratified_split
    train_rows, calibration_rows = splitter(
        source_rows, args.calibration_ratio, args.seed
    )

    # 兼容现有训练器的 validation/test 文件命名，同时保留语义明确的别名。
    write_jsonl(train_rows, output_dir / "train.jsonl")
    write_jsonl(calibration_rows, output_dir / "calibration.jsonl")
    write_jsonl(calibration_rows, output_dir / "validation.jsonl")
    write_jsonl(final_test_rows, output_dir / "final_test.jsonl")
    write_jsonl(final_test_rows, output_dir / "test.jsonl")

    manifest_rows = []
    for index, row in enumerate(final_test_rows):
        manifest_rows.append({"sample_id": f"afqmc-final-{index:05d}", **row})
    write_jsonl(manifest_rows, output_dir / "evaluation_manifest.jsonl")

    metadata = {
        "protocol_version": 1,
        "seed": args.seed,
        "calibration_ratio": args.calibration_ratio,
        "strict_group_split": args.strict_group_split,
        "source_hashes": {
            "train_sha256": sha256_file(train_source),
            "official_validation_sha256": sha256_file(validation_source),
        },
        "splits": {
            "train": split_stats(train_rows),
            "calibration": split_stats(calibration_rows),
            "final_test": split_stats(final_test_rows),
        },
        "leakage_checks": {
            "train_calibration_pairs_with_seen_sentence": sentence_overlap(
                train_rows, calibration_rows
            ),
            "train_final_test_pairs_with_seen_sentence": sentence_overlap(
                train_rows, final_test_rows
            ),
        },
        "notes": [
            "validation.jsonl 是 calibration.jsonl 的兼容别名。",
            "test.jsonl 是 final_test.jsonl 的兼容别名，不是官方无标签 test。",
            "final_test 只能在模型、阈值和超参数冻结后评估。",
        ],
    }
    save_json(metadata, output_dir / "split_metadata.json")
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
