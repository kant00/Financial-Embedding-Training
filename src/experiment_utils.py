"""
实验公共工具：随机状态、环境快照、文件哈希与 JSON/JSONL 落盘。

用途与调用位置：
    数据准备、训练、评估和报告相关模块复用此处工具，无独立命令行入口。
    set_global_seed 设置随机状态及确定性选项；environment_snapshot 记录运行环境。
    sha256_file 用于源数据追溯；to_builtin 转换可序列化对象；
    save_json / write_jsonl 负责创建父目录并写入结果。

输入与输出：
    接收种子、文件路径或待保存对象，返回值和副作用见各函数说明。
    模型训练与指标选择不在本模块中实现。

注意事项：
    相同种子不保证跨硬件、依赖版本或全部算子逐位一致；仍需保存环境和配置。
    同名结果文件会覆盖；协议、数据哈希和输出目录应随实验归档。
    使用流程见 USAGE_GUIDE.md。

说明核对：2026-09-28；完整流程、前置条件及自定义模型路径见 USAGE_GUIDE.md。
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import random
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterable

DEFAULT_SEEDS = (13, 42, 2026)
DEFAULT_BERT_MODEL = "google-bert/bert-base-chinese"


def set_global_seed(seed: int, deterministic: bool = True) -> None:
    """设置训练涉及的 Python、NumPy、Torch 与 CUDA 随机状态。

    用途：
        设置训练涉及的 Python、NumPy、Torch 与 CUDA 随机状态。

    使用场景与调用位置：
        在 src/experiment_utils.py 的“提供复现、文件哈希、序列化和环境快照公共能力”流程中使用。 跨文件调用：src/mine_hard_negatives.py:main()、src/train_biencoder.py:main()、src/train_crossencoder.py:main()。

    参数：
        seed: 可复现实验的整数随机种子。
        deterministic: 是否启用确定性后端设置及算子警告模式。

    返回值：
        None；修改进程全局 RNG 与后端选项。

    注意事项：
        训练/挖掘启动时调用；确定性算子使用 warn_only。运行时设置 PYTHONHASHSEED 不会重置当前解释器的哈希种子；需启动前配置才能保证 set 迭代顺序。
    """
    import numpy as np
    import torch

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        # warn_only 避免少数 GPU 算子没有确定性实现时直接中断训练。
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def sha256_file(path: str | Path) -> str:
    """用流式读取计算数据文件的内容指纹。

    用途：
        用流式读取计算数据文件的内容指纹。

    使用场景与调用位置：
        在 src/experiment_utils.py 的“提供复现、文件哈希、序列化和环境快照公共能力”流程中使用。 跨文件调用：src/prepare_experiment_data.py:main()。

    参数：
        path: 待读取或写入的文件路径，具体模式见用途与注意事项。

    返回值：
        64 位十六进制 SHA-256 字符串。

    注意事项：
        数据协议生成时记录来源；每块最多 1 MiB，避免一次读入大文件；文件不存在或无权限时抛出 I/O 异常。
    """
    digest = hashlib.sha256()
    with open(path, "rb") as file_obj:
        for chunk in iter(lambda: file_obj.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def to_builtin(value: Any) -> Any:
    """递归转换常见实验对象为 JSON 原生类型。

    用途：
        递归转换常见实验对象为 JSON 原生类型。

    使用场景与调用位置：
        在 src/experiment_utils.py 的“提供复现、文件哈希、序列化和环境快照公共能力”流程中使用。 本文件直接调用位置：save_json()、write_jsonl()。

    参数：
        value: 待转换、解析或格式化的输入值。

    返回值：
        转换后的字典/列表/标量/字符串，或未处理对象本身。

    注意事项：
        save_json/write_jsonl 序列化前调用；Path 转字符串，tuple 转列表，item 适用于标量；不支持任意多元素数组或张量。
    """
    if isinstance(value, dict):
        return {str(k): to_builtin(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_builtin(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "item"):
        return value.item()
    return value


def save_json(data: Any, path: str | Path) -> Path:
    """保存完整实验配置或指标 JSON。

    用途：
        保存完整实验配置或指标 JSON。

    使用场景与调用位置：
        在 src/experiment_utils.py 的“提供复现、文件哈希、序列化和环境快照公共能力”流程中使用。 跨文件调用：src/evaluate.py:main()、src/evaluate_retrieval.py:main()、src/mine_hard_negatives.py:main()、src/prepare_experiment_data.py:main()、src/run_baselines.py:evaluate_and_save()、src/run_baselines.py:main()、src/run_experiment_suite.py:main()、src/train_biencoder.py:main()、src/train_crossencoder.py:main()、src_llm/evaluate_sft.py:main()。

    参数：
        data: 已解析的记录或待保存对象，结构见返回值与注意事项。
        path: 待读取或写入的文件路径，具体模式见用途与注意事项。

    返回值：
        目标 pathlib.Path。

    注意事项：
        自动创建父目录并覆盖同名文件；先 to_builtin 再 UTF-8 序列化，缩进为 2。写入不是原子替换。
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as file_obj:
        json.dump(to_builtin(data), file_obj, ensure_ascii=False, indent=2)
    return path


def write_jsonl(rows: Iterable[dict], path: str | Path) -> Path:
    """将可迭代样本逐条写入 JSONL。

    用途：
        将可迭代样本逐条写入 JSONL。

    使用场景与调用位置：
        在 src/experiment_utils.py 的“提供复现、文件哈希、序列化和环境快照公共能力”流程中使用。 跨文件调用：src/evaluate.py:main()、src/evaluate_retrieval.py:main()、src/mine_hard_negatives.py:main()、src/prepare_experiment_data.py:main()、src/run_baselines.py:evaluate_and_save()、src_llm/evaluate_sft.py:main()。

    参数：
        rows: 原始记录列表；句对通常包含 sentence1、sentence2、label。
        path: 待读取或写入的文件路径，具体模式见用途与注意事项。

    返回值：
        目标 pathlib.Path。

    注意事项：
        数据协议、预测明细和难负样本共用；自动创建父目录，以 w 模式覆盖旧内容而不是追加，生成器输入会被消费一次。
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as file_obj:
        for row in rows:
            file_obj.write(json.dumps(to_builtin(row), ensure_ascii=False) + "\n")
    return path


def environment_snapshot(include_torch: bool = True) -> dict[str, Any]:
    """记录 Python、系统、Torch、CUDA、GPU 和 Git commit。

    用途：
        记录 Python、系统、Torch、CUDA、GPU 和 Git commit。

    使用场景与调用位置：
        在 src/experiment_utils.py 的“提供复现、文件哈希、序列化和环境快照公共能力”流程中使用。 跨文件调用：src/evaluate.py:main()、src/evaluate_retrieval.py:main()、src/run_baselines.py:evaluate_and_save()、src/train_biencoder.py:main()、src/train_crossencoder.py:main()、src_llm/evaluate_sft.py:main()。

    参数：
        include_torch: 是否探测 Torch 与 CUDA。CPU 文本基线传入 False。

    返回值：
        可 JSON 序列化的环境字典。

    注意事项：
        基线、训练 checkpoint 和评估日志调用。无 Git 仓库时 commit=None，无 CUDA 时 gpu=None。
    """
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        commit = None

    environment = {
        "python": sys.version,
        "platform": platform.platform(),
        "torch": None,
        "cuda_available": False,
        "cuda_version": None,
        "gpu": None,
        "git_commit": commit,
    }
    if include_torch:
        import torch

        cuda_available = torch.cuda.is_available()
        environment.update(
            torch=torch.__version__,
            cuda_available=cuda_available,
            cuda_version=torch.version.cuda,
            gpu=torch.cuda.get_device_name(0) if cuda_available else None,
        )
    return environment
