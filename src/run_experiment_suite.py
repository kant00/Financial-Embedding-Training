"""
实验套件：顺序调度多模型、多随机种子的训练和最终测试。

用途与流程位置：
    在数据、训练配置和方法集合冻结后运行；不是仅在 calibration 上调参的开发工具。
    支持 cosine、triplet、inbatch、crossencoder；默认模型集合不含 triplet。
    默认种子为 13、42、2026，每次训练完成立即运行 evaluate.py --split test。

输入与输出：
    显式指定协议、BERT 与输出目录；每组写入 方法/seed_种子 下的训练和评估文件。
    全部完成后生成 multi_seed_summary.json，包含单次记录以及指标 mean/std。
    --hard_negative_file 仅传给 triplet；Hard Triplet 应使用独立套件目录。

使用方式（在项目根目录执行，Bash）：
    python src/run_experiment_suite.py --data_dir data/experiments/afqmc_seed42 --output_dir outputs/experiment_suite --models cosine,triplet,inbatch,crossencoder --seeds 13,42,2026 --dry_run

注意事项：
    --dry_run 只打印命令；确认路径/资源后去掉此开关才真正执行。
    不支持自动断点恢复、跳过已有任务或合并多次子集运行；失败会中断，同名结果可被覆盖。
    只转发本脚本定义的选项，不会自动继承开发实验的自定义池化、温度或长度。

说明核对：2026-09-28；完整流程、前置条件及自定义模型路径见 USAGE_GUIDE.md。
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from experiment_utils import DEFAULT_BERT_MODEL, DEFAULT_SEEDS, save_json
from metrics import aggregate_seed_metrics


ROOT = Path(__file__).parent.parent
SRC_DIR = Path(__file__).parent
DEFAULT_DATA = ROOT / "data" / "experiments" / "afqmc_seed42"


def parse_csv(value):
    """把逗号分隔命令行参数拆成非空项。

    用途：
        把逗号分隔命令行参数拆成非空项。

    使用场景与调用位置：
        在 src/run_experiment_suite.py 的“顺序调度多模型、多种子的训练与最终评估子进程”流程中使用。

    参数：
        value: 待转换、解析或格式化的输入值。

    返回值：
        去除首尾空白后的字符串列表。

    注意事项：
        parse_args 用作 models/seeds 的 type；不验证模型名称，不将种子转整数，main 再做校验。
    """
    return [item.strip() for item in value.split(",") if item.strip()]


def run(command, dry_run=False):
    """打印并按需执行一个训练/评估子进程。

    用途：
        打印并按需执行一个训练/评估子进程。

    使用场景与调用位置：
        在 src/run_experiment_suite.py 的“顺序调度多模型、多种子的训练与最终评估子进程”流程中使用。 本文件直接调用位置：main()。

    参数：
        command: 子进程可执行文件与参数组成的列表。
        dry_run: 为 True 时仅打印命令而不启动子进程。

    返回值：
        None；非 dry_run 时等待子进程结束。

    注意事项：
        main 调度每个实验时调用；command 为参数列表而非 shell 字符串，check=True 使非零退出抛 CalledProcessError，后续任务不会自动继续。
    """
    print("$", subprocess.list2cmdline([str(item) for item in command]))
    if not dry_run:
        subprocess.run(command, check=True)


def parse_args():
    """解析顺序调度多模型、多种子的训练与最终评估子进程所需的命令行配置。

    用途：
        解析顺序调度多模型、多种子的训练与最终评估子进程所需的命令行配置。

    使用场景与调用位置：
        本文件 main() 的启动配置阶段调用；独立运行脚本时首先经过此处。 本文件直接调用位置：main()。

    参数：
        无显式参数；所需配置由内部逻辑、模块常量或命令行提供。

    返回值：
        argparse.Namespace；成员名对应下方 add_argument 定义。

    注意事项：
        由本文件 main 在任何主要工作开始前调用；读取 sys.argv，处理默认值、类型和 choices；--help 或不合法参数通过 argparse 触发 SystemExit，不启动训练或推理。
    """
    parser = argparse.ArgumentParser(description="三随机种子可信实验套件")
    parser.add_argument("--data_dir", default=str(DEFAULT_DATA))
    parser.add_argument("--bert_path", default=DEFAULT_BERT_MODEL)
    parser.add_argument("--output_dir", default=str(ROOT / "outputs" / "experiment_suite"))
    parser.add_argument("--models", type=parse_csv, default=["cosine", "inbatch", "crossencoder"])
    parser.add_argument("--seeds", type=parse_csv,
                        default=[str(seed) for seed in DEFAULT_SEEDS])
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_hidden_layers", type=int, default=4)
    parser.add_argument("--hard_negative_file", default=None)
    parser.add_argument("--dry_run", action="store_true")
    return parser.parse_args()


def main():
    """组织本脚本的完整流程：顺序调度多模型、多种子的训练与最终评估子进程。

    用途：
        组织本脚本的完整流程：顺序调度多模型、多种子的训练与最终评估子进程。

    使用场景与调用位置：
        直接运行此文件时由 if __name__ == "__main__" 入口调用；也可显式调用 main()，此时仍读取进程命令行。

    参数：
        无显式参数；所需配置由内部逻辑、模块常量或命令行提供。

    返回值：
        None；产物及副作用见注意事项，配置由命令行读取。

    注意事项：
        对子模型和 seed 顺序执行训练、final-test 评估，再计算 mean/std；dry_run 只打印命令。任何子进程非零退出会中断，不自动重试；旧实验目录同名产物会被子脚本覆盖。
    """
    args = parse_args()
    allowed = {"cosine", "triplet", "inbatch", "crossencoder"}
    unknown = set(args.models) - allowed
    if unknown:
        raise ValueError(f"未知模型: {sorted(unknown)}")
    seeds = [int(seed) for seed in args.seeds]
    output_root = Path(args.output_dir)
    records_by_model = {model: [] for model in args.models}

    for model_name in args.models:
        for seed in seeds:
            run_dir = output_root / model_name / f"seed_{seed}"
            common = [
                "--data_dir", args.data_dir,
                "--bert_path", args.bert_path,
                "--output_dir", str(run_dir),
                "--epochs", str(args.epochs),
                "--batch_size", str(args.batch_size),
                "--num_hidden_layers", str(args.num_hidden_layers),
                "--seed", str(seed),
            ]
            if model_name == "crossencoder":
                train_script = SRC_DIR / "train_crossencoder.py"
                checkpoint = run_dir / "checkpoints" / "crossencoder_best.pt"
                model_type = "crossencoder"
                train_command = [sys.executable, str(train_script), *common]
            else:
                train_script = SRC_DIR / "train_biencoder.py"
                checkpoint = run_dir / "checkpoints" / f"biencoder_{model_name}_best.pt"
                model_type = "biencoder"
                train_command = [
                    sys.executable, str(train_script), *common, "--loss", model_name
                ]
                if model_name == "triplet" and args.hard_negative_file:
                    train_command.extend(["--hard_negative_file", args.hard_negative_file])
            run(train_command, args.dry_run)

            evaluation_dir = run_dir / "evaluation"
            evaluate_command = [
                sys.executable, str(SRC_DIR / "evaluate.py"),
                "--model_type", model_type,
                "--ckpt", str(checkpoint),
                "--data_dir", args.data_dir,
                "--bert_path", args.bert_path,
                "--split", "test",
                "--output_dir", str(evaluation_dir),
                "--seed", str(seed),
            ]
            run(evaluate_command, args.dry_run)

            if not args.dry_run:
                metric_path = evaluation_dir / f"{model_type}_test_metrics.json"
                with open(metric_path, encoding="utf-8") as file_obj:
                    record = json.load(file_obj)
                record["seed"] = seed
                records_by_model[model_name].append(record)

    if not args.dry_run:
        summary = {
            model_name: aggregate_seed_metrics(records)
            for model_name, records in records_by_model.items()
        }
        save_json(
            {"seeds": seeds, "records": records_by_model, "summary": summary},
            output_root / "multi_seed_summary.json",
        )
        print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
