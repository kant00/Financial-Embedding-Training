"""无需模型权重的对比入口回归检查：python -m unittest discover -s tests。"""
import io
import json
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import compare_methods as comparison


class ComparisonInputsTest(unittest.TestCase):
    def args(self, allow_missing=False):
        return SimpleNamespace(data_dir=Path("data"), split="validation",
                               ckpt_dir=Path("checkpoints"), allow_missing=allow_missing)

    def validate(self, rows, exists=True, allow_missing=False):
        text = "\n".join(json.dumps(row) for row in rows)
        with patch.object(Path, "open", return_value=io.StringIO(text)), patch.object(Path, "is_file", return_value=exists):
            return comparison.validate_inputs(self.args(allow_missing))

    def test_unpublished_test_labels_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "-1"):
            self.validate([{"sentence1": "a", "sentence2": "b", "label": -1}])

    def test_empty_data_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "为空"):
            self.validate([])

    def test_missing_checkpoints_do_not_silently_produce_a_full_report(self):
        rows = [{"sentence1": "a", "sentence2": "b", "label": 1}]
        with self.assertRaises(FileNotFoundError):
            self.validate(rows, exists=False)
        with self.assertRaisesRegex(FileNotFoundError, "没有可用"):
            self.validate(rows, exists=False, allow_missing=True)

    def test_explicit_partial_comparison_preserves_sample_order(self):
        rows = [{"sentence1": "b", "sentence2": "a", "label": 0},
                {"sentence1": "a", "sentence2": "a", "label": 1}]
        text = "\n".join(json.dumps(row) for row in rows)
        with patch.object(Path, "open", return_value=io.StringIO(text)), patch.object(
                Path, "is_file", autospec=True, side_effect=lambda p: p.name == "biencoder_inbatch_best.pt"):
            actual, missing = comparison.validate_inputs(self.args(True))
        self.assertEqual(actual, rows)
        self.assertEqual(set(missing), {"biencoder_cosine", "biencoder_triplet", "crossencoder"})

    def test_report_uses_measured_f1_and_labels_partial_validation(self):
        results = []
        for method, f1 in zip(comparison.METHODS[:2], [0.8, 0.6]):
            result = dict(method, training_config={}, best_epoch=1, eval_max_length=64,
                          threshold=0.5, threshold_source="checkpoint")
            result.update({key: f1 for key in comparison.METRIC_KEYS})
            results.append(result)
        metadata = dict(data_path="validation.jsonl", data_sha256="abc", n_samples=2,
                        split="validation", missing_methods=["biencoder_inbatch", "crossencoder"])
        with patch.object(Path, "write_text") as write:
            comparison.build_report(results, metadata, Path("output"), False, [])
        report = write.call_args.args[0]
        self.assertIn("不是独立测试集成绩", report)
        self.assertIn("部分模型报告", report)
        self.assertIn("20.00 个百分点", report)
        self.assertNotIn("validation_f1_curves.png", report)
        self.assertIn("biencoder_cosine（0.8000）", report)


if __name__ == "__main__":
    unittest.main()
