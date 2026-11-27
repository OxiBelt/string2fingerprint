"""Tests for `build_models.py` and the model presets in `models.json`."""

import json
import unittest
from pathlib import Path

import build_models

HERE = Path(__file__).resolve().parent


class PresetTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = json.loads((HERE / "models.json").read_text())

    def test_presets_are_valid(self):
        names = [model["name"] for model in self.config["models"]]
        self.assertEqual(len(names), len(set(names)))
        self.assertIn("default", names)
        for model in self.config["models"]:
            with self.subTest(model=model["name"]):
                self.assertRegex(model["name"], r"^[a-z][a-z0-9_]*$")
                self.assertTrue(1 <= len(model["hidden"]) <= 3)  # `MAX_LAYERS` is 4
                self.assertTrue(all(1 <= h <= 256 for h in model["hidden"]))  # `MAX_HIDDEN`
                self.assertLessEqual(model["quantized_epochs"], model["epochs"])
                self.assertTrue(model["description"])

    def test_splits_use_distinct_seeds(self):
        seeds = [split["seed"] for split in self.config["data"]["splits"].values()]
        self.assertEqual(len(seeds), len(set(seeds)))
        self.assertEqual(set(self.config["data"]["splits"]), {"train", "val", "test"})


class SummaryTest(unittest.TestCase):
    def test_markdown_table_has_a_row_per_model(self):
        rows = [{
            "name": "x", "dimensions": [3072, 8, 8], "bytes": 1234, "mean_auc": 0.9,
            "attack_any_recall@128": 0.95, "benign_any_fpr@128": 0.01,
            "development": {"attacks_flagged": 1, "attacks": 2, "benign_flagged": 0, "benign": 3},
        }]
        table = build_models.markdown(rows).splitlines()
        self.assertEqual(len(table), 3)
        self.assertIn("| `x` | 3072, 8, 8 | 1,234 | 0.900 | 95.0% | 1.00% | 1 of 2 | 0 of 3 |", table)


if __name__ == "__main__":
    unittest.main()
