"""Regression checks against recorded experiments and strict threshold boundaries."""
from __future__ import annotations

import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from evaluate import evaluate, load_scores, predict
from sweep import passing_counts, sweep_rows
from select_threshold import load_grid, select
from compare_vectors import load_vectors

DATASETS = {
    "artbench1k": ("exp4/threshold_eval/full_sweep", (30779, 121, 499379, 221)),
    "wikiart1k": ("exp4/datasets6_sweep", (30770, 225, 499275, 230)),
    "pixiv1k": ("exp4/datasets7_sweep", (30890, 107708, 391792, 110)),
}


class ReproductionTests(unittest.TestCase):
    def test_frozen_operating_point(self):
        for dataset, (_, expected) in DATASETS.items():
            with self.subTest(dataset=dataset):
                scores = load_scores(ROOT / "results/scores" / f"{dataset}.npz")
                actual = evaluate(scores, .87, .50, "and")
                self.assertEqual(tuple(actual[k] for k in ("TP", "FP", "TN", "FN")), expected)

    def test_all_30603_historical_grid_rows(self):
        for dataset, (directory, _) in DATASETS.items():
            with self.subTest(dataset=dataset):
                scores = load_scores(ROOT / "results/scores" / f"{dataset}.npz")
                actual = list(sweep_rows(scores, np.arange(0, 1.01, .01)))
                with (ROOT / "results/reported" / directory / "threshold_sweep_summary.csv").open() as stream:
                    recorded = {(float(r["clip_threshold"]), float(r["vit_threshold"])): r for r in csv.DictReader(stream)}
                self.assertEqual(len(actual), 10201)
                self.assertEqual(len(recorded), 10201)
                for row in actual:
                    expected = recorded[(row["clip_threshold"], row["vit_threshold"])]
                    for key in ("TP", "FP", "TN", "FN"):
                        self.assertEqual(row[key], int(expected[key]), (dataset, key, row))
                    self.assertAlmostEqual(row["f1"], float(expected["f1"]), places=6)

    def test_weighted_selection_matches_slides(self):
        grids = [load_grid(ROOT / "results/reported" / DATASETS[name][0] / "threshold_sweep_summary.csv")
                 for name in ("artbench1k", "wikiart1k")]
        result = select(grids, .99)
        self.assertEqual(result["region_counts"], [374, 243])
        self.assertEqual((result["clip_threshold"], result["vit_threshold"]), (.87, .50))
        self.assertAlmostEqual(result["weighted_f1"], .9937605721231767)

    def test_single_model_absolute_protocol_matches_record(self):
        reference = json.loads((ROOT / "results/reported/analysis/ComparisonSep14/single_model_metrics.json").read_text())
        for dataset, record in zip(("artbench1k", "wikiart1k"), reference["datasets"]):
            scores = load_scores(ROOT / "results/scores" / f"{dataset}.npz", "absolute")
            for rule in ("clip", "vit"):
                actual = evaluate(scores, .87, .50, rule)
                expected = record[rule + "_only"]
                self.assertEqual(tuple(actual[k] for k in ("TP", "FP", "TN", "FN")),
                                 tuple(expected[k] for k in ("TP", "FP", "TN", "FN")))


class ProtocolTests(unittest.TestCase):
    def test_strict_boundaries_and_negative_scores(self):
        values = np.array([-.5, 0, .5, np.nextafter(.5, 1), 1.0])
        actual = predict(values, np.ones(5), .5, .5)
        np.testing.assert_array_equal(actual, [False, False, False, True, True])

    def test_histogram_counts_equal_direct_rules(self):
        # Include exact thresholds, negative scores, adjacent floats and >1 rounding noise.
        clip = np.array([-.3, 0, .5, .6, .7, .9, 1, np.nextafter(np.float32(1), np.float32(2))], dtype=np.float32)
        vit = np.array([.9, 0, .5, .7, .6, .4, 1, 1], dtype=np.float32)
        grid = np.arange(0, 1.01, .01)
        counts = passing_counts(clip, vit, grid)
        for ci, ct in enumerate(grid):
            for vi, vt in enumerate(grid):
                expected = ((clip.astype(np.float64) > ct) & (vit.astype(np.float64) > vt)).sum()
                self.assertEqual(counts[ci, vi], expected)

    def test_rules_apply_identically_to_positive_and_negative_arrays(self):
        c, v = np.array([.9, .7, .9]), np.array([.6, .6, .4])
        scores = {"pos_clip": c, "neg_clip": c, "pos_vit": v, "neg_vit": v}
        for rule in ("and", "or", "clip", "vit"):
            actual = evaluate(scores, .8, .5, rule)
            self.assertEqual(actual["TP"], actual["FP"])
            self.assertEqual(actual["FN"], actual["TN"])

    def test_misaligned_pair_arrays_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.npz"
            np.savez(path, pos_clip=np.ones(2), pos_vit=np.ones(1), neg_clip=np.zeros(2), neg_vit=np.zeros(2))
            with self.assertRaisesRegex(ValueError, "pair counts differ"):
                load_scores(path)

    def test_misaligned_vector_rows_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            for kind, names in (("clip", ["a", "b"]), ("vit", ["b", "a"])):
                np.savez(path / f"{kind}_vectors.npz", embeddings=np.eye(2), image_paths=names, model=kind)
            with self.assertRaisesRegex(ValueError, "row order differs"):
                load_vectors(path)

    def test_incompatible_selection_grids_rejected(self):
        with self.assertRaisesRegex(ValueError, "same threshold grid"):
            select([{(.8, .5): .995}, {(.9, .5): .996}], .99)


if __name__ == "__main__":
    unittest.main()
