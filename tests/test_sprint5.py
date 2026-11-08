import csv
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from sklearn.metrics import f1_score

from src.eval.sprint5 import (CLFS, REPS, cluster_interval,
                              cluster_swap_p, macro, read_grid, video_indices)
from src.eval.density_groups import load_groups
from src.eval.kappa import kappa
from src.features.representations import FastTextDoc
from src.preprocess.density import Density


class Sprint5Tests(unittest.TestCase):
    def test_density_uses_only_counted_words(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "counts.csv"
            path.write_text("comment_id,kk_tokens,ru_tokens\na,2,2\nb,1,4\nc,2,3\nd,3,3\n")
            self.assertEqual(load_groups(path), {"a": "", "b": "low", "c": "med", "d": "high"})
        model = Density({"good": "ru"}, None, None, confident={"good"})
        self.assertEqual(model.measure("good <USER> <URL>")["counted"], 0)

    def test_fasttext_requires_training_video_and_unsampled_id(self):
        with tempfile.TemporaryDirectory() as temp:
            corpus, manifest = Path(temp) / "clean.jsonl", Path(temp) / "manifest.csv"
            manifest.write_text("comment_id\nx\n")
            corpus.write_text("\n".join(json.dumps(dict(comment_id=cid, video_id=v, text=t))
                              for cid, v, t in [("x", "train", "sample"), ("y", "test", "other"),
                                                ("z", "train", "allowed")]))
            with patch("src.features.representations.MANIFEST", manifest):
                rep = FastTextDoc(corpus=corpus, training_videos={"train"})
                self.assertEqual(list(rep._corpus_sentences()), [["allowed"]])

    def test_constant_agreement_has_undefined_kappa(self):
        self.assertIsNone(kappa([("mono", "mono")] * 10))

    def test_pairing_rejects_missing_system_row(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "grid.csv"
            rows = [dict(comment_id="one", video_id="video", gold="pos",
                         pred="pos", density="mono", fold="1", n_tokens="3",
                         representation=rep, classifier=clf)
                    for rep in REPS for clf in CLFS]
            with open(path, "w", encoding="utf-8", newline="") as file:
                writer = csv.DictWriter(file, fieldnames=rows[0])
                writer.writeheader()
                writer.writerows(rows)
            self.assertEqual(len(read_grid(path)[3]), 8)
            with self.assertRaisesRegex(ValueError, "frozen evaluation"):
                read_grid(path, expected={})
            with open(path, "w", encoding="utf-8", newline="") as file:
                writer = csv.DictWriter(file, fieldnames=rows[0])
                writer.writeheader()
                writer.writerows(rows[:-1])
            with self.assertRaises(ValueError):
                read_grid(path)

    def test_identical_predictions_have_zero_interval_and_unit_p(self):
        y = np.array(["pos", "neg", "neu", "pos", "neg", "neu"])
        video = np.array(["a", "a", "b", "c", "c", "d"])
        groups = video_indices(video)
        low, high = cluster_interval(y, y, y, groups, 100, 42)
        self.assertEqual((low, high), (0.0, 0.0))
        self.assertEqual(cluster_swap_p(y, y, y, groups, 100, 42), 1.0)

    def test_fast_macro_matches_fixed_label_sklearn_score(self):
        y = np.array(["pos", "pos", "neg", "neu"])
        pred = np.array(["pos", "neu", "neu", "neu"])
        self.assertAlmostEqual(macro(y, pred), f1_score(
            y, pred, labels=["pos", "neg", "neu"],
            average="macro", zero_division=0))


if __name__ == "__main__":
    unittest.main()
