import csv
import tempfile
import unittest
from pathlib import Path

from src.eval.baseline_report import render
from src.eval.corpus_audit import audit
from src.eval.folds import make_splits
from src.models.run_cv import validate_inputs


def write_csv(path, fields, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


class Sprint3Tests(unittest.TestCase):
    def test_audit_rejects_partial_and_does_not_count_prefilled_density(self):
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp)
            batch = base / "batches"
            manifest = [dict(comment_id=str(i), batch="natural_01", sample="natural")
                        for i in range(4)]
            write_csv(batch / "manifest.csv", ["comment_id", "batch", "sample"], manifest)
            fields = ["comment_id", "text", "sentiment", "density", "base_lang"]
            rows = [dict(comment_id=str(i), text=f"comment {i}",
                         sentiment="pos" if i < 3 else "", density="mono", base_lang="kk")
                    for i in range(4)]
            write_csv(batch / "natural_01.csv", fields, rows)
            write_csv(batch / "retest_01.csv", fields,
                      [dict(comment_id=str(i), text=f"comment {i}", sentiment="",
                            density="mono", base_lang="") for i in range(4)])
            result = audit(batch / "manifest.csv", batch)
            self.assertFalse(result["ready"])
            self.assertEqual(result["unannotated"], 1)
            self.assertNotIn("density", result["intra_annotator"])
            rows[3]["sentiment"] = "neg"
            write_csv(batch / "natural_01.csv", fields, rows)
            self.assertTrue(audit(batch / "manifest.csv", batch)["ready"])

    def test_grouped_folds_cover_rows_without_video_leakage(self):
        rows = [dict(comment_id=str(i), video_id=f"v{i // 5}",
                     sentiment=("pos", "neg", "neu")[i % 3],
                     density=("mono", "low", "med", "high")[i % 4],
                     text=f"comment {i}") for i in range(120)]
        folds, test = make_splits(rows, 5, 0.15, 2026)
        split = dict(folds=folds, test=test, n_labeled=len(rows))
        validate_inputs(rows, split)
        split["test"] = test + [folds[0][0]]
        with self.assertRaises(ValueError):
            validate_inputs(rows, split)

    def test_report_contains_each_confusion_matrix(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "predictions.csv"
            fields = ["comment_id", "gold", "pred", "density", "representation",
                      "classifier", "fold"]
            rows = []
            for rep in ("tfidf_word", "fasttext"):
                for i, label in enumerate(("pos", "neg", "neu")):
                    rows.append(dict(comment_id=str(i), gold=label, pred=label,
                                     density="mono", representation=rep,
                                     classifier="logreg", fold=str(i + 1)))
            write_csv(path, fields, rows)
            report = render(path)
            self.assertIn("| tfidf_word | logreg | 3 | 1.000", report)
            self.assertIn("| fasttext | logreg | 3 | 1.000", report)
            self.assertEqual(report.count("### Confusion matrix"), 2)


if __name__ == "__main__":
    unittest.main()
