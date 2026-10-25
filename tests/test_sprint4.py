import unittest
from unittest.mock import patch

import numpy as np

from src.models.run_cv import run_grid, run_one


class FrozenFeatureTests(unittest.TestCase):
    def test_cv_encodes_only_pool_once_and_predicts_each_fold(self):
        instances = []

        class FrozenFeature:
            def __init__(self):
                self.calls = []
                instances.append(self)

            def fit(self, texts=None):
                self.calls.append(("fit", texts))
                return self

            def transform(self, texts):
                self.calls.append(("transform", tuple(texts)))
                return np.array([[float(text.split()[-1]), 1.0]
                                 for text in texts], dtype=np.float32)

        rows = [dict(comment_id=str(i), text=f"comment {i}",
                     sentiment=("pos", "neg", "neu")[i % 3],
                     density="mono", n_tokens="2") for i in range(7)]
        folds = [["0", "1", "2"], ["3", "4", "5"]]
        with patch.dict("src.models.run_cv.REPRESENTATIONS", {"xlmr": FrozenFeature}):
            predictions, _ = run_one("xlmr", "logreg", rows, folds, verbose=False)

        self.assertEqual(len(instances), 1)
        self.assertEqual(instances[0].calls[0], ("fit", None))
        self.assertEqual(instances[0].calls[1],
                         ("transform", tuple(f"comment {i}" for i in range(6))))
        self.assertEqual(len(instances[0].calls), 2)
        self.assertEqual({p["comment_id"] for p in predictions},
                         {str(i) for i in range(6)})

    def test_classifier_grid_reuses_frozen_features(self):
        instances = []

        class FrozenFeature:
            def __init__(self):
                self.calls = 0
                instances.append(self)

            def fit(self, texts=None):
                return self

            def transform(self, texts):
                self.calls += 1
                return np.array([[float(text.split()[-1]), 1.0]
                                 for text in texts], dtype=np.float32)

        rows = [dict(comment_id=str(i), text=f"comment {i}",
                     sentiment=("pos", "neg", "neu")[i % 3],
                     density="mono", n_tokens="2") for i in range(9)]
        folds = [[str(i) for i in range(3)],
                 [str(i) for i in range(3, 6)],
                 [str(i) for i in range(6, 9)]]
        with patch.dict("src.models.run_cv.REPRESENTATIONS", {"xlmr": FrozenFeature}):
            predictions, _ = run_grid("xlmr", ["logreg", "svm"], rows, folds,
                                      verbose=False)
        self.assertEqual(len(instances), 1)
        self.assertEqual(instances[0].calls, 1)
        self.assertEqual({key: len(value) for key, value in predictions.items()},
                         {"logreg": 9, "svm": 9})


if __name__ == "__main__":
    unittest.main()
