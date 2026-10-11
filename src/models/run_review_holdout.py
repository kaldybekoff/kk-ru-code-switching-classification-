"""Train Sprint 3 baselines on silver labels; test on blind Codex AI review.

Train and test are disjoint by video_id. The test labels are from another AI
reviewer, so these results are exploratory, not gold human performance.
"""

import argparse
import csv
from pathlib import Path

from src.features.representations import REPRESENTATIONS
from src.models.run_cv import BASELINE_REPS, CLASSIFIERS

ROOT = Path(__file__).resolve().parents[2]
TRAIN = ROOT / "data" / "labeled" / "corpus_silver_train.csv"
TEST = ROOT / "data" / "labeled" / "corpus_ai_review_test.csv"
OUT = ROOT / "data" / "labeled" / "predictions_ai_review.csv"


def load(path):
    with open(path, encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--train", type=Path, default=TRAIN)
    p.add_argument("--test", type=Path, default=TEST)
    p.add_argument("--out", type=Path, default=OUT)
    p.add_argument("--fasttext-training-videos", action="store_true",
                   help="restrict fastText pretraining to videos in the supervised training set")
    p.add_argument("--reps", nargs="+", default=BASELINE_REPS,
                   choices=list(REPRESENTATIONS))
    p.add_argument("--clf", nargs="+", default=["logreg"],
                   choices=list(CLASSIFIERS))
    args = p.parse_args()
    train, test = load(args.train), load(args.test)
    if not train or not test:
        raise SystemExit("Train and review test sets must both be nonempty")
    if {r["comment_id"] for r in train} & {r["comment_id"] for r in test}:
        raise SystemExit("Train and review test contain overlapping comment IDs")
    if {r["video_id"] for r in train} & {r["video_id"] for r in test}:
        raise SystemExit("Train and review test contain overlapping videos")
    if {r["sentiment"] for r in train} != {"pos", "neg", "neu"}:
        raise SystemExit("Silver training corpus must contain all three classes")
    for name, rows in (("train", train), ("test", test)):
        if len({r["comment_id"] for r in rows}) != len(rows):
            raise SystemExit(f"Duplicate comment IDs in {name}")
        if any(not r.get("video_id", "").strip() or not r.get("text", "").strip() or
               r.get("sentiment") not in ("pos", "neg", "neu") or
               r.get("density") not in ("mono", "low", "med", "high")
               for r in rows):
            raise SystemExit(f"Invalid text, sentiment, or density in {name}")
    train_text = [r["text"] for r in train]
    test_text = [r["text"] for r in test]
    records = []
    for rep_name in args.reps:
        options = {}
        if rep_name == "fasttext" and args.fasttext_training_videos:
            options["training_videos"] = {r["video_id"] for r in train}
        representation = REPRESENTATIONS[rep_name](**options)
        representation.fit(None if rep_name in ("fasttext", "xlmr") else train_text)
        X_train = representation.transform(train_text)
        X_test = representation.transform(test_text)
        for clf_name in args.clf:
            clf = CLASSIFIERS[clf_name]()
            clf.fit(X_train, [r["sentiment"] for r in train])
            predictions = clf.predict(X_test)
            for row, prediction in zip(test, predictions):
                records.append({"comment_id": row["comment_id"],
                                "video_id": row["video_id"],
                                "gold": row["sentiment"], "pred": prediction,
                                "density": row["density"],
                                "n_tokens": row["n_tokens"],
                                "representation": rep_name,
                                "classifier": clf_name,
                                "fold": "blind_ai_review"})
            print(f"{rep_name} + {clf_name}: predicted {len(test)} reviewed comments")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["comment_id", "video_id", "gold", "density",
                                               "n_tokens", "representation", "classifier",
                                               "fold", "pred"])
        writer.writeheader()
        writer.writerows(records)
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
