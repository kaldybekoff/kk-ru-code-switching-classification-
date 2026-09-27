"""Cross-validate every representation on the frozen folds and record a
prediction for every comment.

    python -m src.models.run_cv                          # all available reps
    python -m src.models.run_cv --reps tfidf_word fasttext
    python -m src.models.run_cv --clf svm

Writes data/labeled/predictions.csv:

    comment_id, gold, density, n_tokens, representation, classifier, fold, pred

One row per (comment, representation) - the per-item predictions McNemar's test
needs. Aggregate scores alone cannot support a paired test, which is why this
saves predictions rather than just metrics.

Every representation is fitted **inside the fold**, on the training half only.
The one exception is fastText, which is trained on the unlabeled corpus and
never sees a label at all; see src/features/representations.py.
"""

import argparse
import csv
import json
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score
from sklearn.svm import LinearSVC

from src.eval.folds import FOLDS, load_labeled
from src.features.representations import REPRESENTATIONS
from src.preprocess.textnorm import utf8_stdout

ROOT = Path(__file__).resolve().parents[2]
PRED = ROOT / "data" / "labeled" / "predictions.csv"

CLASSIFIERS = {
    # class_weight balanced throughout: `neu` dominates, and an unweighted
    # model reaches a respectable accuracy by predicting it almost always.
    # Macro-F1 is the reported metric for the same reason.
    "logreg": lambda: LogisticRegression(
        max_iter=2000, class_weight="balanced", C=1.0, random_state=2026),
    "svm": lambda: LinearSVC(class_weight="balanced", C=1.0, random_state=2026),
}


def run_one(rep_name, clf_name, rows, folds, verbose=True):
    by_id = {r["comment_id"]: r for r in rows}
    preds = []
    t0 = time.time()

    rep_proto = REPRESENTATIONS[rep_name]()
    # Corpus-level representations are fitted once; fold-level ones are rebuilt
    # per fold so no test text touches the vocabulary.
    corpus_level = rep_name in ("fasttext", "xlmr")
    if corpus_level:
        rep_proto.fit()

    for k, fold in enumerate(folds):
        test_ids = [i for i in fold if i in by_id]
        train_ids = [i for j, f in enumerate(folds) if j != k
                     for i in f if i in by_id]
        if not test_ids or not train_ids:
            continue

        Xtr_txt = [by_id[i]["text"] for i in train_ids]
        Xte_txt = [by_id[i]["text"] for i in test_ids]
        ytr = [by_id[i]["sentiment"] for i in train_ids]

        rep = rep_proto if corpus_level else REPRESENTATIONS[rep_name]().fit(Xtr_txt)
        Xtr, Xte = rep.transform(Xtr_txt), rep.transform(Xte_txt)

        clf = CLASSIFIERS[clf_name]()
        clf.fit(Xtr, ytr)
        yhat = clf.predict(Xte)

        for cid, p in zip(test_ids, yhat):
            r = by_id[cid]
            preds.append({
                "comment_id": cid, "gold": r["sentiment"],
                "density": r["density"], "n_tokens": r["n_tokens"],
                "representation": rep_name, "classifier": clf_name,
                "fold": k + 1, "pred": p,
            })
        if verbose:
            print("    fold {}  train {:>5}  test {:>4}".format(
                k + 1, len(train_ids), len(test_ids)))

    gold = [p["gold"] for p in preds]
    yhat = [p["pred"] for p in preds]
    macro = f1_score(gold, yhat, average="macro", zero_division=0)
    if verbose:
        print("  {} + {}: macro-F1 {:.3f}   ({:.0f}s)".format(
            rep_name, clf_name, macro, time.time() - t0))
    return preds, macro


def main():
    utf8_stdout()
    p = argparse.ArgumentParser()
    p.add_argument("--reps", nargs="+", default=list(REPRESENTATIONS),
                   choices=list(REPRESENTATIONS))
    p.add_argument("--clf", nargs="+", default=["logreg"],
                   choices=list(CLASSIFIERS))
    p.add_argument("--labeled")
    p.add_argument("--folds", default=str(FOLDS))
    p.add_argument("--out", default=str(PRED))
    p.add_argument("--skip-missing", action="store_true", default=True,
                   help="carry on when a representation's dependency is absent")
    args = p.parse_args()

    rows = load_labeled(Path(args.labeled)) if args.labeled else load_labeled()
    fpath = Path(args.folds)
    if not fpath.exists():
        raise SystemExit(
            "{} not found.\nRun: python -m src.eval.folds".format(fpath))
    folds = json.loads(fpath.read_text(encoding="utf-8"))["folds"]

    print("{} labeled comments, {} folds".format(len(rows), len(folds)))
    print("label distribution: {}".format(
        dict(Counter(r["sentiment"] for r in rows))))

    all_preds, scores, skipped = [], {}, []
    for rep in args.reps:
        for clf in args.clf:
            print("\n{} + {}".format(rep, clf))
            try:
                preds, macro = run_one(rep, clf, rows, folds)
            except SystemExit as e:
                print("  skipped: {}".format(str(e).splitlines()[0]))
                skipped.append(rep)
                continue
            all_preds += preds
            scores["{}+{}".format(rep, clf)] = macro

    if not all_preds:
        raise SystemExit("\nNothing ran. Install the missing dependencies above.")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=[
            "comment_id", "gold", "density", "n_tokens",
            "representation", "classifier", "fold", "pred"])
        w.writeheader()
        w.writerows(all_preds)

    print("\n=== macro-F1 (5-fold CV) ===")
    for k, v in sorted(scores.items(), key=lambda kv: -kv[1]):
        print("  {:<24} {:.3f}".format(k, v))
    if skipped:
        print("\nnot run (missing dependencies): {}".format(", ".join(skipped)))
    print("\n-> {}".format(out))
    print("Next: python -m src.eval.compare")


if __name__ == "__main__":
    main()
