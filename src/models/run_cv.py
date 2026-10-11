"""Cross-validate every representation on the frozen folds and record a
prediction for every comment.

    python -m src.models.run_cv                          # all available reps
    python -m src.models.run_cv --reps tfidf_word fasttext
    python -m src.models.run_cv --clf svm

Writes data/labeled/predictions.csv:

    comment_id, video_id, gold, density, n_tokens, representation, classifier, fold, pred

One row per (comment, representation, classifier) - the per-item predictions McNemar's test
needs. Aggregate scores alone cannot support a paired test, which is why this
saves predictions rather than just metrics.

TF-IDF fits on the fold's training text and XLM-R uses fixed pretrained weights.
With --fasttext-training-videos, fastText fits on the unsampled comments from the
fold's training videos. The default fastText option reproduces historical runs.
"""

import argparse
import csv
import json
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
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

BASELINE_REPS = ["tfidf_word", "tfidf_char", "fasttext"]


def validate_inputs(rows, split):
    """Fail closed on incomplete corpora and stale or overlapping split files."""
    ids = [r["comment_id"] for r in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate comment_id in labeled corpus")
    if not rows or any(r.get("sentiment") not in ("pos", "neg", "neu") or
                       r.get("density") not in ("mono", "low", "med", "high") or
                       not r.get("video_id", "").strip() or
                       not r.get("text", "").strip() for r in rows):
        raise ValueError("Corpus contains missing or invalid video/text/sentiment/density")
    parts = split["folds"] + [split["test"]]
    flat = [cid for part in parts for cid in part]
    if len(parts) < 3 or len(flat) != len(set(flat)) or set(flat) != set(ids):
        raise ValueError("Frozen folds/test overlap or do not exactly cover corpus IDs")
    if not split["test"] or any(not fold for fold in split["folds"]):
        raise ValueError("Frozen folds and held-out test must all be nonempty")
    if split.get("n_labeled") != len(rows):
        raise ValueError("Frozen folds were generated for a different corpus size")
    by_id = {r["comment_id"]: r for r in rows}
    video_part = {}
    for part_number, part in enumerate(parts):
        for cid in part:
            video = by_id[cid].get("video_id") or cid
            if video in video_part and video_part[video] != part_number:
                raise ValueError("A video_id crosses frozen CV/test partitions")
            video_part[video] = part_number


def run_grid(rep_name, clf_names, rows, folds, verbose=True, fasttext_training_videos=False):
    by_id = {r["comment_id"]: r for r in rows}
    predictions = {name: [] for name in clf_names}
    t0 = time.time()

    rep_proto = REPRESENTATIONS[rep_name]()
    # Corpus-level representations are fitted once; fold-level ones are rebuilt
    # per fold so no test text touches the vocabulary.
    grouped_fasttext = rep_name == "fasttext" and fasttext_training_videos
    corpus_level = rep_name == "xlmr" or (rep_name == "fasttext" and not grouped_fasttext)
    if corpus_level:
        rep_proto.fit()
        # Frozen unsupervised features can be encoded once for the CV pool.
        # The held-out test remains excluded, and no labels enter this step.
        pool_ids = list(dict.fromkeys(cid for fold in folds for cid in fold))
        pool_features = rep_proto.transform([by_id[cid]["text"] for cid in pool_ids])
        feature_by_id = dict(zip(pool_ids, pool_features))

    for k, fold in enumerate(folds):
        test_ids = [i for i in fold if i in by_id]
        train_ids = [i for j, f in enumerate(folds) if j != k
                     for i in f if i in by_id]
        if not test_ids or not train_ids:
            continue

        Xtr_txt = [by_id[i]["text"] for i in train_ids]
        Xte_txt = [by_id[i]["text"] for i in test_ids]
        ytr = [by_id[i]["sentiment"] for i in train_ids]

        if corpus_level:
            Xtr = np.stack([feature_by_id[cid] for cid in train_ids])
            Xte = np.stack([feature_by_id[cid] for cid in test_ids])
        else:
            if grouped_fasttext:
                videos = {by_id[i]["video_id"] for i in train_ids}
                rep = REPRESENTATIONS[rep_name](training_videos=videos)
            else:
                rep = REPRESENTATIONS[rep_name]()
            rep.fit(Xtr_txt)
            Xtr, Xte = rep.transform(Xtr_txt), rep.transform(Xte_txt)
            del rep

        for clf_name in clf_names:
            clf = CLASSIFIERS[clf_name]()
            clf.fit(Xtr, ytr)
            yhat = clf.predict(Xte)
            for cid, p in zip(test_ids, yhat):
                r = by_id[cid]
                predictions[clf_name].append({
                    "comment_id": cid, "video_id": r.get("video_id", ""),
                    "gold": r["sentiment"], "density": r["density"],
                    "n_tokens": r["n_tokens"], "representation": rep_name,
                    "classifier": clf_name, "fold": k + 1, "pred": p,
                })
        if verbose:
            print("    fold {}  train {:>5}  test {:>4}".format(
                k + 1, len(train_ids), len(test_ids)))

    scores = {}
    for clf_name, preds in predictions.items():
        scores[clf_name] = f1_score(
            [p["gold"] for p in preds], [p["pred"] for p in preds],
            labels=["pos", "neg", "neu"], average="macro", zero_division=0)
        if verbose:
            print("  {} + {}: macro-F1 {:.3f}   ({:.0f}s)".format(
                rep_name, clf_name, scores[clf_name], time.time() - t0))
    return predictions, scores


def run_one(rep_name, clf_name, rows, folds, verbose=True):
    predictions, scores = run_grid(rep_name, [clf_name], rows, folds, verbose)
    return predictions[clf_name], scores[clf_name]


def main():
    utf8_stdout()
    p = argparse.ArgumentParser()
    p.add_argument("--reps", nargs="+", default=BASELINE_REPS,
                   choices=list(REPRESENTATIONS))
    p.add_argument("--clf", nargs="+", default=["logreg"],
                   choices=list(CLASSIFIERS))
    p.add_argument("--labeled")
    p.add_argument("--folds", default=str(FOLDS))
    p.add_argument("--out", default=str(PRED))
    p.add_argument("--fasttext-training-videos", action="store_true",
                   help="fit fastText on unsampled comments from this fold's training videos")
    p.add_argument("--skip-missing", action="store_true",
                   help="carry on when a representation's dependency is absent")
    args = p.parse_args()

    rows = load_labeled(Path(args.labeled)) if args.labeled else load_labeled()
    fpath = Path(args.folds)
    if not fpath.exists():
        raise SystemExit(
            "{} not found.\nRun: python -m src.eval.folds".format(fpath))
    split = json.loads(fpath.read_text(encoding="utf-8"))
    validate_inputs(rows, split)
    folds = split["folds"]

    print("{} labeled comments, {} folds".format(len(rows), len(folds)))
    print("label distribution: {}".format(
        dict(Counter(r["sentiment"] for r in rows))))

    all_preds, scores, skipped = [], {}, []
    for rep in args.reps:
        print("\n{} + {}".format(rep, ", ".join(args.clf)))
        try:
            rep_preds, rep_scores = run_grid(rep, args.clf, rows, folds,
                                             fasttext_training_videos=args.fasttext_training_videos)
        except SystemExit as e:
            if not args.skip_missing:
                raise
            print("  skipped: {}".format(str(e).splitlines()[0]))
            skipped.append(rep)
            continue
        for clf in args.clf:
            all_preds += rep_preds[clf]
            scores["{}+{}".format(rep, clf)] = rep_scores[clf]

    if not all_preds:
        raise SystemExit("\nNothing ran. Install the missing dependencies above.")
    expected = sum(len(fold) for fold in folds) * len(scores)
    if len(all_preds) != expected:
        raise SystemExit("Incomplete CV predictions: {} of {}".format(
            len(all_preds), expected))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=[
            "comment_id", "video_id", "gold", "density", "n_tokens",
            "representation", "classifier", "fold", "pred"])
        w.writeheader()
        w.writerows(all_preds)

    print("\n=== macro-F1 (5-fold CV) ===")
    for k, v in sorted(scores.items(), key=lambda kv: -kv[1]):
        print("  {:<24} {:.3f}".format(k, v))
    if skipped:
        print("\nnot run (missing dependencies): {}".format(", ".join(skipped)))
    print("\n-> {}".format(out))
    print("Use src.eval.sprint5 to compare the complete week 10 grid.")


if __name__ == "__main__":
    main()
