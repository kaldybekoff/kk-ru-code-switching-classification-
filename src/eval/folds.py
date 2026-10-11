"""Freeze the cross-validation splits once, and never regenerate them.

    python -m src.eval.folds                    # create data/labeled/folds.json
    python -m src.eval.folds --show             # describe the existing splits

Why frozen: every representation (TF-IDF, fastText, XLM-R) and every
preprocessing variant must be scored on exactly the same partitions, or the
comparison measures split luck instead of representation quality. McNemar's
test in particular is a *paired* test - it compares two models' predictions on
the same items - so it is meaningless if the two models saw different folds.

The file is written once and committed. `--force` exists but rewriting it after
any result has been recorded invalidates every number already reported.

Design:
  - A held-out test set is split off first and never touched during development.
  - The remaining data is cut into stratified k folds for cross-validation.
  - Stratification targets (sentiment x density), not sentiment alone: the
    study compares performance by density bucket. Exact cell counts are not
    always possible when all comments from a video stay together.
  - Grouping by video_id is strict, so reactions to the same video cannot sit
    on opposite sides of a split. The test split is selected from 200 seeded
    group candidates to approximate the target row share and cell balance.
"""

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold

from src.preprocess.textnorm import utf8_stdout

ROOT = Path(__file__).resolve().parents[2]
LABELED = ROOT / "data" / "labeled" / "corpus_labeled.csv"
FOLDS = ROOT / "data" / "labeled" / "folds.json"


def load_labeled(path=LABELED):
    if not path.exists():
        raise SystemExit(
            "{} not found.\nRun: python -m src.preprocess.merge_annotations "
            "data/interim/batches/*.csv".format(path))
    with open(path, encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def stratum(r):
    """The cell a row must be balanced on."""
    return "{}|{}".format(r["sentiment"], r["density"])


def make_splits(rows, k, test_share, seed):
    if not 0 < test_share < 1 or k < 2:
        raise ValueError("test_share must be in (0, 1), and k >= 2")
    if len({r.get("video_id") or r["comment_id"] for r in rows}) < k + 1:
        raise ValueError("Not enough video groups for a grouped test split and CV")
    labels = [stratum(r) for r in rows]
    groups = [r.get("video_id") or r["comment_id"] for r in rows]
    counts = Counter(labels)

    # GroupShuffleSplit is repeated with a fixed seed; choose the candidate
    # closest to the requested row share and joint sentiment×density mix.
    chooser = GroupShuffleSplit(n_splits=200, test_size=test_share, random_state=seed)
    best = None
    for train_idx, test_idx in chooser.split(rows, labels, groups):
        test_counts = Counter(labels[i] for i in test_idx)
        score = abs(len(test_idx) / len(rows) - test_share)
        score += sum(abs(test_counts[cell] / counts[cell] - test_share)
                     for cell in counts) / len(counts)
        if best is None or score < best[0]:
            best = (score, train_idx, test_idx)
    _, train_idx, test_idx = best
    pool = [rows[i] for i in train_idx]
    pool_labels = [stratum(r) for r in pool]
    pool_groups = [r.get("video_id") or r["comment_id"] for r in pool]
    if len(set(pool_groups)) < k:
        raise ValueError("Not enough training video groups for CV")
    cv = StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=seed)
    folds = [[pool[i]["comment_id"] for i in valid_idx]
             for _, valid_idx in cv.split(pool, pool_labels, pool_groups)]
    test = [rows[i]["comment_id"] for i in test_idx]
    all_ids = [cid for fold in folds for cid in fold] + test
    if any(not fold for fold in folds) or not test or len(all_ids) != len(set(all_ids)):
        raise ValueError("Grouped splitting produced an empty or overlapping partition")
    if set(all_ids) != {r["comment_id"] for r in rows}:
        raise ValueError("Grouped splitting did not cover every comment")
    return folds, test


def describe(rows, folds, test):
    by_id = {r["comment_id"]: r for r in rows}

    def counts(ids, key):
        return Counter(by_id[i][key] for i in ids if i in by_id)

    print("\n{:<10}{:>7}   {:<34}{}".format("split", "n", "sentiment", "density"))
    print("-" * 86)

    def line(name, ids):
        cs = counts(ids, "sentiment")
        cd = counts(ids, "density")
        s = " ".join("{} {}".format(k, cs[k]) for k in ("pos", "neg", "neu"))
        d = " ".join("{} {}".format(k, cd[k]) for k in ("mono", "low", "med", "high"))
        print("{:<10}{:>7}   {:<34}{}".format(name, len(ids), s, d))

    for i, f in enumerate(folds, 1):
        line("fold {}".format(i), f)
    line("TEST", test)
    print("-" * 86)
    line("total", [i for f in folds for i in f] + test)

    # The number that actually limits the study.
    cd = counts([i for f in folds for i in f], "density")
    print("\nper-fold density counts (the subgroup analysis lives here):")
    for d in ("mono", "low", "med", "high"):
        per = [sum(1 for i in f if by_id.get(i, {}).get("density") == d) for f in folds]
        flag = "  <-- thin" if min(per) < 20 else ""
        print("  {:<6} {}  (min {}){}".format(d, per, min(per), flag))


def main():
    utf8_stdout()
    p = argparse.ArgumentParser()
    p.add_argument("--k", type=int, default=5, help="number of CV folds")
    p.add_argument("--test-share", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--labeled", default=str(LABELED))
    p.add_argument("--out", default=str(FOLDS))
    p.add_argument("--show", action="store_true", help="describe existing splits")
    p.add_argument("--force", action="store_true",
                   help="overwrite existing folds - invalidates every result "
                        "already reported against them")
    args = p.parse_args()

    rows = load_labeled(Path(args.labeled))
    out = Path(args.out)

    if args.show:
        if not out.exists():
            raise SystemExit("{} does not exist yet.".format(out))
        d = json.loads(out.read_text(encoding="utf-8"))
        print("{}\nseed {}  k {}  test_share {}".format(
            out, d["seed"], d["k"], d["test_share"]))
        describe(rows, d["folds"], d["test"])
        return

    if out.exists() and not args.force:
        raise SystemExit(
            "{} already exists.\nEvery reported result is tied to it - pass "
            "--force only if nothing has been reported yet.".format(out))

    folds, test = make_splits(rows, args.k, args.test_share, args.seed)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "seed": args.seed, "k": args.k, "test_share": args.test_share,
        "n_labeled": len(rows), "folds": folds, "test": test,
    }, indent=1), encoding="utf-8")

    describe(rows, folds, test)
    print("\n-> {}".format(out))
    print("Commit this file. Do not regenerate it.")


if __name__ == "__main__":
    main()
