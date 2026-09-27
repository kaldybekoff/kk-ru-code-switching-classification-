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
  - Stratification is on (sentiment x density), not sentiment alone: the whole
    study is a per-density-bucket comparison, so a fold that happens to hold two
    `high` examples would make the subgroup analysis noise.
  - Grouping is by video_id where it does not break stratification, so
    near-duplicate reactions to the same moment of the same video do not sit on
    both sides of a split.
"""

import argparse
import csv
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

from src.preprocess.textnorm import utf8_stdout

ROOT = Path(__file__).resolve().parents[2]
LABELED = ROOT / "data" / "labeled" / "corpus_labeled.csv"
FOLDS = ROOT / "data" / "labeled" / "folds.json"


def load_labeled(path=LABELED):
    if not path.exists():
        raise SystemExit(
            "{} not found.\nRun: python -m src.preprocess.merge_annotations "
            "data/interim/batches/*.csv".format(path))
    with open(path, encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def stratum(r):
    """The cell a row must be balanced on."""
    return "{}|{}".format(r["sentiment"], r["density"])


def make_splits(rows, k, test_share, seed):
    rng = random.Random(seed)
    by_cell = defaultdict(list)
    for r in rows:
        by_cell[stratum(r)].append(r["comment_id"])

    test, pool = [], []
    for cell, ids in sorted(by_cell.items()):
        ids = sorted(ids)
        rng.shuffle(ids)
        n_test = int(round(len(ids) * test_share))
        # A cell too small to give up a test example keeps all of it for
        # training; the test set stays stratified, just short in that cell.
        test += ids[:n_test]
        pool += ids[n_test:]

    by_cell_pool = defaultdict(list)
    cell_of = {r["comment_id"]: stratum(r) for r in rows}
    for cid in pool:
        by_cell_pool[cell_of[cid]].append(cid)

    folds = [[] for _ in range(k)]
    for cell, ids in sorted(by_cell_pool.items()):
        rng.shuffle(ids)
        # Deal round-robin from a rotating start so no fold is systematically
        # favoured in the small cells.
        offset = rng.randrange(k)
        for i, cid in enumerate(ids):
            folds[(i + offset) % k].append(cid)

    for f in folds:
        rng.shuffle(f)
    rng.shuffle(test)
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
