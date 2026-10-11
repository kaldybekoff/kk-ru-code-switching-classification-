"""Draw a blind, density-stratified subset for independent human annotation.

The output has empty labels even when pass one is filled. Give this sheet to a
second annotator without showing the original labels, then compare with
``python -m src.eval.corpus_audit --second <completed-sheet>``.
"""

import argparse
import csv
import random
from collections import defaultdict
from pathlib import Path

from src.eval.corpus_audit import BATCHES, MANIFEST, load_sheets, read_csv

OUT = BATCHES.parent / "validation_subset_blind.csv"
BUCKETS = ("mono", "low", "med", "high")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n", type=int, default=100)
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--out", type=Path, default=OUT)
    args = p.parse_args()
    if args.n < 4:
        raise SystemExit("Use at least four rows to cover all density strata")
    _, manifest = read_csv(MANIFEST)
    first, issues = load_sheets(sorted(p for p in BATCHES.glob("*.csv")
                                       if p.stem.startswith(("natural_", "enriched_"))))
    if issues:
        raise SystemExit("Fix batch structure before drawing validation subset")
    by_bucket = defaultdict(list)
    for m in manifest:
        if m["sample"] == "retest":
            continue
        cid = m["comment_id"]
        if cid in first and m["stratum"] in BUCKETS:
            by_bucket[m["stratum"]].append(cid)
    rng = random.Random(args.seed)
    chosen = []
    quota = args.n // len(BUCKETS)
    for bucket in BUCKETS:
        ids = sorted(by_bucket[bucket])
        chosen.extend(rng.sample(ids, min(quota, len(ids))))
    leftovers = sorted(set().union(*map(set, by_bucket.values())) - set(chosen))
    chosen.extend(rng.sample(leftovers, min(args.n - len(chosen), len(leftovers))))
    rng.shuffle(chosen)
    if args.out.exists():
        raise SystemExit(f"Refusing to overwrite existing validation sheet: {args.out}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["comment_id", "text", "sentiment", "density", "base_lang", "notes"])
        for cid in chosen:
            writer.writerow([cid, first[cid][0]["text"], "", "", "", ""])
    print(f"Wrote {len(chosen)} blinded comments to {args.out}")


if __name__ == "__main__":
    main()
