"""Freeze word counts used for subgroup analysis, independently of sentiment labels."""

import csv
from pathlib import Path

from src.preprocess.density import Density, density_bucket
from src.preprocess.textnorm import MIN_DENSITY_TOKENS

DATA = Path(__file__).resolve().parents[2] / "data" / "labeled"
COUNTS = DATA / "density_word_counts.csv"


def load_groups(path):
    groups = {}
    with open(path, encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            cid = row["comment_id"]
            kk, ru = int(row["kk_tokens"]), int(row["ru_tokens"])
            if not cid or cid in groups or min(kk, ru) < 0:
                raise ValueError(f"Invalid word counts for {cid!r}")
            groups[cid] = density_bucket(kk, ru) if kk + ru >= MIN_DENSITY_TOKENS else ""
    return groups


def main():
    model = Density.load()
    with open(DATA / "corpus_silver.csv", encoding="utf-8-sig", newline="") as handle:
        rows = []
        for row in csv.DictReader(handle):
            counts = model.measure(row["text"])
            rows.append((row["comment_id"], counts["kk"], counts["ru"]))
    if not rows or len({cid for cid, _, _ in rows}) != len(rows):
        raise ValueError("Silver corpus must contain unique comments")
    with COUNTS.with_suffix(".tmp").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("comment_id", "kk_tokens", "ru_tokens"))
        writer.writerows(rows)
    COUNTS.with_suffix(".tmp").replace(COUNTS)
    print(f"{len(rows)} comments -> {COUNTS}")


if __name__ == "__main__":
    main()
