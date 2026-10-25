"""Audit the Sprint 4 corpus and XLM-R tokenization before model evaluation."""

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np

from src.features.representations import CACHE, XLMR_MODEL, XLMR_REVISION
from src.models.run_cv import validate_inputs

ROOT = Path(__file__).resolve().parents[2]
TRAIN = ROOT / "data/labeled/corpus_silver_train.csv"
TEST = ROOT / "data/labeled/corpus_ai_review_test.csv"
FOLDS = ROOT / "data/labeled/folds_silver.json"
OUT = ROOT / "reports/sprint4_corpus_audit.md"


def read_rows(path):
    with open(path, encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def audit(train, test, split, tokenizer, max_len=256):
    validate_inputs(train, split)
    if not test or len({r["comment_id"] for r in test}) != len(test):
        raise ValueError("Review test is empty or contains duplicate IDs")
    if {r["comment_id"] for r in train} & {r["comment_id"] for r in test}:
        raise ValueError("Train and review test overlap by comment ID")
    if {r["video_id"] for r in train} & {r["video_id"] for r in test}:
        raise ValueError("Train and review test overlap by video ID")
    for row in test:
        if (not row.get("text", "").strip() or
                row.get("sentiment") not in ("pos", "neg", "neu") or
                row.get("density") not in ("mono", "low", "med", "high")):
            raise ValueError("Invalid review test text or label")
    lines = ["# Sprint 4 transformer corpus audit", "",
             f"Encoder: `{XLMR_MODEL}@{XLMR_REVISION}`; max sequence length: {max_len} tokens.",
             "The tokenizer receives the same cleaned comment text used by all baselines. "
             "Padding is per batch; truncation is at the stated maximum.", "",
             "| Partition | n | label counts (pos/neg/neu) | density counts (mono/low/med/high) | "
             "median tokens | p95 tokens | truncated |", "|---|---:|---|---|---:|---:|---:|"]
    for name, rows in (("Silver training corpus", train),
                       ("Blind AI review", test)):
        lengths = [len(ids) for ids in tokenizer(
            [r["text"] for r in rows], add_special_tokens=True,
            truncation=False)["input_ids"]]
        labels = Counter(r["sentiment"] for r in rows)
        density = Counter(r["density"] for r in rows)
        lines.append(
            f"| {name} | {len(rows)} | "
            f"{'/'.join(str(labels[k]) for k in ('pos', 'neg', 'neu'))} | "
            f"{'/'.join(str(density[k]) for k in ('mono', 'low', 'med', 'high'))} | "
            f"{np.median(lengths):.0f} | {np.percentile(lengths, 95):.0f} | "
            f"{sum(n > max_len for n in lengths)} |")
    lines += ["", f"Frozen split: {len(split['folds'])} video-grouped CV folds, "
              f"{sum(map(len, split['folds']))} out-of-fold predictions, "
              f"{len(split['test'])} reserved silver-label test rows.",
              "The blind AI review is video-disjoint from the silver training corpus.",
              "", "Silver sentiment labels were assigned by a sentiment-tuned XLM-R "
              "teacher. The pretrained encoder evaluated here shares an architecture "
              "family with that teacher, so agreement with silver labels can favor XLM-R. "
              "The blind AI review is exploratory and is not human gold."]
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, default=TRAIN)
    parser.add_argument("--test", type=Path, default=TEST)
    parser.add_argument("--folds", type=Path, default=FOLDS)
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--max-len", type=int, default=256)
    args = parser.parse_args()
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        XLMR_MODEL, revision=XLMR_REVISION, cache_dir=str(CACHE / "hf"))
    report = audit(read_rows(args.train), read_rows(args.test),
                   json.loads(args.folds.read_text(encoding="utf-8")),
                   tokenizer, args.max_len)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
