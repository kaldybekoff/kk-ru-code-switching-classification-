"""Report Sprint 3 metrics from CV or a blind AI review comparison."""

import argparse
import csv
from collections import defaultdict
from pathlib import Path

from sklearn.metrics import (accuracy_score, confusion_matrix,
                             precision_recall_fscore_support)
from src.eval.density_groups import COUNTS, load_groups

ROOT = Path(__file__).resolve().parents[2]
PRED = ROOT / "data" / "labeled" / "predictions.csv"
OUT = ROOT / "reports" / "sprint3_baselines.md"
LABELS = ("pos", "neg", "neu")
DENSITY = ("mono", "low", "med", "high")


def scores(rows):
    gold = [r["gold"] for r in rows]
    pred = [r["pred"] for r in rows]
    p, recall, f1, support = precision_recall_fscore_support(
        gold, pred, labels=LABELS, zero_division=0)
    return {"accuracy": accuracy_score(gold, pred),
            "macro_p": sum(p) / 3, "macro_r": sum(recall) / 3,
            "macro_f1": sum(f1) / 3,
            "per_class": list(zip(LABELS, p, recall, f1, support)),
            "confusion": confusion_matrix(gold, pred, labels=LABELS)}


def render(path, protocol="cv", title="Sprint 3 baseline results", counts_path=None):
    with open(path, encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError("No predictions")
    if counts_path is None and Path(path).name in (
            "predictions_sprint5_grouped_ai_review.csv", "predictions_sprint5_grouped_silver_cv.csv"):
        counts_path = COUNTS
    density_groups = load_groups(counts_path) if counts_path is not None else None
    required = {"comment_id", "gold", "pred", "density", "representation",
                "classifier", "fold"}
    if not required <= set(rows[0]):
        raise ValueError(f"Prediction CSV missing {required - set(rows[0])}")
    groups = defaultdict(dict)
    for row in rows:
        if row["gold"] not in LABELS or row["pred"] not in LABELS or row["density"] not in DENSITY:
            raise ValueError("Invalid label in predictions")
        key = (row["representation"], row["classifier"])
        cid = row["comment_id"]
        if cid in groups[key]:
            raise ValueError(f"Duplicate prediction: {key} {cid}")
        groups[key][cid] = row
    reference = None
    for key, items in groups.items():
        signature = {cid: (r["gold"], r["density"], r["fold"])
                     for cid, r in items.items()}
        if reference is None:
            reference = signature
        elif signature != reference:
            raise ValueError(f"Different items, labels or folds for {key}")
    if protocol == "cv":
        scope = (f"Out-of-fold predictions: **{len(reference)} comments**. Frozen "
                 "held-out test items are excluded.")
    elif protocol == "blind_ai_review":
        scope = (f"Video-disjoint blind AI review: **{len(reference)} comments**. "
                 "Training sentiment labels came from a local AI model; test labels "
                 "came from Codex without seeing those predictions. This is not "
                 "human-gold accuracy or human inter-annotator agreement. "
                 "An alternate annotation model was inspected on this review "
                 "before final model selection, so these scores are exploratory.")
    else:
        raise ValueError(f"Unknown protocol: {protocol}")
    lines = [f"# {title}", "",
             scope + " The corpus is density-enriched; these scores describe this "
             "sample, not natural YouTube prevalence.", "",
             "| Representation | Classifier | n | Accuracy | Macro precision | Macro recall | Macro F1 |",
             "|---|---|---:|---:|---:|---:|---:|"]
    for key in sorted(groups):
        rep, clf = key
        items = [groups[key][cid] for cid in sorted(groups[key])]
        metric = scores(items)
        lines.append(f"| {rep} | {clf} | {len(items)} | {metric['accuracy']:.3f} | "
                     f"{metric['macro_p']:.3f} | {metric['macro_r']:.3f} | {metric['macro_f1']:.3f} |")
    for key in sorted(groups):
        rep, clf = key
        items = list(groups[key].values())
        metric = scores(items)
        lines += ["", f"## {rep} + {clf}", "", "### Per-class metrics", "",
                  "| Class | Support | Precision | Recall | F1 |",
                  "|---|---:|---:|---:|---:|"]
        for label, p, recall, f1, n in metric["per_class"]:
            lines.append(f"| {label} | {n} | {p:.3f} | {recall:.3f} | {f1:.3f} |")
        lines += ["", "### Confusion matrix", "", "Rows are reference labels; columns are predictions.", "",
                  "| Reference \\ predicted | pos | neg | neu |", "|---|---:|---:|---:|"]
        for label, values in zip(LABELS, metric["confusion"]):
            lines.append(f"| {label} | " + " | ".join(str(int(x)) for x in values) + " |")
        lines += ["", "### By density", ""]
        if density_groups is not None:
            lines += ["Groups use the deterministic Kazakh and Russian word counts. "
                      "At least five counted words are required.", ""]
        lines += [
                  "| Density | n | Accuracy | Macro F1 |", "|---|---:|---:|---:|"]
        for density in DENSITY:
            subset = [r for r in items if (density_groups[r["comment_id"]]
                      if density_groups is not None else r["density"]) == density]
            if not subset:
                lines.append(f"| {density} | 0 | n/a | n/a |")
                continue
            s = scores(subset)
            lines.append(f"| {density} | {len(subset)} | {s['accuracy']:.3f} | {s['macro_f1']:.3f} |")
    return "\n".join(lines) + "\n"


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pred", type=Path, default=PRED)
    p.add_argument("--out", type=Path, default=OUT)
    p.add_argument("--protocol", choices=["cv", "blind_ai_review"], default="cv")
    p.add_argument("--title", default="Sprint 3 baseline results")
    p.add_argument("--density-counts", type=Path, help="use eligible groups from frozen kk/ru word counts")
    args = p.parse_args()
    report = render(args.pred, args.protocol, args.title, args.density_counts)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(report, encoding="utf-8")
    print(report)
    print(f"Report: {args.out}")


if __name__ == "__main__":
    main()
