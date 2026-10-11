"""Audit AI silver labels against existing labels and a blind Codex review.

The reviewers are models, not two independent human annotators. The report
uses "agreement" deliberately and never calls this human IAA or gold accuracy.
"""

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

from src.eval.corpus_audit import BATCHES, load_sheets
from src.eval.kappa import kappa

ROOT = Path(__file__).resolve().parents[2]
PRED = ROOT / "data" / "interim" / "silver_predictions.jsonl"
REVIEW = ROOT / "data" / "labeled" / "validation_ai_review.csv"
OUT = ROOT / "reports" / "sprint3_silver_audit.md"
LABELS = ("pos", "neg", "neu", "skip")


def compare(pairs):
    if not pairs:
        return None
    return {"n": len(pairs), "exact": sum(a == b for a, b in pairs) / len(pairs),
            "kappa": kappa(pairs),
            "confusion": Counter(pairs)}


def format_kappa(result):
    return "undefined" if result["kappa"] is None else f"{result['kappa']:.3f}"


def table(result, left, right):
    if result is None:
        return ["No overlapping valid labels."]
    lines = [f"Comparable: **{result['n']}**; exact agreement: "
             f"**{result['exact']:.3f}**; Cohen's κ: **{format_kappa(result)}**.", "",
             f"| {left} \\ {right} | " + " | ".join(LABELS) + " |",
             "|---|" + "---:|" * len(LABELS)]
    for a in LABELS:
        lines.append(f"| {a} | " + " | ".join(
            str(result["confusion"][(a, b)]) for b in LABELS) + " |")
    return lines


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pred", type=Path, default=PRED)
    parser.add_argument("--review", type=Path, default=REVIEW)
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    predictions = {}
    for line in args.pred.read_text(encoding="utf-8").splitlines():
        if line.strip():
            record = json.loads(line)
            predictions[record["comment_id"]] = record
    model_names = {r["model"] for r in predictions.values()}
    if len(model_names) != 1:
        raise SystemExit("Silver predictions must use one model")
    revisions = {r.get("revision") for r in predictions.values()}
    if len(revisions) != 1:
        raise SystemExit("Silver predictions must use one model revision")
    paths = sorted(p for p in BATCHES.glob("*.csv")
                   if p.stem.startswith(("natural_", "enriched_")))
    first, issues = load_sheets(paths)
    if issues:
        raise SystemExit("Annotation sheets contain structural errors")
    original = [(row[0]["sentiment"], predictions[cid]["label"])
                for cid, row in first.items() if cid in predictions and
                row[0]["sentiment"] in LABELS and predictions[cid]["label"] in LABELS]
    with open(args.review, encoding="utf-8-sig", newline="") as f:
        review_rows = list(csv.DictReader(f))
    reviewed = [(row["sentiment"], predictions[row["comment_id"]]["label"])
                for row in review_rows if row["comment_id"] in predictions and
                row["sentiment"] in LABELS and
                predictions[row["comment_id"]]["label"] in LABELS]
    density_pairs = [(row["density"], first[row["comment_id"]][0]["density"])
                     for row in review_rows if row["comment_id"] in first and
                     row["density"] in ("mono", "low", "med", "high") and
                     first[row["comment_id"]][0]["density"] in ("mono", "low", "med", "high")]
    by_density = defaultdict(list)
    for row in review_rows:
        cid = row["comment_id"]
        if cid in predictions and row["sentiment"] in LABELS and predictions[cid]["label"] in LABELS:
            by_density[first[cid][0]["density"]].append(
                (row["sentiment"], predictions[cid]["label"]))
    counts = Counter(r["label"] or "unresolved" for r in predictions.values())
    lines = ["# Silver annotation audit", "",
             f"Annotator model: `{next(iter(model_names))}` at revision "
             f"`{next(iter(revisions))}`. Predictions: {len(predictions)}.",
             f"Label counts: {dict(counts)}.", "",
             "These are AI-generated silver labels. The 100-row blind review was "
             "done by Codex without seeing the local model's predictions. "
             "Neither comparison is inter-annotator agreement between humans, "
             "and neither estimates gold-label accuracy.", "",
             "## Existing sparse labels vs local AI", ""]
    lines += table(compare(original), "existing", "local AI")
    lines += ["", "## Blind Codex review vs local AI", ""]
    lines += table(compare(reviewed), "Codex", "local AI")
    density_result = compare(density_pairs)
    lines += ["", "## Blind Codex density review vs deterministic density", ""]
    if density_result:
        lines += [f"Comparable: **{density_result['n']}**; exact agreement: "
                  f"**{density_result['exact']:.3f}**; Cohen's κ: "
                  f"**{format_kappa(density_result)}**.", "",
                  "| Codex \\ algorithm | mono | low | med | high |",
                  "|---|---:|---:|---:|---:|"]
        for label in ("mono", "low", "med", "high"):
            lines.append(f"| {label} | " + " | ".join(
                str(density_result["confusion"][(label, other)])
                for other in ("mono", "low", "med", "high")) + " |")
    else:
        lines.append("No comparable density annotations.")
    lines += ["", "## Blind review agreement by density", "",
              "| Density | n | Exact agreement | Cohen's κ |",
              "|---|---:|---:|---:|"]
    for density in ("mono", "low", "med", "high"):
        result = compare(by_density[density])
        if result:
            lines.append(f"| {density} | {result['n']} | {result['exact']:.3f} | "
                         f"{format_kappa(result)} |")
        else:
            lines.append(f"| {density} | 0 | n/a | n/a |")
    lines += ["", "Small cells, including the 24 existing labels, are exploratory. "
              "Disagreements need human adjudication before this becomes a gold corpus."]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Report: {args.out}")
    print(f"Blind review pairs: {len(reviewed)}")


if __name__ == "__main__":
    main()
