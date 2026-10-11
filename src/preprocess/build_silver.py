"""Build a separate, explicitly AI-labeled corpus from local_silver output.

The human corpus and its finalization gate are never changed by this script.
"""

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

from src.eval.corpus_audit import BATCHES, MANIFEST, audit, load_sheets, read_csv
from src.preprocess.merge_annotations import CLEAN, FIELDS

ROOT = Path(__file__).resolve().parents[2]
PRED = ROOT / "data" / "interim" / "silver_predictions.jsonl"
OUT = ROOT / "data" / "labeled" / "corpus_silver.csv"
TRAIN = ROOT / "data" / "labeled" / "corpus_silver_train.csv"
TEST = ROOT / "data" / "labeled" / "corpus_ai_review_test.csv"
REVIEW = ROOT / "data" / "labeled" / "validation_ai_review.csv"
BLIND = ROOT / "data" / "interim" / "validation_subset_blind.csv"


def write_rows(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".csv.tmp")
    with open(temp, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS + ["label_source", "annotation_model"])
        writer.writeheader()
        writer.writerows(rows)
    temp.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pred", type=Path, default=PRED)
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--train-out", type=Path, default=TRAIN)
    parser.add_argument("--test-out", type=Path, default=TEST)
    parser.add_argument("--review", type=Path, default=REVIEW)
    args = parser.parse_args()
    status = audit()
    if status["issues"]:
        raise SystemExit(f"Annotation sheets have {len(status['issues'])} structural issue(s)")
    paths = sorted(p for p in BATCHES.glob("*.csv")
                   if p.stem.startswith(("natural_", "enriched_")))
    sheets, issues = load_sheets(paths)
    if issues:
        raise SystemExit(f"Annotation sheets have {len(issues)} duplicate/column issue(s)")
    _, manifest_rows = read_csv(MANIFEST)
    manifest = {r["comment_id"]: r for r in manifest_rows if r["sample"] != "retest"}
    if not args.pred.exists():
        raise SystemExit(f"Predictions missing: {args.pred}")
    predictions = {}
    for line in args.pred.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        cid = record["comment_id"]
        if cid in predictions:
            raise SystemExit(f"Duplicate AI prediction: {cid}")
        predictions[cid] = record
    if set(predictions) != set(manifest):
        raise SystemExit(f"AI predictions cover {len(predictions)} of {len(manifest)} "
                         "sampled IDs; regenerate missing predictions first")
    unresolved = [cid for cid, r in predictions.items()
                  if r.get("label") not in ("pos", "neg", "neu", "skip")]
    if unresolved:
        raise SystemExit(f"{len(unresolved)} AI predictions have no valid label")
    models = {r["model"] for r in predictions.values()}
    if len(models) != 1:
        raise SystemExit("AI predictions came from different model versions")
    revisions = {r.get("revision") for r in predictions.values()}
    if len(revisions) != 1:
        raise SystemExit("AI predictions came from different model revisions")
    clean = {}
    with open(CLEAN, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                row = json.loads(line)
                if row["comment_id"] in manifest:
                    clean[row["comment_id"]] = row
    if set(clean) != set(manifest):
        raise SystemExit("Sampled IDs are missing from the clean corpus")
    for cid, prediction in predictions.items():
        expected = prediction.get("text_sha256")
        if expected and expected != hashlib.sha256(clean[cid]["text"].encode()).hexdigest():
            raise SystemExit(f"AI prediction text differs from clean corpus: {cid}")
    rows = []
    for cid, m in manifest.items():
        prediction = predictions[cid]
        if prediction["label"] == "skip":
            continue
        c = clean[cid]
        sheet = sheets[cid][0]
        rows.append({
            "comment_id": cid, "text": c["text"],
            "sentiment": prediction["label"], "density": sheet["density"],
            "base_lang": sheet["base_lang"],
            "notes": "AI silver label; not independently human verified",
            "video_id": c["video_id"], "n_tokens": c["n_tokens"],
            "density_eligible": c["density_eligible"],
            "sample": m["sample"], "screen_bucket": c["screen_bucket"],
            "inclusion_prob": m["inclusion_prob"],
            "label_source": "local_ai", "annotation_model": prediction["model"],
        })
    if not args.review.exists():
        raise SystemExit(f"Blind review missing: {args.review}")
    with open(args.review, encoding="utf-8-sig", newline="") as f:
        review_rows = list(csv.DictReader(f))
    review_ids = {r["comment_id"] for r in review_rows}
    if len(review_ids) != len(review_rows) or not review_ids <= set(manifest):
        raise SystemExit("Blind review contains duplicate or unsampled IDs")
    with open(BLIND, encoding="utf-8-sig", newline="") as f:
        blind_rows = list(csv.DictReader(f))
    blind_ids = {r["comment_id"] for r in blind_rows}
    if len(blind_ids) != len(blind_rows) or review_ids != blind_ids:
        raise SystemExit("Blind review does not exactly cover the validation subset")
    for review in review_rows:
        if review["text"] != clean[review["comment_id"]]["text"]:
            raise SystemExit(f"Blind review text differs from clean corpus: {review['comment_id']}")
    known_skips = {cid for cid, (sheet, _, _) in sheets.items()
                   if sheet["sentiment"] == "skip"}
    known_skips.update(r["comment_id"] for r in review_rows
                       if r["sentiment"] == "skip")
    rows = [r for r in rows if r["comment_id"] not in known_skips]
    test_rows = []
    for review in review_rows:
        cid = review["comment_id"]
        if review["sentiment"] == "skip":
            continue
        if (review["sentiment"] not in ("pos", "neg", "neu") or
                review["density"] not in ("mono", "low", "med", "high") or
                review["base_lang"] not in ("kk", "ru")):
            raise SystemExit(f"Incomplete independent review for {cid}")
        c, m = clean[cid], manifest[cid]
        test_rows.append({
            "comment_id": cid, "text": c["text"],
            "sentiment": review["sentiment"], "density": review["density"],
            "base_lang": review["base_lang"],
            "notes": "blind Codex AI review; not human gold",
            "video_id": c["video_id"], "n_tokens": c["n_tokens"],
            "density_eligible": c["density_eligible"],
            "sample": m["sample"], "screen_bucket": c["screen_bucket"],
            "inclusion_prob": m["inclusion_prob"],
            "label_source": "codex_blind_ai_review", "annotation_model": "Codex",
        })
    review_videos = {clean[cid]["video_id"] for cid in review_ids}
    train_rows = [r for r in rows if r["video_id"] not in review_videos]
    write_rows(args.out, rows)
    write_rows(args.train_out, train_rows)
    write_rows(args.test_out, test_rows)
    print(f"Saved {len(rows)} silver rows to {args.out}")
    print(f"Excluded {len(known_skips)} known skip comments from the modeling corpus")
    print(f"Video-disjoint training set: {len(train_rows)} rows -> {args.train_out}")
    print(f"Independent AI review test: {len(test_rows)} rows -> {args.test_out}")
    print("Sentiment:", dict(Counter(r["sentiment"] for r in rows)))
    print("Density:", dict(Counter(r["density"] for r in rows)))
    print("Annotation model:", models.pop())


if __name__ == "__main__":
    main()
