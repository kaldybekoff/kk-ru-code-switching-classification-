"""Assign silver sentiment labels with a pretrained multilingual classifier.

This is weak supervision, not human annotation. Predictions are checkpointed and
include the model revision, probabilities and input-text hash for auditing.
"""

import argparse
import hashlib
import json
import os
from collections import Counter
from pathlib import Path

from src.eval.corpus_audit import BATCHES, load_sheets

ROOT = Path(__file__).resolve().parents[2]
MODEL = "cardiffnlp/twitter-xlm-roberta-base-sentiment"
MODEL_REVISION = "f2f1202b1bdeb07342385c3f807f9c07cd8f5cf8"
OUT = ROOT / "data/interim/silver_predictions.jsonl"
CACHE = ROOT / "models_cache/hf"
LABELS = {"positive": "pos", "negative": "neg", "neutral": "neu"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--revision", default=None,
                        help="Hugging Face commit hash; default pins Cardiff model")
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--calibrate-only", action="store_true")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    paths = sorted(p for p in BATCHES.glob("*.csv")
                   if p.stem.startswith(("natural_", "enriched_")))
    sheets, issues = load_sheets(paths)
    if issues:
        raise SystemExit(f"Annotation sheets contain {len(issues)} issues")
    selected = []
    for cid, (row, _, _) in sheets.items():
        gold = (row.get("sentiment") or "").strip().lower()
        if args.calibrate_only and gold not in (*LABELS.values(), "skip"):
            continue
        selected.append({"comment_id": cid, "text": row["text"], "gold": gold})
    if args.limit is not None:
        selected = selected[:args.limit]
    if not selected:
        raise SystemExit("No comments selected")

    # Regular HTTP supports resumable downloads in the local Windows setup.
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    CACHE.mkdir(parents=True, exist_ok=True)
    if args.revision:
        revision = args.revision
    elif args.model == MODEL:
        revision = MODEL_REVISION
    else:
        from huggingface_hub import model_info
        revision = model_info(args.model).sha
    existing = {}
    if args.out.exists() and not args.calibrate_only:
        for line in args.out.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rec = json.loads(line)
                if rec["model"] != args.model or rec.get("revision") != revision:
                    raise SystemExit("Checkpoint belongs to another model/revision")
                if rec["comment_id"] in existing:
                    raise SystemExit("Duplicate checkpoint prediction")
                existing[rec["comment_id"]] = rec
        selected = [r for r in selected if r["comment_id"] not in existing]
    if not selected:
        print(f"All {len(existing)} predictions already present")
        return

    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=revision,
                                               cache_dir=CACHE, use_fast=False)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model, revision=revision, cache_dir=CACHE).eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    if device == "cpu":
        torch.set_num_threads(8)
    mapping = {int(i): LABELS[label.lower()] for i, label in model.config.id2label.items()}
    if set(mapping.values()) != set(LABELS.values()):
        raise SystemExit(f"Unexpected model label mapping: {mapping}")
    print(f"Model {args.model}@{revision} on {device}; {len(selected)} comments", flush=True)
    confusion = Counter()
    counts = Counter()
    handle = None if args.calibrate_only else args.out.open("a", encoding="utf-8")
    try:
        for start in range(0, len(selected), args.batch_size):
            batch = selected[start:start + args.batch_size]
            inputs = tokenizer([r["text"] for r in batch], truncation=True,
                               max_length=args.max_length, padding=True,
                               return_tensors="pt").to(device)
            with torch.inference_mode():
                scores = model(**inputs).logits.softmax(-1).cpu().tolist()
            for row, probs in zip(batch, scores):
                label = mapping[max(range(len(probs)), key=lambda i: probs[i])]
                counts[label] += 1
                if args.calibrate_only:
                    confusion[(row["gold"], label)] += 1
                    print(row["comment_id"][:12], row["gold"], label,
                          round(max(probs), 3), flush=True)
                else:
                    handle.write(json.dumps({
                        "comment_id": row["comment_id"], "model": args.model,
                        "revision": revision, "label": label,
                        "probabilities": {mapping[i]: round(p, 6)
                                          for i, p in enumerate(probs)},
                        "text_sha256": hashlib.sha256(row["text"].encode()).hexdigest(),
                        "method": f"XLM-R sequence classification, max_length={args.max_length}",
                    }, ensure_ascii=False) + "\n")
            if handle:
                handle.flush()
            print(f"{start + len(batch)}/{len(selected)} {dict(counts)}", flush=True)
    finally:
        if handle:
            handle.close()
    if args.calibrate_only:
        correct = sum(v for (gold, pred), v in confusion.items() if gold == pred)
        print(f"Calibration agreement: {correct}/{sum(confusion.values())}")
        print(dict(confusion))
    else:
        print(f"Saved predictions: {args.out}", flush=True)


if __name__ == "__main__":
    main()
