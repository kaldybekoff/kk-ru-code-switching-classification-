"""Merge annotated batches into the single labeled dataset the models train on.

    python -m src.preprocess.merge_annotations data/interim/batches/natural_*.csv \
                                               data/interim/batches/enriched_*.csv
    python -m src.preprocess.merge_annotations "data/interim/batches/*.csv" --exclude retest

Writes data/labeled/corpus_labeled.csv - the one data file that is committed.

It refuses to run while `validate_annotations` still reports errors. That is
deliberate: a dataset built from a batch with a stray label is worse than no
dataset, because the error survives into every experiment downstream.

Columns written:
    comment_id, text, sentiment, density, base_lang, notes
    video_id, n_tokens, density_eligible      - joined from the clean corpus
    sample, screen_bucket, inclusion_prob     - joined from the sampling manifest

The last three exist so the sampling design travels with the data. Any claim
about how frequent something is in the population must be reweighted by
1/inclusion_prob, or restricted to rows with sample == "natural".
`skip` rows are dropped here and counted in the summary.
"""

import argparse
import csv
import glob
import json
import sys
from collections import Counter
from pathlib import Path

from src.eval.validate_annotations import VALID, load_manifest
from src.preprocess.textnorm import utf8_stdout

ROOT = Path(__file__).resolve().parents[2]
CLEAN = ROOT / "data" / "interim" / "corpus_clean.jsonl"
OUT = ROOT / "data" / "labeled" / "corpus_labeled.csv"

FIELDS = ["comment_id", "text", "sentiment", "density", "base_lang", "notes",
          "video_id", "n_tokens", "density_eligible",
          "sample", "screen_bucket", "inclusion_prob"]


def load_clean_index():
    if not CLEAN.exists():
        raise SystemExit("{} not found. Run build_corpus first.".format(CLEAN))
    idx = {}
    with open(CLEAN, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                idx[r["comment_id"]] = r
    return idx


def main():
    utf8_stdout()
    p = argparse.ArgumentParser()
    p.add_argument("files", nargs="+", help="annotated batch CSVs (globs allowed)")
    p.add_argument("--exclude", nargs="*", default=["retest", "manifest"],
                   help="skip files whose name contains any of these")
    p.add_argument("--out", default=str(OUT))
    p.add_argument("--force", action="store_true",
                   help="write even if rows fail validation (they are dropped)")
    args = p.parse_args()

    paths = []
    for pat in args.files:
        paths += sorted(glob.glob(pat)) or [pat]
    paths = [p for p in paths
             if not any(x in Path(p).name for x in args.exclude)]
    if not paths:
        raise SystemExit("No batch files matched.")

    clean = load_clean_index()
    manifest = load_manifest()

    merged, bad, skipped, blank = {}, [], 0, 0
    for path in paths:
        with open(path, encoding="utf-8-sig", newline="") as f:
            for i, r in enumerate(csv.DictReader(f), start=2):
                cid = (r.get("comment_id") or "").strip()
                sent = (r.get("sentiment") or "").strip().lower()
                dens = (r.get("density") or "").strip().lower()
                lang = (r.get("base_lang") or "").strip().lower()
                if not cid or not sent:
                    blank += 1
                    continue
                if sent == "skip":
                    skipped += 1
                    continue
                if (sent not in VALID["sentiment"] or dens not in VALID["density"]
                        or lang not in VALID["base_lang"]):
                    bad.append("{}:{}  {}/{}/{}".format(
                        Path(path).name, i, sent or "-", dens or "-", lang or "-"))
                    continue
                if cid not in clean:
                    bad.append("{}:{}  comment_id not in corpus_clean".format(
                        Path(path).name, i))
                    continue
                c = clean[cid]
                m = manifest.get(cid, {})
                # A comment re-annotated in two batches keeps its first label;
                # disagreements belong in the kappa report, not here.
                merged.setdefault(cid, {
                    "comment_id": cid,
                    "text": c["text"],
                    "sentiment": sent,
                    "density": dens,
                    "base_lang": lang,
                    "notes": (r.get("notes") or "").strip(),
                    "video_id": c["video_id"],
                    "n_tokens": c["n_tokens"],
                    "density_eligible": c["density_eligible"],
                    "sample": m.get("sample", ""),
                    "screen_bucket": c["screen_bucket"],
                    "inclusion_prob": m.get("inclusion_prob", ""),
                })

    if bad and not args.force:
        print("{} invalid row(s):".format(len(bad)))
        for b in bad[:20]:
            print("   " + b)
        sys.exit("\nRun `python -m src.eval.validate_annotations <files>` and fix "
                 "them, or pass --force to drop them.")

    rows = list(merged.values())
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)

    print("{} batches -> {} labeled comments".format(len(paths), len(rows)))
    print("  dropped: {} skip, {} unannotated, {} invalid".format(
        skipped, blank, len(bad)))
    for axis in ("sentiment", "density", "sample"):
        c = Counter(r[axis] for r in rows)
        print("  {:<10} {}".format(
            axis, "  ".join("{}={}".format(k, v) for k, v in c.most_common())))
    dens_ok = sum(1 for r in rows if r["density_eligible"])
    print("  density-eligible (>= 5 tokens): {} of {}".format(dens_ok, len(rows)))
    print("\n-> {}".format(out))


if __name__ == "__main__":
    main()
