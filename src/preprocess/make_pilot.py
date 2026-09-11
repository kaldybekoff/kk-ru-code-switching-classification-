"""Turn raw JSONL comments into a clean, anonymized pilot batch for annotation.

    python -m src.preprocess.make_pilot --n 100

Produces two files in data/interim/:
  pilot_<n>.csv        - the batch (comment_id, text)
  pilot_<n>_blank.csv  - the annotation sheet, copy one per annotator

Filtering here is deliberately light: the goal is to test the guideline on
realistic text, not to build the final corpus.
"""

import argparse
import csv
import hashlib
import json
import random
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"
INTERIM = ROOT / "data" / "interim"

URL_RE = re.compile(r"https?://\S+|www\.\S+")
MENTION_RE = re.compile(r"@[\wЀ-ӿ.\-]+")
WS_RE = re.compile(r"\s+")
# Any Cyrillic letter - comments with none are almost certainly not kk/ru.
CYRILLIC_RE = re.compile(r"[Ѐ-ӿ]")


def _utf8_stdout():
    """Windows consoles default to a legacy codepage and mangle Cyrillic."""
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


def anonymize(text):
    text = URL_RE.sub("<URL>", text)
    text = MENTION_RE.sub("<USER>", text)
    return WS_RE.sub(" ", text).strip()


def is_usable(text):
    """Cheap pre-filter. Real spam/off-topic removal is the annotator's `skip`."""
    if len(text) < 3 or len(text) > 500:
        return False
    if not CYRILLIC_RE.search(text):
        return False
    if "<URL>" in text and len(text) < 20:  # bare link
        return False
    return True


def norm_key(text):
    """Dedup key: lowercased, punctuation-stripped, so near-identical spam collapses."""
    k = re.sub(r"[^\w\s]", "", text.lower())
    return hashlib.md5(WS_RE.sub(" ", k).strip().encode()).hexdigest()


def main():
    _utf8_stdout()
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=100)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--infile", help="specific raw JSONL (default: all in data/raw/)")
    p.add_argument("--mixed-share", type=float, default=0.0,
                   help="target share of screened code-switched comments, e.g. 0.5. "
                        "A uniform sample of YouTube comments is ~90%% monolingual, "
                        "which leaves too few mixed examples to exercise the density "
                        "rules. Oversampling is fine for a GUIDELINE PILOT but must be "
                        "documented and NOT reused for the final corpus, whose label "
                        "and density distributions have to stay representative.")
    args = p.parse_args()

    files = [Path(args.infile)] if args.infile else sorted(RAW.glob("*.jsonl"))
    if not files:
        raise SystemExit("No raw JSONL in data/raw/. Run the pull script first.")

    rows, seen_keys, seen_ids = [], set(), set()
    for fp in files:
        for line in open(fp, encoding="utf-8"):
            if not line.strip():
                continue
            c = json.loads(line)
            if c["comment_id"] in seen_ids:
                continue
            seen_ids.add(c["comment_id"])
            text = anonymize(c.get("text", ""))
            if not is_usable(text):
                continue
            key = norm_key(text)
            if key in seen_keys:  # duplicates and near-duplicate spam
                continue
            seen_keys.add(key)
            # Author name is dropped here and never carried further.
            rows.append({"comment_id": c["comment_id"], "text": text})

    print("{} raw -> {} usable & deduplicated".format(len(seen_ids), len(rows)))
    if len(rows) < args.n:
        print("! only {} available, taking all of them".format(len(rows)))

    rng = random.Random(args.seed)
    rng.shuffle(rows)

    if args.mixed_share > 0:
        from src.preprocess.cs_screen import bucket, classify
        mixed, mono = [], []
        for r in rows:
            kk, ru, _ = classify(r["text"])
            (mixed if bucket(kk, ru) in ("low", "med", "high") else mono).append(r)
        want_mixed = min(len(mixed), int(args.n * args.mixed_share))
        batch = mixed[:want_mixed] + mono[: args.n - want_mixed]
        rng.shuffle(batch)
        print("oversampled: {} screened-mixed + {} other "
              "(pool had {} mixed / {} other)".format(
                  want_mixed, len(batch) - want_mixed, len(mixed), len(mono)))
    else:
        batch = rows[: args.n]

    INTERIM.mkdir(parents=True, exist_ok=True)
    ref = INTERIM / "pilot_{}.csv".format(len(batch))
    with open(ref, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["comment_id", "text"])
        w.writeheader()
        w.writerows(batch)

    def write_sheet(path, ordered):
        with open(path, "w", encoding="utf-8-sig", newline="") as f:
            w = csv.writer(f)
            w.writerow(["comment_id", "text", "sentiment", "density", "base_lang", "notes"])
            for r in ordered:
                w.writerow([r["comment_id"], r["text"], "", "", "", ""])

    pass1 = INTERIM / "pilot_{}_pass1.csv".format(len(batch))
    write_sheet(pass1, batch)

    # Same comments, different order. Annotating the batch twice a day apart gives
    # intra-annotator agreement (test-retest reliability) - the solo substitute for
    # inter-annotator kappa. Reordering stops you recognising rows by position.
    pass2_order = list(batch)
    random.Random(args.seed + 1).shuffle(pass2_order)
    pass2 = INTERIM / "pilot_{}_pass2.csv".format(len(batch))
    write_sheet(pass2, pass2_order)

    print("\n{}\n{}\n{}".format(ref, pass1, pass2))
    print("\nAnnotate pass1 today. Tomorrow annotate pass2 WITHOUT looking at pass1 "
          "(same comments, different order). Then run:\n"
          "  python -m src.eval.kappa data/interim/{} data/interim/{}".format(
              pass1.name, pass2.name))


if __name__ == "__main__":
    main()
