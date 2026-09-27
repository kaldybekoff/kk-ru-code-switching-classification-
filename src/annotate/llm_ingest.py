"""Read the model's JSON replies back into the batch CSVs, and build the audit
queue the sprint plan requires.

    python -m src.annotate.llm_ingest
    python -m src.annotate.llm_ingest --dry-run

Reads data/interim/llm/reply_*.json, checks each one against the prompt it
answers, and writes `sentiment` / `base_lang` / `notes` into the matching rows
of data/interim/batches/*.csv. `density` is never written here.

Also writes data/interim/llm/audit_queue.csv - every row the model marked
`confidence: low`, plus a stratified random slice of the `high`-confidence ones.
That second part matters: auditing only what the model already doubted measures
its self-knowledge, not its accuracy. The random slice is what makes the
spot-check an unbiased estimate.

Replies are treated as untrusted input. A model that renames a field, invents
an id, drops rows from the middle of an array or wraps the JSON in markdown
will be caught and reported rather than silently written into the corpus.
"""

import argparse
import csv
import glob
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

from src.eval.validate_annotations import VALID
from src.preprocess.textnorm import utf8_stdout

ROOT = Path(__file__).resolve().parents[2]
BATCHES = ROOT / "data" / "interim" / "batches"
LLMDIR = ROOT / "data" / "interim" / "llm"
AUDIT = LLMDIR / "audit_queue.csv"

FIELDS = ["comment_id", "text", "sentiment", "density", "base_lang", "notes"]
# Share of high-confidence rows pulled into the audit queue at random.
SPOTCHECK_RATE = 0.12


def parse_reply(path):
    """-> (records, error). Tolerates a markdown fence; rejects anything else."""
    raw = path.read_text(encoding="utf-8").strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```[a-zA-Z]*\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw).strip()
    # A reply that starts with prose but contains one array is still recoverable.
    if not raw.startswith("["):
        m = re.search(r"\[.*\]", raw, re.S)
        if not m:
            return None, "no JSON array found"
        raw = m.group(0)
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as e:
        return None, "invalid JSON: {}".format(e)
    if not isinstance(data, list):
        return None, "top level is {}, expected a list".format(type(data).__name__)
    return data, None


def check_record(rec, expected_ids, seen):
    """-> (clean_record, error)."""
    if not isinstance(rec, dict):
        return None, "not an object"
    cid = str(rec.get("id", "")).strip()
    if not cid:
        return None, "missing id"
    if cid not in expected_ids:
        return None, "id {} was not in this prompt".format(cid[:24])
    if cid in seen:
        return None, "id {} answered twice".format(cid[:24])

    sent = str(rec.get("sentiment", "")).strip().lower()
    lang = str(rec.get("base_lang", "") or "").strip().lower()
    conf = str(rec.get("confidence", "high")).strip().lower()
    note = str(rec.get("note", "") or "").strip()

    if sent not in VALID["sentiment"]:
        return None, "{}: sentiment {!r}".format(cid[:12], sent)
    if sent == "skip":
        lang = ""
    elif lang not in VALID["base_lang"]:
        return None, "{}: base_lang {!r}".format(cid[:12], lang)
    if conf not in ("high", "low"):
        conf = "low"                      # anything unexpected goes to the audit

    return {"id": cid, "sentiment": sent, "base_lang": lang,
            "confidence": conf, "note": note}, None


def main():
    utf8_stdout()
    p = argparse.ArgumentParser()
    p.add_argument("--dry-run", action="store_true",
                   help="validate the replies, write nothing")
    p.add_argument("--spotcheck-rate", type=float, default=SPOTCHECK_RATE)
    p.add_argument("--seed", type=int, default=2026)
    args = p.parse_args()

    index_path = LLMDIR / "index.json"
    if not index_path.exists():
        raise SystemExit("{} not found. Run llm_batches first.".format(index_path))
    index = {e["prompt"]: e for e in json.loads(index_path.read_text(encoding="utf-8"))}

    replies = sorted(glob.glob(str(LLMDIR / "reply_*.json")))
    if not replies:
        raise SystemExit(
            "No reply_*.json in {}.\nPaste each model answer as "
            "reply_01.json, reply_02.json, ...".format(LLMDIR))

    accepted, problems = {}, []
    missing_total = 0
    for rp in replies:
        stem = Path(rp).stem.replace("reply_", "prompt_")
        meta = index.get(stem)
        if not meta:
            problems.append("{}: no matching prompt".format(Path(rp).name))
            continue
        data, err = parse_reply(Path(rp))
        if err:
            problems.append("{}: {}".format(Path(rp).name, err))
            continue

        expected = set(meta["ids"])
        seen = set()
        for rec in data:
            clean, err = check_record(rec, expected, seen)
            if err:
                problems.append("{}: {}".format(Path(rp).name, err))
                continue
            seen.add(clean["id"])
            accepted[clean["id"]] = clean

        missing = expected - seen
        if missing:
            missing_total += len(missing)
            problems.append("{}: {} of {} comments unanswered".format(
                Path(rp).name, len(missing), len(expected)))

    print("{} replies read, {} usable annotations".format(len(replies), len(accepted)))
    if problems:
        print("\n{} problem(s):".format(len(problems)))
        for pr in problems[:25]:
            print("   " + pr)
        if len(problems) > 25:
            print("   ... and {} more".format(len(problems) - 25))
        if missing_total:
            print("\n{} comments were not answered - rerun llm_batches after "
                  "ingesting to regenerate prompts for what is still "
                  "missing.".format(missing_total))
    if not accepted:
        raise SystemExit("\nNothing usable. Fix the replies above.")

    conf = Counter(r["confidence"] for r in accepted.values())
    sent = Counter(r["sentiment"] for r in accepted.values())
    print("\nsentiment: {}".format(
        "  ".join("{}={}".format(k, sent[k]) for k in ("pos", "neg", "neu", "skip"))))
    print("confidence: high={}  low={} ({:.1f}% flagged for audit)".format(
        conf["high"], conf["low"], conf["low"] / len(accepted) * 100))

    if args.dry_run:
        print("\n--dry-run: nothing written.")
        return

    # ---- write into the batch CSVs ------------------------------------------
    written = 0
    texts = {}
    for bf in sorted(glob.glob(str(BATCHES / "*.csv"))):
        if "manifest" in Path(bf).name:
            continue
        with open(bf, encoding="utf-8-sig", newline="") as f:
            rows = list(csv.DictReader(f))
        touched = False
        for r in rows:
            cid = (r.get("comment_id") or "").strip()
            texts[cid] = r.get("text", "")
            a = accepted.get(cid)
            if not a or (r.get("sentiment") or "").strip():
                continue
            r["sentiment"] = a["sentiment"]
            r["base_lang"] = a["base_lang"]
            if a["note"]:
                r["notes"] = a["note"]
            touched = True
            written += 1
        if touched:
            tmp = Path(str(bf) + ".tmp")
            with open(tmp, "w", encoding="utf-8-sig", newline="") as f:
                w = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
                w.writeheader()
                w.writerows(rows)
            tmp.replace(bf)

    print("\nwrote {} annotations into the batch CSVs".format(written))
    print("density left empty on purpose - run: python -m src.preprocess.density")

    # ---- audit queue ---------------------------------------------------------
    rng = random.Random(args.seed)
    low = [a for a in accepted.values() if a["confidence"] == "low"]
    high = [a for a in accepted.values() if a["confidence"] == "high"]
    rng.shuffle(high)
    n_spot = int(round(len(high) * args.spotcheck_rate))
    queue = low + high[:n_spot]
    rng.shuffle(queue)

    with open(AUDIT, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["comment_id", "text", "llm_sentiment", "llm_base_lang",
                    "llm_confidence", "llm_note",
                    "human_sentiment", "human_base_lang", "human_note"])
        for a in queue:
            w.writerow([a["id"], texts.get(a["id"], ""), a["sentiment"],
                        a["base_lang"], a["confidence"], a["note"], "", "", ""])

    print("\naudit queue: {} rows ({} the model flagged + {} random "
          "high-confidence)".format(len(queue), len(low), n_spot))
    print("-> {}".format(AUDIT))
    print("\nFill human_sentiment / human_base_lang, then:")
    print("  python -m src.annotate.audit_report")


if __name__ == "__main__":
    main()
