"""Check finished annotation batches, and report the distributions that decide
whether the corpus can actually test H2 and H3.

    python -m src.eval.validate_annotations data/interim/batches/enriched_01.csv
    python -m src.eval.validate_annotations data/interim/batches/*.csv
    python -m src.eval.validate_annotations data/interim/batches/*.csv --strict

Run this after every batch, not once at the end. Annotation is done in Excel,
which offers no validation of its own: a stray "pso", a density left blank, a
row deleted by accident or a comment_id reformatted by autocorrect all pass
silently and only surface weeks later when the models refuse to train.

`--strict` exits non-zero on any error, so it can gate a commit.

## The two distributions that matter

Beyond typo-checking, this prints two things that are easy to forget until it
is too late to fix them:

**Sentiment within each density bucket.** H2 says the transformer's advantage
shrinks as density rises. If the `high` bucket turns out to be 85% positive,
a majority-class baseline already scores ~0.85 there and no representation can
show a meaningful gap - H2 becomes untestable on this corpus, not disproved by
it. Catch that while there is still time to sample differently.

**Human density vs. the screen's guess.** The screen picked which comments got
annotated, so its errors are the corpus's selection bias. Quantifying that
agreement is the only way to state the bias honestly in the report.
"""

import argparse
import csv
import glob
import sys
from collections import Counter, defaultdict
from pathlib import Path

from src.preprocess.textnorm import utf8_stdout

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "data" / "interim" / "batches" / "manifest.csv"

VALID = {
    "sentiment": {"pos", "neg", "neu", "skip"},
    "density": {"mono", "low", "med", "high"},
    "base_lang": {"kk", "ru"},
}
# Typos seen in practice, mapped to what was meant. Reported, not silently fixed.
LIKELY_TYPO = {
    "pos": "pos", "poz": "pos", "posit": "pos", "p": "pos", "+": "pos",
    "neg": "neg", "negative": "neg", "n": "neg", "-": "neg",
    "neu": "neu", "neutral": "neu", "net": "neu", "0": "neu",
    "mono": "mono", "m": "mono", "none": "mono",
    "hi": "high", "h": "high", "med": "med", "medium": "med", "l": "low",
    "kaz": "kk", "kz": "kk", "rus": "ru",
}


def load_manifest():
    """comment_id -> its sampling row.

    A comment drawn into the retest batch appears in manifest.csv twice: once
    for the batch that sampled it, once for the retest re-draw. The retest row
    carries no inclusion probability (it is not an independent draw), so it
    must not overwrite the real one.
    """
    if not MANIFEST.exists():
        return {}
    out = {}
    with open(MANIFEST, encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            cid = r["comment_id"]
            if cid in out and r.get("sample") == "retest":
                continue
            out[cid] = r
    return out


def check_row(r, lineno, path, errors, warnings):
    cid = (r.get("comment_id") or "").strip()
    sent = (r.get("sentiment") or "").strip().lower()
    dens = (r.get("density") or "").strip().lower()
    lang = (r.get("base_lang") or "").strip().lower()
    where = "{}:{}".format(Path(path).name, lineno)

    if not cid:
        errors.append("{}  missing comment_id".format(where))
        return None
    if not sent:
        return None                                  # not annotated yet

    if sent not in VALID["sentiment"]:
        hint = LIKELY_TYPO.get(sent)
        errors.append("{}  sentiment={!r}{}".format(
            where, sent, "  (did you mean {!r}?)".format(hint) if hint else ""))

    if sent == "skip":
        # A skipped comment leaves the corpus; density/base_lang must be blank
        # so it cannot be mistaken for a labelled row downstream.
        if dens or lang:
            warnings.append("{}  sentiment=skip but density/base_lang filled "
                            "- they will be discarded".format(where))
        return {"comment_id": cid, "sentiment": "skip", "density": "", "base_lang": ""}

    if not dens:
        errors.append("{}  sentiment filled but density blank".format(where))
    elif dens not in VALID["density"]:
        hint = LIKELY_TYPO.get(dens)
        errors.append("{}  density={!r}{}".format(
            where, dens, "  (did you mean {!r}?)".format(hint) if hint else ""))

    if not lang:
        errors.append("{}  sentiment filled but base_lang blank".format(where))
    elif lang not in VALID["base_lang"]:
        hint = LIKELY_TYPO.get(lang)
        errors.append("{}  base_lang={!r}{}".format(
            where, lang, "  (did you mean {!r}?)".format(hint) if hint else ""))

    return {"comment_id": cid, "sentiment": sent, "density": dens, "base_lang": lang}


def main():
    utf8_stdout()
    p = argparse.ArgumentParser()
    p.add_argument("files", nargs="+", help="annotated batch CSVs (globs allowed)")
    p.add_argument("--strict", action="store_true",
                   help="exit non-zero if there is any error")
    args = p.parse_args()

    paths = []
    for pat in args.files:
        paths += sorted(glob.glob(pat)) or [pat]
    paths = [p for p in paths if Path(p).name not in ("manifest.csv",)]

    manifest = load_manifest()
    errors, warnings = [], []
    labeled, blank, total = [], 0, 0
    seen_ids = set()

    for path in paths:
        if not Path(path).exists():
            errors.append("{}  file not found".format(path))
            continue
        # A retest batch re-issues comments that are already in the pass-1
        # batches - that is what it is for, so its ids are not duplicates.
        is_retest = "retest" in Path(path).name
        with open(path, encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            missing = [c for c in ("comment_id", "text", "sentiment", "density",
                                   "base_lang") if c not in (reader.fieldnames or [])]
            if missing:
                errors.append("{}  missing columns: {}".format(
                    Path(path).name, ", ".join(missing)))
                continue
            for i, r in enumerate(reader, start=2):
                total += 1
                cid = (r.get("comment_id") or "").strip()
                if not is_retest:
                    if cid in seen_ids:
                        warnings.append("{}:{}  comment_id appears twice across "
                                        "batches".format(Path(path).name, i))
                    seen_ids.add(cid)
                if manifest and cid and cid not in manifest:
                    errors.append("{}:{}  comment_id not in manifest.csv - the "
                                  "cell was probably edited".format(
                                      Path(path).name, i))
                row = check_row(r, i, path, errors, warnings)
                if row is None:
                    blank += 1
                else:
                    labeled.append(row)

    print("files: {}   rows: {}   annotated: {}   blank: {}".format(
        len(paths), total, len(labeled), blank))

    if errors:
        print("\n!! {} ERROR(S)".format(len(errors)))
        for e in errors[:40]:
            print("   " + e)
        if len(errors) > 40:
            print("   ... and {} more".format(len(errors) - 40))
    if warnings:
        print("\n{} warning(s)".format(len(warnings)))
        for w in warnings[:20]:
            print("   " + w)
        if len(warnings) > 20:
            print("   ... and {} more".format(len(warnings) - 20))

    usable = [r for r in labeled if r["sentiment"] != "skip"]
    n_skip = len(labeled) - len(usable)
    if labeled:
        print("\n--- sentiment ---")
        c = Counter(r["sentiment"] for r in labeled)
        for k in ("pos", "neg", "neu", "skip"):
            print("  {:<6} {:>5}  {:>5.1f}%".format(
                k, c[k], c[k] / len(labeled) * 100))
        print("  skip rate {:.1f}% -> pull-to-label ratio is about {:.2f}x".format(
            n_skip / len(labeled) * 100,
            len(labeled) / max(1, len(usable))))

    if usable:
        print("\n--- density ---")
        c = Counter(r["density"] for r in usable)
        for k in ("mono", "low", "med", "high"):
            print("  {:<6} {:>5}  {:>5.1f}%".format(k, c[k], c[k] / len(usable) * 100))

        print("\n--- sentiment WITHIN each density bucket ---")
        print("  If one class dominates a bucket, H2/H3 cannot be tested there.")
        print("  {:<6} {:>5} {:>7} {:>7} {:>7} {:>10}".format(
            "", "n", "pos", "neg", "neu", "majority"))
        for d in ("mono", "low", "med", "high"):
            rows = [r for r in usable if r["density"] == d]
            if not rows:
                print("  {:<6} {:>5}   -- empty --".format(d, 0))
                continue
            cc = Counter(r["sentiment"] for r in rows)
            maj = max(cc.values()) / len(rows)
            flag = "  <-- degenerate" if maj >= 0.80 else ""
            print("  {:<6} {:>5} {:>6.0f}% {:>6.0f}% {:>6.0f}% {:>9.0f}%{}".format(
                d, len(rows), cc["pos"] / len(rows) * 100,
                cc["neg"] / len(rows) * 100, cc["neu"] / len(rows) * 100,
                maj * 100, flag))

        print("\n--- base_lang ---")
        c = Counter(r["base_lang"] for r in usable)
        for k in ("kk", "ru"):
            print("  {:<6} {:>5}  {:>5.1f}%".format(k, c[k], c[k] / len(usable) * 100))

    # ---- human density vs. the screen that selected these comments ----------
    if usable and manifest:
        pairs = [(manifest[r["comment_id"]]["screen_bucket"], r["density"])
                 for r in usable if r["comment_id"] in manifest]
        if pairs:
            agree = sum(a == b for a, b in pairs)
            print("\n--- screen vs. human density ({} comments) ---".format(len(pairs)))
            print("  exact agreement: {:.1f}%".format(agree / len(pairs) * 100))
            mixed = {"low", "med", "high"}
            tp = sum(1 for a, b in pairs if a in mixed and b in mixed)
            fp = sum(1 for a, b in pairs if a in mixed and b not in mixed)
            fn = sum(1 for a, b in pairs if a not in mixed and b in mixed)
            prec = tp / (tp + fp) if tp + fp else 0
            rec = tp / (tp + fn) if tp + fn else 0
            print("  screen as a mixed/not-mixed detector: "
                  "precision {:.2f}, recall {:.2f}".format(prec, rec))
            print("  -> recall is the selection bias: {} truly mixed comments "
                  "the screen would have missed.".format(fn))
            grid = defaultdict(Counter)
            for a, b in pairs:
                grid[a][b] += 1
            print("\n  rows = screen, cols = human")
            print("  {:<9}{:>7}{:>7}{:>7}{:>7}".format("", "mono", "low", "med", "high"))
            for a in ("mono", "low", "med", "high", "unknown"):
                if not grid[a]:
                    continue
                print("  {:<9}{:>7}{:>7}{:>7}{:>7}".format(
                    a, grid[a]["mono"], grid[a]["low"], grid[a]["med"], grid[a]["high"]))

    if errors:
        print("\nFix the errors above in the source CSVs, then re-run.")
        if args.strict:
            sys.exit(1)
    elif labeled:
        print("\nNo errors.")


if __name__ == "__main__":
    main()
