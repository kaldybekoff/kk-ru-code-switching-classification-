"""Cohen's kappa between two annotation sheets, computed per axis.

    python -m src.eval.kappa data/interim/annotations_yk.csv data/interim/annotations_partner.csv

The disagreement table is the useful output - it tells you what to fix in the
guideline before annotating the full corpus.
"""

import csv
import sys
from collections import Counter
from pathlib import Path


def _utf8_stdout():
    """Windows consoles default to a legacy codepage and mangle Cyrillic."""
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


def load(path):
    out = {}
    with open(path, encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            cid = (r.get("comment_id") or "").strip()
            if cid:
                out[cid] = r
    return out


def kappa(pairs):
    """Cohen's kappa for a list of (label_a, label_b) pairs."""
    n = len(pairs)
    if n == 0:
        return None
    po = sum(a == b for a, b in pairs) / n
    ca = Counter(a for a, _ in pairs)
    cb = Counter(b for _, b in pairs)
    pe = sum(ca[k] / n * cb[k] / n for k in set(ca) | set(cb))
    if pe == 1.0:  # both annotators used a single label throughout
        return 1.0 if po == 1.0 else 0.0
    return (po - pe) / (1 - pe)


def interpret(k):
    if k is None:
        return "n/a"
    for threshold, label in ((0.80, "almost perfect"), (0.60, "substantial"),
                             (0.40, "moderate"), (0.20, "fair")):
        if k >= threshold:
            return label
    return "slight - guideline needs work"


def report(a, b, axis, name_a, name_b, target):
    pairs, disagreements = [], []
    for cid in sorted(set(a) & set(b)):
        va = (a[cid].get(axis) or "").strip().lower()
        vb = (b[cid].get(axis) or "").strip().lower()
        if not va or not vb:
            continue
        # A comment skipped by either annotator leaves the corpus entirely, so it
        # is not a disagreement about density - exclude it from that axis.
        if axis != "sentiment" and "skip" in (va, vb):
            continue
        pairs.append((va, vb))
        if va != vb:
            disagreements.append((cid, va, vb, (a[cid].get("text") or "")[:70]))

    print("\n=== {} ===".format(axis))
    if not pairs:
        print("no comparable rows")
        return

    k = kappa(pairs)
    agree = sum(x == y for x, y in pairs)
    print("compared: {}   agreed: {} ({:.1f}%)".format(
        len(pairs), agree, agree / len(pairs) * 100))
    flag = "OK" if k >= target else "BELOW TARGET ({})".format(target)
    print("Cohen's kappa: {:.3f}  - {}  [{}]".format(k, interpret(k), flag))

    if disagreements:
        print("\ndisagreements ({}):".format(len(disagreements)))
        print("{:<28} {:<8} {:<8} text".format("comment_id", name_a, name_b))
        for cid, va, vb, txt in disagreements[:40]:
            print("{:<28} {:<8} {:<8} {}".format(cid, va, vb, txt))
        if len(disagreements) > 40:
            print("... and {} more".format(len(disagreements) - 40))
        confusions = Counter(tuple(sorted((va, vb))) for _, va, vb, _ in disagreements)
        print("\nmost common confusions:")
        for (x, y), c in confusions.most_common(5):
            print("  {} <-> {}: {}".format(x, y, c))


def main():
    _utf8_stdout()
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    fa, fb = Path(sys.argv[1]), Path(sys.argv[2])
    a, b = load(fa), load(fb)

    common = set(a) & set(b)
    print("{}: {} rows | {}: {} rows | overlap: {}".format(
        fa.name, len(a), fb.name, len(b), len(common)))
    if not common:
        sys.exit("No shared comment_ids - did both annotate the same batch?")

    name_a, name_b = fa.stem[-6:], fb.stem[-6:]
    report(a, b, "sentiment", name_a, name_b, 0.60)
    report(a, b, "density", name_a, name_b, 0.70)

    print("\nNext: review every disagreement together, add the recurring patterns to "
          "annotation/guideline.md, bump the version, re-run on a FRESH batch.")


if __name__ == "__main__":
    main()
