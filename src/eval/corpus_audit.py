"""Audit the sampled annotation sheets before publishing a training corpus.

Run ``python -m src.eval.corpus_audit`` from the repository root. This reads
the frozen sampling manifest and pass-one sheets; it never changes annotations.
An optional ``--second`` sheet is an *independent* annotator's labels, not the
retest sheet from the same annotator.
"""

import argparse
import csv
import glob
import json
from collections import Counter, defaultdict
from pathlib import Path

from src.eval.kappa import kappa
from src.eval.validate_annotations import VALID

ROOT = Path(__file__).resolve().parents[2]
BATCHES = ROOT / "data" / "interim" / "batches"
MANIFEST = BATCHES / "manifest.csv"
REPORT = ROOT / "reports" / "sprint3_corpus_audit.md"
CLEAN = ROOT / "data" / "interim" / "corpus_clean.jsonl"
SENTIMENT = ("pos", "neg", "neu", "skip")
DENSITY = ("mono", "low", "med", "high")


def read_csv(path):
    with open(path, encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        return reader.fieldnames or [], list(reader)


def load_sheets(paths):
    rows = {}
    issues = []
    for path in paths:
        fields, sheet = read_csv(path)
        missing = {"comment_id", "text", "sentiment", "density", "base_lang"} - set(fields)
        if missing:
            issues.append(f"{path.name}: missing columns {sorted(missing)}")
            continue
        for line, row in enumerate(sheet, 2):
            cid = (row.get("comment_id") or "").strip()
            if not cid:
                issues.append(f"{path.name}:{line}: empty comment_id")
            elif cid in rows:
                issues.append(f"{path.name}:{line}: duplicate comment_id {cid}")
            else:
                rows[cid] = (row, path.name, line)
    return rows, issues


def agreement(first, second, axis):
    pairs = []
    for cid in first.keys() & second.keys():
        # Density is already populated by the same deterministic script in both
        # sheets. Compare it only where both passes actually contain a label;
        # even then this is a pipeline check, not independent human agreement.
        if not (first[cid][0].get("sentiment") or "").strip() or not (
                second[cid][0].get("sentiment") or "").strip():
            continue
        a = (first[cid][0].get(axis) or "").strip().lower()
        b = (second[cid][0].get(axis) or "").strip().lower()
        if a not in VALID[axis] or b not in VALID[axis]:
            continue
        if axis == "density" and (first[cid][0].get("sentiment") == "skip" or
                                  second[cid][0].get("sentiment") == "skip"):
            continue
        pairs.append((a, b))
    return {"n": len(pairs), "agreement": (sum(a == b for a, b in pairs) / len(pairs)
                                          if pairs else None), "kappa": kappa(pairs)}


def audit(manifest_path=MANIFEST, batches=BATCHES, second=None):
    manifest_path = Path(manifest_path)
    batches = Path(batches)
    if not manifest_path.exists():
        raise ValueError(f"Sampling manifest missing: {manifest_path}")
    fields, manifest_rows = read_csv(manifest_path)
    if not {"comment_id", "batch", "sample"} <= set(fields):
        raise ValueError("Manifest lacks comment_id, batch or sample")
    expected = {}
    issues = []
    for m in manifest_rows:
        if m["sample"] == "retest":
            continue
        cid = m["comment_id"].strip()
        if cid in expected:
            issues.append(f"manifest: duplicate pass-one comment_id {cid}")
        expected[cid] = m
    paths = sorted(p for p in batches.glob("*.csv")
                   if p.stem.startswith(("natural_", "enriched_")))
    first, sheet_issues = load_sheets(paths)
    issues.extend(sheet_issues)
    if manifest_path.resolve() == MANIFEST.resolve() and CLEAN.exists():
        canonical = {}
        with open(CLEAN, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    item = json.loads(line)
                    if item["comment_id"] in first:
                        canonical[item["comment_id"]] = item["text"]
        for cid, (row, filename, line) in first.items():
            if cid not in canonical:
                issues.append(f"{filename}:{line}: comment_id missing from clean corpus")
            elif row["text"] != canonical[cid]:
                issues.append(f"{filename}:{line}: text differs from clean corpus")
    for cid in sorted(expected.keys() - first.keys()):
        issues.append(f"missing sampled comment_id {cid}")
    for cid in sorted(first.keys() - expected.keys()):
        issues.append(f"unsampled comment_id {cid}")
    counts = Counter()
    density_counts = Counter()
    by_density = defaultdict(Counter)
    complete = 0
    for cid in sorted(first.keys() & expected.keys()):
        row, filename, line = first[cid]
        where = f"{filename}:{line}"
        if Path(filename).stem != expected[cid]["batch"]:
            issues.append(f"{where}: batch differs from manifest for {cid}")
        sent = (row.get("sentiment") or "").strip().lower()
        dens = (row.get("density") or "").strip().lower()
        lang = (row.get("base_lang") or "").strip().lower()
        if not sent:
            counts["unannotated"] += 1
            continue
        if sent not in VALID["sentiment"]:
            issues.append(f"{where}: invalid sentiment {sent!r}")
            continue
        if sent == "skip":
            counts["skip"] += 1
            complete += 1
            continue
        if dens not in VALID["density"] or lang not in VALID["base_lang"]:
            issues.append(f"{where}: invalid density/base_lang {dens!r}/{lang!r}")
            continue
        counts[sent] += 1
        density_counts[dens] += 1
        by_density[dens][sent] += 1
        complete += 1
    result = {"expected": len(expected), "found": len(first), "complete": complete,
              "unannotated": counts["unannotated"], "usable": sum(counts[x] for x in SENTIMENT[:3]),
              "sentiment": {x: counts[x] for x in SENTIMENT},
              "density": {x: density_counts[x] for x in DENSITY},
              "sentiment_by_density": {d: dict(by_density[d]) for d in DENSITY},
              "issues": issues, "batch_files": [p.name for p in paths]}
    if second:
        second_rows, second_issues = load_sheets([Path(p) for pat in second
                                                  for p in glob.glob(str(pat))])
        result["second_issues"] = second_issues
        result["inter_annotator"] = {axis: agreement(first, second_rows, axis)
                                      for axis in ("sentiment", "density")}
    retest_paths = sorted(batches.glob("retest_*.csv"))
    if retest_paths:
        retest, retest_issues = load_sheets(retest_paths)
        result["retest_issues"] = retest_issues
        # Both density columns were populated by the same algorithm before any
        # annotator saw the sheets. Their agreement is tautological.
        result["intra_annotator"] = {"sentiment": agreement(first, retest, "sentiment")}
    result["ready"] = complete == len(expected) and not issues and bool(expected)
    return result


def markdown(r):
    lines = ["# Sprint 3 corpus audit", "", "Generated by `python -m src.eval.corpus_audit`.", "",
             f"- Sampled comments: {r['expected']}", f"- Complete annotations: {r['complete']}",
             f"- Unannotated: {r['unannotated']}", f"- Usable (excluding skip): {r['usable']}",
             f"- Ready to finalize: {'yes' if r['ready'] else 'no'}", "",
             "## Label distribution", "", "| Axis | Label | Count |", "|---|---|---:|"]
    for axis in ("sentiment", "density"):
        for label, count in r[axis].items():
            lines.append(f"| {axis} | {label} | {count} |")
    lines += ["", "## Agreement", ""]
    for key, title in (("intra_annotator", "Retest by the same annotator"),
                       ("inter_annotator", "Independent second annotator")):
        if key not in r:
            lines.append(f"- {title}: unavailable")
            continue
        for axis, value in r[key].items():
            k = "n/a" if value["kappa"] is None else f"{value['kappa']:.3f}"
            caution = " (too few pairs for a reliability claim)" if value["n"] < 30 else ""
            lines.append(f"- {title}, {axis}: n={value['n']}, Cohen's κ={k}{caution}")
    lines += ["", "A retest by one person measures intra-annotator consistency. "
              "It cannot establish inter-annotator agreement. Density values in the existing "
              "batch sheets are prefilled by one deterministic algorithm, so matching density "
              "values do not measure human agreement.", "", "## Problems", ""]
    if r["issues"]:
        lines += [f"- {problem}" for problem in r["issues"][:30]]
        if len(r["issues"]) > 30:
            lines.append(f"- ... and {len(r['issues']) - 30} more")
    else:
        lines.append("No structural or label errors found. Incomplete rows are counted above.")
    return "\n".join(lines) + "\n"


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", type=Path, default=MANIFEST)
    p.add_argument("--batches", type=Path, default=BATCHES)
    p.add_argument("--second", nargs="+", help="independent annotator CSV(s)")
    p.add_argument("--out", type=Path, default=REPORT)
    p.add_argument("--strict", action="store_true", help="fail if corpus is not complete")
    args = p.parse_args()
    result = audit(args.manifest, args.batches, args.second)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(markdown(result), encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k != "batch_files"},
                     ensure_ascii=False, indent=2))
    print(f"Report: {args.out}")
    if args.strict and not result["ready"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
