"""Sprint 2: draw the annotation sample from the clean corpus and cut it into
Excel-sized batches.

    python -m src.preprocess.make_batches --target 2400
    python -m src.preprocess.make_batches --target 2400 --batch-size 200 --retest 150

Writes to data/interim/batches/:
    natural_01.csv ...   the unbiased probe (see below)
    enriched_01.csv ...  the code-switch-enriched batches
    retest_01.csv ...    a re-draw of already-sampled comments, for pass 2
    manifest.csv         comment_id -> stratum, inclusion probability, batch
    sampling_manifest.md the human-readable record for the report

## Why the sample is not uniform

A uniform sample of these comments is ~88% monolingual by the screen. H2 and H3
compare models *within* density buckets, so a uniform sample would spend almost
the whole annotation budget on `mono` and leave the `high` bucket too small for
McNemar's test to resolve anything.

So the sample is stratified and deliberately enriched. The cost of that is that
label and density frequencies in the corpus no longer match the population.
That cost is paid honestly rather than hidden:

1. Every sampled row records the **stratum it was drawn from and its inclusion
   probability**, so any frequency claim can be reweighted by 1/p back to
   population scale.
2. A separate **natural probe** is drawn uniformly at random, with no
   enrichment at all. It is annotated with the same guideline and gives an
   unbiased estimate of how common code-switching actually is - the question
   reports/research_report.md Part II flagged as needing an answer.

## Why the annotator does not see the screen bucket

The batch CSVs carry `comment_id` and `text` only. Showing the screen's guess
would anchor the human density judgement to it, and the agreement between the
two is something this project needs to *measure* (it is how the screen's
sampling bias gets quantified), not manufacture.
"""

import argparse
import csv
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

from src.preprocess.textnorm import utf8_stdout

ROOT = Path(__file__).resolve().parents[2]
INTERIM = ROOT / "data" / "interim"
CLEAN = INTERIM / "corpus_clean.jsonl"
OUTDIR = INTERIM / "batches"

SHEET_FIELDS = ["comment_id", "text", "sentiment", "density", "base_lang", "notes"]

# How the enrichment budget is split across screen strata. `high` and `med` are
# taken exhaustively because they are the binding constraint; `low` and `mono`
# fill the rest. Tuned to the pool, not to a round number.
ENRICH_ORDER = ["high", "med", "low", "unknown", "mono"]
# Strata when sampling on the measured density metric rather than the screen.
DENSITY_ORDER = ["high", "med", "low", "mono"]


def load_clean(strata_by="density"):
    """Load the pool and attach the field the sample will be stratified on.

    `density` is the measured metric from src/preprocess/density.py and is the
    default. `screen_bucket` is the Sprint 1 heuristic, kept only so an older
    draw can be reproduced.

    The difference is not cosmetic. The screen recognises words from two closed
    lists, so it called 302 of these comments `high`; the measured metric puts
    85 of them there and finds 380 genuine ones elsewhere in the pool. Sampling
    on the screen therefore spends the budget on the wrong comments - which is
    precisely the bucket H2 depends on.
    """
    if not CLEAN.exists():
        raise SystemExit(
            "{} not found. Run: python -m src.preprocess.build_corpus".format(CLEAN))
    with open(CLEAN, encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]

    if strata_by == "density":
        from src.preprocess.density import Density
        dens = Density.load()
        for r in rows:
            m = dens.measure(r["text"])
            r["density"] = m["density"] or "mono"
            r["measured_base_lang"] = m["base_lang"]
            r["minority_share"] = m["minority_share"]
    return rows


def spread_by_video(rows, rng):
    """Round-robin across videos so a stratum is not dominated by one video.

    This matters: if the `high` bucket came mostly from one interview, a
    later 'performance differs by density' result could just as well be
    'performance differs by topic'. Profiling the Sprint 1 pull showed
    med/high already spread over 13-20 of 30 videos, and this keeps it that
    way as the pool grows.
    """
    by_vid = defaultdict(list)
    for r in rows:
        by_vid[r["video_id"]].append(r)
    for v in by_vid.values():
        rng.shuffle(v)
    order = sorted(by_vid, key=lambda v: -len(by_vid[v]))
    out = []
    while any(by_vid[v] for v in order):
        for v in order:
            if by_vid[v]:
                out.append(by_vid[v].pop())
    return out


def write_sheets(rows, prefix, batch_size, rng=None):
    """Cut rows into batch CSVs. Returns [(batch_name, rows)]."""
    OUTDIR.mkdir(parents=True, exist_ok=True)
    made = []
    for i in range(0, len(rows), batch_size):
        chunk = rows[i:i + batch_size]
        name = "{}_{:02d}".format(prefix, i // batch_size + 1)
        path = OUTDIR / (name + ".csv")
        with open(path, "w", encoding="utf-8-sig", newline="") as f:
            w = csv.writer(f)
            w.writerow(SHEET_FIELDS)
            for r in chunk:
                w.writerow([r["comment_id"], r["text"], "", "", "", ""])
        made.append((name, chunk))
    return made


def main():
    utf8_stdout()
    p = argparse.ArgumentParser()
    p.add_argument("--target", type=int, default=2400,
                   help="total comments to annotate, natural probe included")
    p.add_argument("--natural", type=int, default=400,
                   help="size of the uniform, un-enriched probe")
    p.add_argument("--retest", type=int, default=150,
                   help="comments re-issued as a second pass for intra-annotator "
                        "agreement (they are NOT extra comments - they are a "
                        "re-draw of ones already in the sample)")
    p.add_argument("--batch-size", type=int, default=200,
                   help="rows per CSV; keep it small enough to finish in one sitting")
    p.add_argument("--include-short", action="store_true",
                   help="also enrich from comments below the 5-token density "
                        "threshold. Off by default: `low` is arithmetically "
                        "unreachable for them, so they would skew the buckets")
    p.add_argument("--strata-by", default="density",
                   choices=["density", "screen_bucket"],
                   help="field to stratify on; see load_clean()")
    p.add_argument("--strata", default=None,
                   help="which screen strata to enrich from, scarcest first. "
                        "Dropping `unknown` concentrates a small budget on the "
                        "four real density levels; those comments still reach "
                        "the corpus through the natural probe")
    p.add_argument("--take-all", default="high",
                   help="comma-separated strata to take exhaustively before the "
                        "rest share what is left")
    p.add_argument("--seed", type=int, default=2026)
    args = p.parse_args()

    rng = random.Random(args.seed)
    pool = load_clean(args.strata_by)
    order = DENSITY_ORDER if args.strata_by == "density" else ENRICH_ORDER
    print("clean pool: {} comments".format(len(pool)))

    # ---- 1. natural probe: uniform, no enrichment ---------------------------
    shuffled = list(pool)
    rng.shuffle(shuffled)
    n_natural = min(args.natural, len(shuffled))
    natural = shuffled[:n_natural]
    natural_ids = {r["comment_id"] for r in natural}
    p_natural = n_natural / len(pool)

    # ---- 2. enriched sample from the remainder ------------------------------
    remainder = [r for r in pool if r["comment_id"] not in natural_ids]
    if not args.include_short:
        remainder = [r for r in remainder if r["density_eligible"]]

    field = args.strata_by
    by_stratum = defaultdict(list)
    for r in remainder:
        by_stratum[r[field]].append(r)

    budget = max(0, args.target - n_natural)
    enriched, taken = [], {}
    # Equal allocation with spillover, scarcest stratum first: each stratum may
    # take up to an even share of what is left, and whatever a scarce stratum
    # cannot fill is redistributed to the ones after it.
    #
    # The point is balanced density buckets. Taking `low` and `med`
    # exhaustively first would swallow the whole budget and leave `mono` -
    # the reference level H2 is measured against - to the natural probe alone.
    # `high` is the scarcest stratum and the one H2 stands on, so it is taken in
    # full before anything is shared out. At a smaller --target an even split
    # would otherwise cap it below what is available, which is the one loss the
    # budget must never take.
    strata = ([x.strip() for x in args.strata.split(",") if x.strip()]
              if args.strata else list(order))
    take_all = {x.strip() for x in args.take_all.split(",") if x.strip()}
    for s in strata:
        if s not in take_all:
            continue
        avail = by_stratum.get(s, [])
        want = min(len(avail), budget - len(enriched))
        picked = spread_by_video(avail, rng)[:want]
        enriched += picked
        taken[s] = len(picked)

    rest = [s for s in strata if s not in take_all]
    for i, s in enumerate(rest):
        avail = by_stratum.get(s, [])
        remaining_budget = budget - len(enriched)
        remaining_strata = len(rest) - i
        if not avail or remaining_budget <= 0:
            taken[s] = 0
            continue
        cap = -(-remaining_budget // remaining_strata)   # ceil
        want = min(len(avail), cap, remaining_budget)
        picked = spread_by_video(avail, rng)[:want]
        enriched += picked
        taken[s] = len(picked)

    rng.shuffle(enriched)

    # ---- 3. retest subset: a re-draw, not new comments ----------------------
    sampled = natural + enriched
    n_retest = min(args.retest, len(sampled))
    # Stratify the retest across screen buckets so agreement is measured on
    # mixed text too, not just on the monolingual majority.
    retest_pool = defaultdict(list)
    for r in sampled:
        retest_pool[r[field]].append(r)
    retest = []
    per = max(1, n_retest // max(1, len(retest_pool)))
    for s, rows in retest_pool.items():
        rng.shuffle(rows)
        retest += rows[:per]
    retest_ids = {r["comment_id"] for r in retest}
    leftovers = [r for r in sampled if r["comment_id"] not in retest_ids]
    rng.shuffle(leftovers)
    retest += leftovers[: max(0, n_retest - len(retest))]
    rng.shuffle(retest)

    # ---- 4. write ------------------------------------------------------------
    OUTDIR.mkdir(parents=True, exist_ok=True)
    nat_batches = write_sheets(natural, "natural", args.batch_size)
    enr_batches = write_sheets(enriched, "enriched", args.batch_size)
    ret_batches = write_sheets(retest, "retest", args.batch_size)

    probs = {}
    for s, n in taken.items():
        avail = len(by_stratum.get(s, []))
        probs[s] = (n / avail) if avail else 0.0

    with open(OUTDIR / "manifest.csv", "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["comment_id", "batch", "sample", "stratum", "screen_bucket",
                    "video_id", "n_tokens", "density_eligible", "inclusion_prob"])
        for name, chunk in nat_batches:
            for r in chunk:
                w.writerow([r["comment_id"], name, "natural", r[field],
                            r["screen_bucket"], r["video_id"], r["n_tokens"],
                            r["density_eligible"],
                            "{:.6f}".format(p_natural)])
        for name, chunk in enr_batches:
            for r in chunk:
                w.writerow([r["comment_id"], name, "enriched", r[field],
                            r["screen_bucket"], r["video_id"], r["n_tokens"],
                            r["density_eligible"],
                            "{:.6f}".format(probs.get(r[field], 0.0))])
        for name, chunk in ret_batches:
            for r in chunk:
                w.writerow([r["comment_id"], name, "retest", r[field],
                            r["screen_bucket"], r["video_id"], r["n_tokens"],
                            r["density_eligible"], ""])

    # ---- 5. record the design ------------------------------------------------
    md = []
    a = md.append
    a("# Sampling manifest")
    a("")
    a("Generated by `src/preprocess/make_batches.py` (seed {}). ".format(args.seed) +
      "Regenerate rather than editing by hand.")
    a("")
    a("| | |")
    a("|---|---|")
    a("| Clean pool | {} |".format(len(pool)))
    a("| Natural probe | {} |".format(len(natural)))
    a("| Enriched sample | {} |".format(len(enriched)))
    a("| **To annotate (pass 1)** | **{}** |".format(len(sampled)))
    a("| Retest (pass 2, re-draw) | {} |".format(len(retest)))
    a("")
    a("## Inclusion probabilities")
    a("")
    a("Reweight any population-level frequency claim by 1/p.")
    a("")
    a("Stratified on: **{}**".format(field))
    a("")
    a("| Sample | Stratum | Available | Taken | p |")
    a("|---|---|---|---|---|")
    a("| natural | (all) | {} | {} | {:.4f} |".format(len(pool), len(natural), p_natural))
    for s in strata:
        a("| enriched | {} | {} | {} | {:.4f} |".format(
            s, len(by_stratum.get(s, [])), taken.get(s, 0), probs.get(s, 0.0)))
    a("")
    a("## Videos covered")
    a("")
    for label, rows in (("natural", natural), ("enriched", enriched)):
        vids = Counter(r["video_id"] for r in rows)
        top = vids.most_common(1)
        a("- **{}**: {} comments across {} videos; largest single video "
          "contributes {:.1f}%.".format(
              label, len(rows), len(vids),
              (top[0][1] / len(rows) * 100) if rows else 0.0))
    a("")
    a("## Files")
    a("")
    for name, chunk in nat_batches + enr_batches + ret_batches:
        a("- `{}.csv` - {} rows".format(name, len(chunk)))
    a("")
    a("## Procedure")
    a("")
    a("1. Annotate `natural_*` and `enriched_*` per `annotation/guideline.md`.")
    a("2. Validate each finished batch: `python -m src.eval.validate_annotations "
      "data/interim/batches/<file>.csv`")
    a("3. At least one day after finishing, annotate `retest_*` without looking "
      "at pass 1.")
    a("4. `python -m src.eval.kappa <pass1 merged> <retest merged>` -> "
      "intra-annotator agreement. Log it in `annotation/pilot_log.md`.")
    a("")
    (OUTDIR / "sampling_manifest.md").write_text("\n".join(md), encoding="utf-8")

    print("\nnatural probe : {:>5}  (p = {:.4f})".format(len(natural), p_natural))
    print("enriched      : {:>5}".format(len(enriched)))
    for s in strata:
        if taken.get(s):
            print("  {:<8} {:>5} / {:<5}  p = {:.4f}".format(
                s, taken[s], len(by_stratum.get(s, [])), probs.get(s, 0.0)))
    print("retest        : {:>5}".format(len(retest)))
    print("\nTOTAL to annotate (pass 1): {}".format(len(sampled)))
    print("plus {} retest rows for pass 2".format(len(retest)))
    print("\n-> {}".format(OUTDIR))


if __name__ == "__main__":
    main()
