# Pilot Annotation Log

One entry per pilot round. The point of the pilot is to break the guideline on 100
comments instead of on 2,000.

**Solo mode:** agreement here is *intra*-annotator — the same person, two passes, at
least a day apart. That is test-retest reliability, not inter-annotator kappa. Label it
as such wherever it is reported.

**Targets:** sentiment kappa ≥ 0.60, density kappa ≥ 0.70.
Below target → fix the guideline, bump the version, run a **fresh** batch.

---

## Round 1 — <date>

- Guideline version: v0.1
- Batch: `pilot_100_pass1.csv` / `pilot_100_pass2.csv` (seed 42)
- Passes: pass1 <date> / pass2 <date>
- Time spent: <...>

### Results

| Axis | Compared | Agreed | Cohen's kappa (intra) | Target met? |
|------|----------|--------|---------------|-------------|
| sentiment | | | | |
| density | | | | |

### Most common confusions

| Pair | Count | Root cause | Guideline fix |
|------|-------|-----------|---------------|
| | | | |

### Observations from the raw data

- Share of comments marked `skip`: <...>  → tells us the pull-to-label ratio for Sprint 2
- Rough label distribution: pos <...> / neg <...> / neu <...>
- Rough density distribution: mono <...> / low <...> / med <...> / high <...>
- **Share of non-`mono` comments: <...>** → if this is low, the current channel mix will
  not produce enough medium/high examples and the source list needs rebalancing toward
  the priority-1 informal channels **before** the main pull.

### Decisions

- [ ] Guideline changes made:
- [ ] New version number:
- [ ] Another round needed? yes / no

---

## Round 2 — <date>

(copy the block above)
