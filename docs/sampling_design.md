# Sampling design — Sprint 2

The decision deferred in Sprint 1 (`reports/research_report.md`, Part II): **stratified sampling with a
documented target distribution, or uniform sampling with a much larger corpus.**

This document is the answer, and the numbers behind it. It is written to be pasted into
the methodology section of the final report, because a stratified corpus whose sampling
is undocumented misrepresents how common code-switching actually is.

---

## 1. The decision: stratified, with a uniform probe alongside it

Two samples are drawn from the same cleaned pool:

| Sample | How | What it is for |
|---|---|---|
| **natural probe** | uniform random, no enrichment | An unbiased estimate of the real density and sentiment distribution. The only part of the corpus that may be quoted for "how common is X". |
| **enriched sample** | stratified on the **measured** density metric, equal allocation across the four buckets | Supplies enough medium- and high-density examples for H2 and H3 to be testable. |

Every enriched row records the stratum it came from and its inclusion probability, so any
frequency claim can be reweighted by 1/p back to population scale. Both samples are
annotated with the same guideline by the same annotator, in interleaved batches.

## 2. Why uniform sampling alone does not work

Measured on the full pull (51,425 raw comments from 185 long-form videos), counting
only density-eligible comments:

| | Count | Share |
|---|---|---|
| clean pool after dedup/filtering | 43,363 | — |
| density-eligible (≥ 5 tokens) | 36,371 | 100% |
| screened `mono` | 31,819 | 87.5% |
| screened `unknown` | 1,777 | 4.9% |
| screened `low` | 1,687 | 4.6% |
| screened `med` | 784 | 2.2% |
| screened `high` | 304 | **0.84%** |

H1 is an aggregate comparison and a uniform corpus would serve it. H2 and H3 are not —
they compare representations *within* a density bucket. At these proportions a uniform
sample of 1,000 comments yields roughly **8 high-density examples**. McNemar's test on
8 items cannot resolve anything; the study's actual contribution would be unmeasurable
while the aggregate replication looked fine.

So enrichment is not a convenience. It is what makes H2 and H3 answerable at all.

## 3. What enrichment costs, and how it is paid

**Cost 1 — the corpus is not a population sample.** Mitigated by the natural probe and by
recording inclusion probabilities. Any sentence in the report of the form "X% of
Kazakhstani YouTube comments are code-switched" must come from the probe, not from the
corpus as a whole.

**Cost 2 — selection used to be made by a heuristic with unknown recall.** The Sprint 1
screen recognised only words from two closed lists, so content-word switches were
invisible to it. Measuring the corpus properly showed how badly that distorted the draw:
the screen called 302 sampled comments `high`, and the deterministic metric puts 85 of
them there while finding 380 genuine ones elsewhere in the pool. The sample is therefore
stratified on the **measured** density (`src/preprocess/density.py`), and the screen is
retired to what it was always good for — deciding which channels were worth collecting
from at all.

Note this bias runs in a helpful direction for corpus size and an unhelpful one for
purity: because screen recall is below 1, part of the `mono` stratum is truly mixed, so
the human `high` count will exceed the screened `high` count. That is extra data, not a
problem — but it does mean the screen strata and the final density labels are different
things and must never be reported as if they were the same.

**Cost 3 — enrichment concentrates on whatever the screen is good at finding.** Handled
by `spread_by_video()`, which round-robins across videos inside each stratum. The
realised spread is in §4; per-video coverage is printed to `sampling_manifest.md` on
every run so it stays checkable rather than assumed.

**Cost 4 — the strata are not equally sized, so the budget has to be shared.** The
sampler takes `high` in full first — it is the scarcest stratum and the one H2 stands
on — then gives every remaining stratum an even share of what is left. Taking `low` and
`med` exhaustively instead would swallow the budget and leave `mono`, the reference level
H2 is measured against, to the natural probe alone.

Realised allocation at `--target 1000`: **302 `high`** (every one available), 183 `med`,
183 `low`, 182 `mono`, plus a 150-comment natural probe. Total 1,000, with 100 of them
re-issued as the retest pass.

The size was set from test sensitivity, not convenience. With ~20% discordant pairs
between two reasonable models, McNemar at alpha .05 / power .80 resolves a gap of ~6.2
points at n=200 per bucket and ~8.8 at n=100, against the 4-8 point gaps this literature
reports. Below roughly 200 per bucket the test is blind, so that is the floor.

The screen stratum `unknown` is not enriched at this size: it is not a density level but
a set of comments the screen could not read, and at a small budget those places are worth
more to the four real levels. Such comments still enter the corpus through the probe.

Batches are shuffled before being cut, so each one is a proportional miniature of the
whole design (~50 `high` per 150 rows). Annotation can therefore stop after any batch and
still leave a balanced corpus - only smaller.

## 4. How much raw data this required — and what was collected

Screened `high` is 0.84% of the density-eligible pool, so reaching ~250 high-density
comments needed roughly 30,000 usable comments. The Sprint 1 pull of 6,000 was about one
sixth of that.

Collected in Sprint 2:

| | |
|---|---|
| Videos | 185 (13 supplied by hand, 172 found by `discover_videos.py`) |
| Raw comments pulled | 45,832 new, 51,425 total with Sprint 1 |
| Clean pool | 43,363 |
| Density-eligible | 36,371 |
| Screened `high` available | **304** |
| Screened `med` available | **784** |

Both clear the ≥250 needed per bucket, so annotation was not started until they did.

Quota was never the obstacle: `commentThreads.list` costs 1 unit per 100 comments, so the
whole pull cost ~460 units of the 10,000 daily budget, plus ~1,100 for video discovery.
The obstacle was finding long-form videos with live comment sections.

**Video spread.** The enriched sample draws on 190+ distinct videos with no single video
contributing more than ~1% of it; the exact figures for the current draw are in
`data/interim/batches/sampling_manifest.md`. The risk that
"performance differs by density" is really "performance differs by topic" is therefore
low by construction, not by assumption.

## 4b. Why no source was dropped at the 5% threshold

Sprint 1 proposed dropping sources below ~5% screened-mixed. Measured on the full pool,
three land just under it — Janar Baisemiz 4.9%, Stand Up Astana 4.5%, CMN KZ 4.4% — and
together they still supply 148 `med`+`high` comments.

They were kept. The 5% rule was a rule for deciding **where to spend collection effort**
before the data existed. Now that it does, the stratified sampler already concentrates
annotation on mixed comments wherever they come from, so excluding a source would discard
usable high-density examples for no gain. The measured rate is a lower bound in any case,
and a 4.5% vs 5.0% difference is well inside its error.

The one source actually rejected is `Altyn Bala TV` (kids' cartoons, comments disabled),
on the same grounds as the Sprint 1 gaming channels.

Per-source rates are recorded in `data/sources.csv`. Highest measured: TARTARIA FILMS
13.8% (214 `med`+`high`, the single most productive source), Qazaq Stand Up 11.8%,
kana beisekeyev 9.3%. Lowest: Информбюро 31 at 2.7% — short news clips rather than
long-form conversation, which matches the Sprint 1 finding about content format.

## 5. Ordering is a sampling decision

`youtube_pull.py --order relevance` (the Sprint 1 default) returns YouTube's own ranking,
which front-loads heavily-liked comments. Those are not representative of what people
write — they are what people upvote, which skews long, quotable and emotionally marked.

The Sprint 2 corpus pull uses `--order time`. Sprint 1 batches keep `relevance` so they
stay reproducible, and the two are kept in separate raw files so the difference can be
checked rather than assumed.

## 6. Comments that are excluded, and why

Decided during Sprint 2 cleaning (`src/preprocess/textnorm.py`), all counted in
`data/interim/corpus_report.md`:

| Excluded | Reason |
|---|---|
| no letters at all (emoji/punctuation only) | Density is undefined with zero word tokens; the guideline already calls these `skip`. |
| timestamp pointers (`9:36 🤣`) | A reference into the video, not an opinion. |
| URL **and** a promotional keyword | Spam. The URL alone is not enough — concert-ticket links appear in genuine comments. |
| cross-video copypasta (same text under ≥3 videos) | Bot or campaign text, not an individual's writing. |
| exact and token-reordered duplicates | Inflates apparent agreement and leaks between CV folds. |
| over 1,500 characters | Essay-length outliers that would dominate a TF-IDF vocabulary. |

Deliberately **not** excluded on a keyword: anything matching `подпис*`. Profiling the
7,832 Sprint 1 comments found 36 matches, nearly all genuine praise ("Подписалась на
Александра", "Лайк, подписка"). A generic spam list would have deleted real positive
examples — which is why the filter was built from this data rather than from a template.

## 7. Two sub-populations held aside

**Romanized text.** 5.5% of raw comments contain no Cyrillic, and some are genuine
code-switching written in Latin script:

> *Mne kajetsya Aldik otirik istoryalar aytatin syaqti*
> (ru "мне кажется" + kk "өтірік историялар айтатын сияқты")

This is the same phenomenon in a different orthography — and a different regime for the
research question. The project's argument is that Kazakh and Russian share the Cyrillic
alphabet so the script gives no signal; in romanized text the transliteration conventions
themselves carry signal. Mixing the two would confound the thing being measured, and
there is not enough romanized data to study it separately.

Decision: routed to `data/interim/corpus_romanized.jsonl`, excluded from the corpus,
reported as a scoped limitation and as future work.

**Comments shorter than 5 meaningful tokens.** Density is a ratio, so `low`
(minority_share ≤ 0.20) needs at least 5 tokens to be reachable — with 4 tokens one
switch is already 0.25, which is `med`. The artifact is visible in the Sprint 1 data:
**0.0%** of screened-`low` comments have ≤5 words, against 9.4% of `mono` and 15.2% of
`high`.

Left unaddressed this would be a direct threat to H2: if `high` comments are
systematically shorter than `mono` ones, a drop in performance at high density could be
explained entirely by length rather than by mixing.

Decision: short comments stay in the corpus and are labeled normally (they are listed in
the week 11–12 error analysis as a category of interest), but they are flagged
`density_eligible = False` and excluded from the density-stratified comparison. Length
distribution per density bucket is reported alongside the results either way.

## 8. Reproducing the sample

```bash
python -m src.collect.discover_videos --per-source 15 --min-comments 200
python -m src.collect.youtube_pull --from-videos data/videos.csv --max 600 --order time
python -m src.preprocess.build_corpus
python -m src.preprocess.make_batches --target 1000 --natural 150 --retest 100 \n    --batch-size 150 --strata high,med,low,mono
```

`make_batches` is seeded (`--seed 2026`). The seed, the inclusion probabilities and the
per-video coverage are written to `data/interim/batches/sampling_manifest.md` on every
run. Regenerating the sample after annotation has started invalidates it — if the pool
grows, draw a **supplementary** batch instead, with its own manifest.
