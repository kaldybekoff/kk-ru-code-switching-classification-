# Kazakh–Russian Code-Switched Text Classification

Text classification on code-switched Kazakh–Russian YouTube comments, and a controlled
comparison of three text representations across levels of code-switching density.

## Problem

Kazakhstani users routinely mix Kazakh and Russian inside a single sentence:

> Единственное мое пожелание для Молдира — чтобы это трудное время поскорее забылось, бұл қиын уақыт…

> Осы интервьюден кейін жеңілденіп кететін сияқты, ей станет легче, и Мөлдір бәрі жақсы болады

Almost all NLP tooling is built and benchmarked on monolingual corpora, so its behaviour
on text like this is uncharacterized. Code-switching NLP itself has concentrated on a
handful of language pairs — mainly Hindi–English and Spanish–English, which have
dedicated benchmarks (GLUECoS, LinCE) and a shared task (SemEval-2020 Task 9).

Kazakh–Russian is only now starting to get that attention. Recent work has released a
code-switched parallel corpus for machine translation (Borisov et al., 2025), a gold
language-identification set (Savelyev, 2026), and a 100k-review Kazakhstani sentiment
corpus benchmarking TF-IDF against mBERT, XLM-R and RemBERT (Yeshpanov, 2026). That last
one reports that transformers consistently beat classical baselines.

**What none of them report is whether that advantage survives as the text becomes
genuinely mixed.** Every published comparison on this pair is aggregate.

The pair is also structurally harder than the studied ones. Hindi–English and
Spanish–English code-switching typically crosses two scripts, or appears in romanized
form, so the surface form itself signals the switch. **Kazakh and Russian share the
Cyrillic alphabet**, and in informal writing the Kazakh-specific characters
(ә ғ қ ң ө ұ ү һ і) are frequently dropped. That free signal does not exist here.

## Research question

> How do TF-IDF, fastText and multilingual transformer representations compare on
> Kazakh–Russian code-switched text classification, and does the gap between them depend
> on how heavily the text is mixed?

### Hypotheses

| | Hypothesis |
|---|---|
| **H1** | Multilingual transformer representations perform best overall. |
| **H2** | Their advantage **shrinks as code-switching density rises** — they are pretrained on monolingual corpora and have effectively never seen mixed sentences. |
| **H3** | fastText degrades less than TF-IDF on heavily mixed text, because character n-grams absorb Kazakh's agglutinative morphology and informal spelling variation. |

H1 is a replication of Yeshpanov (2026) in a different domain. **H2 and H3 are the
contribution** — no published work on this pair breaks performance down by how heavily
the text is code-switched, and fastText is absent from existing comparisons.

## Task and data

- **Task:** 3-class sentiment classification (positive / negative / neutral), matching the
  formulation of SemEval-2020 Task 9.
- **Data:** YouTube comments from Kazakhstani channels, collected via the YouTube Data
  API v3, deduplicated and anonymized.
- **Annotation:** two independent axes per comment — a sentiment label, and a
  code-switching density bucket (`mono` / `low` / `med` / `high`) based on the share of
  minority-language tokens. Borrowings shared by both languages (интернет, видео,
  телефон) count as the base language rather than as switches.

### Source selection is measured, not assumed

Sources are screened for how much code-switching they actually contain before any
annotation effort is spent on them. Two sampling frames were compared:

| Sampling frame | Comments | Screened code-switched |
|---|---|---|
| YouTube's most-popular chart for Kazakhstan | 1,832 | **~3%** |
| Long-form interviews, podcasts, stand-up | 6,000 | **~8%** |

The popular chart is dominated by children's gaming channels whose comment sections are
almost entirely monolingual Kazakh. Long-form adult content produced roughly 2.5× the
mixed rate, and genuine intra-sentential switching rather than isolated borrowings. Both
figures are **lower bounds** — the screen only recognizes words in its closed word lists.

## Method

Three representations, one evaluation protocol:

| Representation | Rationale |
|---|---|
| **TF-IDF** + linear classifier | Reference baseline. No pretraining; expected to fragment on agglutinative morphology and spelling variation. |
| **fastText** | Subword character n-grams — built for morphologically rich languages, degrades gracefully under misspelling. |
| **Multilingual transformer** (XLM-R) | Frozen encoder + linear probe. Cross-lingual pretraining, but on monolingual data. |

**Evaluation:** stratified 5-fold cross-validation over frozen splits (generated once,
committed, never regenerated), plus a held-out test set. Primary metric **macro-F1** —
accuracy is misleading when `neutral` dominates. Pairwise significance via **McNemar's
test** with Holm–Bonferroni correction, reported alongside effect size. All metrics are
additionally broken down by density bucket.

## Repository layout

```
├── data/
│   ├── raw/              # pulled comments (gitignored - carries author names)
│   ├── interim/          # cleaned, deduplicated, anonymized (gitignored)
│   ├── labeled/          # final annotated dataset + frozen CV folds
│   ├── sources.csv       # channels with measured code-switch rates
│   └── videos.csv        # the 185 videos pulled from
├── src/
│   ├── collect/          # YouTube Data API pull, video discovery
│   ├── preprocess/       # cleaning, language filter, density metric, sampling
│   ├── annotate/         # annotation tooling and LLM prompt pipeline
│   ├── features/         # TF-IDF, fastText, transformer representations
│   ├── models/           # cross-validation harness
│   └── eval/             # folds, validation, agreement, significance testing
├── annotation/           # guideline (v0.3) and pilot log
├── docs/                 # research framing, sampling design, setup notes
├── reports/
│   ├── research_report.md   # literature review + all findings, both sprints
│   └── video/               # scripts, slides and upload notes (local only)
└── notebooks/            # exploratory analysis
```

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env      # then add a YouTube Data API v3 key
```

### Pipeline

```bash
# 1. Find long-form videos with live comment sections, ranked by comment count
python -m src.collect.discover_videos --per-source 15 --min-comments 200

# 2. Pull comments. --order time, not relevance: relevance front-loads
#    heavily-upvoted comments, which are not what people typically write.
python -m src.collect.youtube_pull --from-videos data/videos.csv --max 600 --order time

# 3. Screen a source: how much code-switching does it actually contain?
python -m src.preprocess.cs_screen --by-video --show 10

# 4. Clean, deduplicate, anonymize and screen the whole pool
python -m src.preprocess.build_corpus

# 5. Draw the stratified annotation sample + the uniform natural probe
python -m src.preprocess.make_batches --target 1000 --natural 150 --retest 100

# 6. Annotate a batch (keyboard-driven; the CSVs also open fine in Excel)
python -m src.annotate.cli data/interim/batches/enriched_01.csv

# 7. Check every finished batch before moving on
python -m src.eval.validate_annotations data/interim/batches/enriched_01.csv

# 8. Merge the annotated batches into the final dataset
python -m src.preprocess.merge_annotations data/interim/batches/*.csv

# Annotation agreement between two sheets (pass 1 vs. pass 2)
python -m src.eval.kappa data/interim/pilot_100_pass1.csv data/interim/pilot_100_pass2.csv
```

Run everything from the repository root — the scripts are invoked as modules.

### Sampling is stratified, and says so

A uniform sample of these comments is ~88% monolingual and would yield roughly 8
high-density examples out of 1,000 — far too few for the subgroup comparison H2 and H3
depend on. The corpus is therefore enriched for code-switching, and the enrichment is
recorded rather than hidden: every row carries the stratum it was drawn from and its
inclusion probability, and a separate uniform **natural probe** is annotated alongside it
to give an unbiased estimate of how common code-switching actually is. Frequency claims
come from the probe or from 1/p reweighting, never from the corpus as a whole.

Two sub-populations are held aside and reported as scoped limitations: romanized
(Latin-script) comments, where transliteration conventions carry a signal that Cyrillic
does not, and comments under 5 meaningful tokens, for which the `low` density band is
arithmetically unreachable. Full rationale and numbers in `docs/sampling_design.md`.

## Data handling

- `data/raw/` and `data/interim/` are gitignored. Raw comments carry author display
  names and are never committed.
- The preprocessing step drops author fields entirely and replaces URLs with `<URL>` and
  @mentions with `<USER>`. Nothing downstream carries identifying information.
- Only the final anonymized labeled dataset is published.

## Status

Corpus construction is in progress. Research framing, annotation guideline, source
selection, the collection pipeline, the cleaning and stratified-sampling pipeline and
the annotation validation tooling are complete. Collection is scaling toward the ~35,000
raw comments the density-stratified analysis requires; baseline models follow.

## References

Yeshpanov (2026), *100,000+ Movie Reviews from Kazakhstan* ·
Savelyev (2026), *Loanword or Switch?* ·
Borisov et al. (2025), *Low-resource MT for Code-switched Kazakh–Russian* ·
Sitaram et al. (2019), *A Survey of Code-switched Speech and Language Processing* ·
Doğruöz et al. (ACL 2021), *A Survey of Code-switching* ·
Patwa et al., *SemEval-2020 Task 9: Sentiment Analysis of Code-Mixed Tweets* ·
Khanuja et al. (ACL 2020), *GLUECoS* ·
Aguilar et al. (LREC 2020), *LinCE* ·
Bojanowski et al. (TACL 2017), *Enriching Word Vectors with Subword Information* ·
Conneau et al. (ACL 2020), *Unsupervised Cross-lingual Representation Learning at Scale* ·
Winata et al. (CALCS 2021), *Are Multilingual Models Effective in Code-Switching?* ·
Dietterich (1998), *Approximate Statistical Tests for Comparing Supervised Classification
Learning Algorithms*
