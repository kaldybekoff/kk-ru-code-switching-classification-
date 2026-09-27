# Kazakh–Russian Code-Switched Text Classification

**Academic Beta Career 2026**
Student: Yeskendir Kaldybek (ID 230103282) · Mentor: Ayaulym Parmash

## 1. Research overview

- **Research problem:** Kazakhstani digital text routinely mixes Kazakh and Russian within the same sentence (code-switching), but almost all NLP tooling is built and benchmarked on monolingual corpora — its real-world performance on Kazakhstani text is largely uncharacterized.
- **Research question:** How do TF-IDF, fastText, and multilingual transformer embeddings compare on Kazakh–Russian code-switched text classification, and does explicit code-switch-aware preprocessing improve performance?
- **Hypotheses:** H1 — transformers win overall; H2 — their advantage shrinks as code-switch density rises; H3 — fastText degrades less than TF-IDF on high-density text (subword n-grams). H2 is the contribution; see [docs/research_framing.md](docs/research_framing.md).
- **Classification task:** 3-class sentiment (positive / negative / neutral). Chosen in Sprint 1 — topic labels would be confounded with the source channel, which would invalidate the density-stratified analysis. Rationale in [docs/research_framing.md](docs/research_framing.md).

## 2. Team & division of labor

This is a joint project with a second Beta Career research assistant. Data collection/annotation (weeks 1–6) is nominally shared; from week 7 the work splits as below.

**Sprint 1 was executed solo**, so pilot annotation reliability is measured as *intra*-annotator agreement (two passes, one day apart) rather than inter-annotator kappa.

| | Yeskendir (this repo) | Second assistant |
|---|---|---|
| Focus | Models & comparative experiments | Data & preprocessing |
| Weeks 7–8 | Multilingual transformer representation | Standard vs. code-switch-aware preprocessing |
| Weeks 9–10 | Comparative model experiments, significance testing | Preprocessing experiments across density groups |
| Weeks 11–12 | Model/error analysis by density group | Linguistic error-pattern analysis |

## 3. Repository structure

```
├── data/
│   ├── raw/              # raw pulled comments (not committed — see .gitignore)
│   ├── interim/          # cleaned, deduplicated, anonymized
│   └── labeled/          # final annotated dataset + label-distribution reports
├── annotation/
│   ├── guideline.md      # label taxonomy + code-switch density scale
│   └── pilot_log.md      # inter-annotator agreement rounds
├── src/
│   ├── collect/          # YouTube Data API / Playwright pull scripts
│   ├── preprocess/       # cleaning, anonymization, code-switch screening
│   ├── features/         # TF-IDF, fastText, transformer embedding extraction
│   ├── models/           # baseline + transformer classifiers
│   └── eval/             # cross-validation, significance testing, error analysis
├── notebooks/            # exploratory analysis, UMAP visualization
├── docs/                 # research framing, sampling design, setup notes
├── reports/              # sprint write-ups, figures, final report draft
├── requirements.txt
└── README.md
```

## 4. Environment & tech stack

- **Language:** Python 3.11+
- **Core libraries:** `scikit-learn`, `gensim` (fastText — the official wheel has no cp313 build), `transformers` (XLM-R), `pandas`, `numpy`
- **Data collection:** `google-api-python-client` (YouTube Data API v3), `playwright` (for sources without an API)
- **Stats/eval:** `scipy` (McNemar's test), `statsmodels`
- **Visualization:** `matplotlib`/`seaborn`, `umap-learn`
- **Repo hygiene:** raw/interim data and API keys stay out of version control (`.gitignore` covers `data/raw/`, `.env`)

Install and run instructions: [docs/setup_notes.md](docs/setup_notes.md).

## 5. Sprint roadmap

### Sprint 1 (Weeks 1–2) — Literature Review & Research Framing
- [x] Set up repo, environment, and YouTube Data API pull script
- [x] Research framing: RQ, hypotheses H1–H3, classification task, evaluation protocol — [docs/research_framing.md](docs/research_framing.md)
- [x] Draft annotation guideline v0.1, two axes (sentiment + density) — [annotation/guideline.md](annotation/guideline.md)
- [x] Candidate source list — [data/sources.csv](data/sources.csv)
- [x] Pilot tooling: anonymized batch builder + Cohen's kappa — [src/preprocess/make_pilot.py](src/preprocess/make_pilot.py), [src/eval/kappa.py](src/eval/kappa.py)
- [x] Literature review with research gap — [reports/research_report.md](reports/research_report.md), Part I
- [x] Sprint 1 video report script — [reports/video/sprint1_script.md](reports/video/sprint1_script.md)
- [ ] Verify the ⚠ fact-checks listed at the end of the literature review
- [x] Collected 7,832 raw comments; screened two sampling frames and selected sources by measured code-switch rate — [reports/research_report.md](reports/research_report.md), Part II
- [x] Pilot batch built (100 comments, oversampled to 50% screened-mixed)
- [ ] Annotate pass1 / pass2 and log kappa in [annotation/pilot_log.md](annotation/pilot_log.md)
- [ ] Record and submit the sprint 1 video report

### Sprint 2 (Weeks 3–4) — Dataset Preparation & Annotation

> *Collect and clean the raw YouTube corpus, removing irrelevant, duplicated and spam
> content while anonymizing personal information. Leverage LLMs for automated annotation
> against the established guideline, with spot-checks and auditing of ambiguous cases.*

**Collection** — 185 videos, ~1,560 of 10,000 daily quota units
- [x] `discover_videos` found 172 videos; 13 supplied by hand → [data/videos.csv](data/videos.csv)
- [x] Dropped `Altyn Bala TV` (kids' cartoon, comments disabled)
- [x] **45,832** new comments pulled with `--order time`; 51,425 total with Sprint 1
- [x] Per-source code-switch rates re-measured → [data/sources.csv](data/sources.csv). Best: TARTARIA FILMS 13.8%; worst: Информбюро 31 at 2.7%
- [x] No source dropped at the 5% line — rationale in [docs/sampling_design.md](docs/sampling_design.md) §4b

**Cleaning, deduplication, spam, anonymization** — 51,425 → **43,363**
- [x] Rules built from profiling the raw data, not from a template — [src/preprocess/textnorm.py](src/preprocess/textnorm.py)
- [x] Pipeline + audit trail — [src/preprocess/build_corpus.py](src/preprocess/build_corpus.py), `data/interim/corpus_report.md`
- [x] 7,846 removed: 4,766 too short, 2,207 emoji-only, 269 exact dupes, 245 over-long, 199 timestamps, 108 cross-video copypasta, 43 reordered dupes, 9 promo
- [x] Author names dropped on load; URLs → `<URL>`, mentions → `<USER>`
- [x] 216 romanized comments held aside with documented rationale

**Density metric — deterministic, not model-assigned**
- [x] Document-level bootstrap → char n-gram LMs → borrowing detection — [src/preprocess/density.py](src/preprocess/density.py)
- [x] Validated against the guideline's worked examples; found and fixed **three arithmetic errors** in them → guideline v0.3
- [x] True corpus distribution: mono 74.1% / low 16.4% / med 8.1% / high 1.3%
- **Why not the LLM:** H2 asks whether a multilingual transformer degrades as mixing rises. An LLM assigning the buckets would share that architecture's blind spots, correlating the stratification variable with the effect under test. See the module docstring.

**Sampling** — 1,000 + 100 retest
- [x] Stratified on the **measured** density, not the Sprint 1 screen — [src/preprocess/make_batches.py](src/preprocess/make_batches.py)
- [x] Balanced buckets: **213 high / 213 med / 212 low / 212 mono** + 150 uniform natural probe
- [x] Inclusion probabilities recorded for reweighting; 196 videos, largest ~1%

**LLM annotation + audit**
- [x] Prompt generator — [src/annotate/llm_batches.py](src/annotate/llm_batches.py): 28 self-contained prompts, sentiment + base_lang + confidence + note
- [x] Reply ingestion with strict validation — [src/annotate/llm_ingest.py](src/annotate/llm_ingest.py)
- [x] Audit queue = every low-confidence row + a 12% random slice of high-confidence ones (auditing only what the model doubted measures self-knowledge, not accuracy)
- [x] Keyboard tool for the human audit pass — [src/annotate/cli.py](src/annotate/cli.py)
- [ ] Run the 28 prompts through the LLM, save replies, ingest
- [ ] Audit the queue by hand
- [ ] `audit_report` → LLM-vs-human agreement **broken down by density bucket**

### Sprint 3 (Weeks 5–6) — Corpus Finalization & Baseline Models
- [ ] Finish annotation; compute intra-annotator agreement on the retest subset
- [x] Frozen stratified CV folds + held-out test set — [src/eval/folds.py](src/eval/folds.py)
- [x] TF-IDF word and char pipelines — [src/features/representations.py](src/features/representations.py)
- [x] fastText trained on the 43k unlabeled corpus (41,856-word vocab, cached in `models_cache/`)
- [x] CV harness recording per-item predictions — [src/models/run_cv.py](src/models/run_cv.py)
- [x] Metrics, confusion matrices, per-bucket breakdown — [src/eval/compare.py](src/eval/compare.py)
- [ ] Run the above on the real labels once annotation lands

### Sprint 4 (Weeks 7–8) — Transformer-Based Models
- [x] XLM-R frozen encoder + mean pooling + linear probe — [src/features/representations.py](src/features/representations.py)
- [x] Stratified k-fold harness shared by all representations — [src/models/run_cv.py](src/models/run_cv.py)
- [x] Density bucket carried on every row from annotation through to predictions
- [ ] Run XLM-R on the real labels

### Sprint 5 (Weeks 9–10) — Comparative Model Experiments
- [x] Representation × classifier grid over the frozen folds — [src/models/run_cv.py](src/models/run_cv.py)
- [x] Exact McNemar + Holm–Bonferroni, per density bucket, with Wilson intervals and majority-class floors — [src/eval/compare.py](src/eval/compare.py)
- [x] Length control table (guards against reading a length effect as a density effect)
- [ ] Run on the real labels and write up
- [ ] Coordinate with the second assistant on the preprocessing-variant comparison

### Sprint 6 (Weeks 11–12) — Model Analysis & Error Analysis
- [ ] Break down performance by code-switch density bucket
- [ ] Manually inspect misclassified examples for systematic patterns (short texts, spelling variation, informal language, mixed-language structures)
- [ ] Produce UMAP visualization of embedding space colored by density

### Sprint 7 (Weeks 13–15) — Research Finalization & Presentation
- [ ] Finalize report (methodology, results, discussion, limitations, future work)
- [ ] Clean and document repository (this README, reproducible run instructions)
- [ ] Build final presentation deck + live demo
- [ ] Rehearse technical Q&A

## 6. Data collection reference (candidate sources)

Prioritize first: **SMITTV** (Kazakh-language livestreams), **ZAMANDAS** (Kana Beisekeev), **Общуха** (Moldir Matzhanova) — younger, informal audiences with the most natural code-switching.

Add for volume once the pipeline works: interview/talk shows (Dinara Satzhan, Beibit Alibekov, "Честно говоря"), analytical/political ("Уақыт көрсетеді"), comedy (ДАЛАДА Podcast, Qazaq Stand Up), gaming (Q BRO, Papo4ka), general entertainment (Saspens, NNN), bilingual history channels.

## 7. Evaluation metrics

- **Classification:** accuracy, precision, recall, F1-score, confusion matrix
- **Representation comparison:** McNemar's test for pairwise significance
- **Label quality:** Cohen's kappa (inter-annotator agreement)
- **Subgroup analysis:** performance broken down by code-switch density (low/medium/high)

## 8. Deliverables checklist

- [ ] Labeled Kazakh–Russian code-switched dataset
- [ ] Reproducible comparison pipeline (TF-IDF / fastText / transformer)
- [ ] Density-stratified error analysis
- [ ] UMAP embedding visualization
- [ ] Final research report
- [ ] Final presentation + live demo
