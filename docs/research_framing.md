# Research Framing

Sprint 1 (weeks 1–2) · Yeskendir Kaldybek
Status: **decided**. Working solo this sprint; the partner's preprocessing track is
tracked separately and does not block anything below.

---

## 1. Research question

> How do TF-IDF, fastText, and multilingual transformer representations compare on
> Kazakh–Russian code-switched text classification, and does the performance gap
> between them depend on how heavily the text is code-switched?

Sub-question (owned by the second assistant): does code-switch-aware preprocessing
improve performance over standard preprocessing, and does that effect differ by
code-switch density?

## 2. Hypotheses

| ID | Hypothesis | How it gets tested |
|----|-----------|--------------------|
| H1 | Multilingual transformer representations outperform TF-IDF and fastText on overall macro-F1. | Weeks 9–10: representation × classifier grid, McNemar between top pairs. |
| H2 | The transformer's advantage **shrinks as code-switch density rises**, because its pretraining data is overwhelmingly monolingual. | Weeks 11–12: macro-F1 per density bucket; interaction between representation and bucket. |
| H3 | fastText degrades less than TF-IDF on high-density text, because subword n-grams generalize across the morphology of both languages. | Same subgroup breakdown as H2. |

H2 is the interesting claim — H1 alone would just replicate known results. The
contribution of this project is the density-stratified comparison on a language
pair that has no existing code-switched benchmark.

## 3. Classification task — decided

**Task: 3-class sentiment classification.**

| Label | Definition |
|-------|-----------|
| `positive` | Approval, praise, gratitude, affection, enthusiasm toward the video/people in it. |
| `negative` | Criticism, anger, mockery, disappointment, insult. |
| `neutral` | Factual statements, questions, on-topic remarks with no evaluative charge. |

### Why sentiment and not topic

1. **Direct precedent.** SemEval-2020 Task 9 (SentiMix) is sentiment classification
   on Hindi–English and Spanish–English code-switched text. Same task shape, so our
   setup and results are comparable to published work on other language pairs.
2. **Topic would be confounded with the source channel.** Comment topic correlates
   heavily with which channel it came from (gaming vs. political). A topic classifier
   would partly learn channel vocabulary. Worse: code-switch density *also* varies by
   channel, so the density-stratified analysis (H2, H3 — the core contribution) would
   be confounded by source. Sentiment is far more evenly distributed across channels.
3. **Annotation cost.** Sentiment needs no per-channel taxonomy and reaches usable
   inter-annotator agreement faster.

### Known risk

`neutral` will likely dominate on YouTube comments. Mitigations:
- **macro-F1 is the primary metric**, not accuracy — a majority-class baseline scores
  poorly on macro-F1 and well on accuracy.
- Report the majority-class baseline explicitly so every model is compared against it.
- If a class falls below ~10% of the corpus, oversample during annotation rather than
  rebalancing after the fact (documented in the annotation log).

## 4. Evaluation protocol

Splits are generated once and frozen. If the partner track later runs preprocessing
variants, it must load the same fold file — otherwise the two comparisons cannot be read
against each other.

### Splits
- Hold out a **stratified 15% test set**, frozen and untouched until weeks 9–10.
- On the remaining 85%: **5-fold stratified cross-validation**.
- Folds generated **once**, with a fixed seed, and committed to
  `data/labeled/folds.json` as comment-ID lists. Load that file; never regenerate.
- Stratify on the **class label**. Check afterwards that density buckets are also
  reasonably balanced across folds; if not, stratify on the (label, density) pair.

### Metrics
- **Primary:** macro-F1.
- **Secondary:** accuracy, per-class precision/recall/F1, confusion matrix.
- **Subgroup:** all of the above, broken down by density bucket (low/medium/high).
- **Baseline:** majority class, plus TF-IDF + Logistic Regression as the reference system.

### Significance testing
- **McNemar's test** for pairwise comparison of two representations on the held-out
  test set (Dietterich 1998 — the standard choice when comparing two classifiers on
  one test set).
- More than two comparisons will be made, so apply **Holm–Bonferroni** correction and
  report both raw and corrected p-values.
- Report effect size (macro-F1 difference + bootstrap CI), not only the p-value.

### Reproducibility
- Fixed seed (`RANDOM_SEED = 42`) in one shared config module.
- Every experiment run writes a row to `reports/results.csv`:
  representation, classifier, preprocessing variant, fold, metric values, seed, timestamp.

## 5. Scope boundaries

**In scope:** 3-class sentiment on Kazakh–Russian YouTube comments; three
representations; frozen-encoder linear probing for the transformer.

**Out of scope:** full fine-tuning of the transformer (compute budget), language
identification at token level as a standalone task, generation, speech.

## 6. Open decisions

- [ ] Confirm the density metric survives the pilot — if the manual buckets are hard to
      apply consistently, simplify the scale before Sprint 2.
- [ ] Target corpus size. Working figure: **2,000 labeled comments**, which supports
      ~600 per class and ~650 per density bucket. Confirm after the pilot.
- [ ] Which multilingual encoder: XLM-R base is the default. Verify Kazakh is in its
      pretraining languages and note the corpus size — that fact directly motivates H2.
