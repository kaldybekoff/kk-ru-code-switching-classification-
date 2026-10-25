# Sprint 4 — multilingual transformer representation

The week 10 comparison supersedes the earlier significance calculation and density
tables below. It fits fastText on training videos only and uses video-level
permutation of paired predictions on the AI review. [Current results](sprint5_status.md)

## Implementation and evaluation

- Frozen `FacebookAI/xlm-roberta-base` encoder at commit
  `e73636d4f797dec63c3081bb6ed5c7b0bb3f2089`, mean pooling over
  non-padding tokens, L2 normalization, then the same class-balanced logistic
  regression (`C=1`, seed 2026) as the baselines. No encoder fine-tuning.
- The same cleaned comment text is tokenized with a 256-token limit. In the
  557-row silver training corpus, 7 comments exceed this limit; none of the
  98 blind AI review comments do. See `sprint4_corpus_audit.md`.
- All four representations use the frozen video-grouped five-fold split:
  475 out-of-fold predictions and 82 reserved silver-label test rows. The
  video-disjoint blind AI review trains on the 557 silver rows and evaluates
  the same 98 reviewed comments as the Sprint 3 baselines. It does not use
  the reserved 82 silver-label rows as a second test.
- Re-running the three baselines produced exactly the same 1,425 CV and 294
  review predictions as Sprint 3. The comparison uses paired predictions
  with identical IDs, reference labels, density, and fold assignments.

## Results

| Representation | Silver CV macro-F1 (n=475) | Blind AI review macro-F1 (n=98) | Review accuracy |
|---|---:|---:|---:|
| Word TF-IDF | 0.432 | 0.462 | 0.480 |
| Character TF-IDF | 0.561 | **0.552** | 0.571 |
| fastText | **0.593** | 0.532 | 0.571 |
| Frozen XLM-R | 0.505 | 0.371 | 0.398 |

The transformer does not lead this comparison. On the review set it predicts
only 4 of 29 negative comments correctly (recall 0.138), and its accuracy is
below the positive-majority floor of 0.459. On silver CV, fastText exceeds
XLM-R by 0.088 macro-F1. The paired accuracy difference between fastText and
XLM-R passes Holm correction across the CV comparison family (adjusted
McNemar p=0.0369). This test concerns paired correctness, not a macro-F1
confidence interval, and CV predictions are dependent through overlapping
training folds. Treat this as an exploratory comparison.

The density breakdown does not show a monotone XLM-R advantage or a monotone
decline in its margin. The review has only 7 high-density examples, so it
cannot settle H2. Full precision, recall, class F1, confusion matrices and
density tables are in `sprint4_cv_metrics.md` and `sprint4_review_metrics.md`;
paired comparisons are in `sprint4_cv_comparison.md` and
`sprint4_review_comparison.md`.

## Limits and next gate

The sentiment training labels are silver predictions from a sentiment-tuned
XLM-R model. That model shares an architecture family with the evaluated
encoder, which can bias silver-label agreement toward the transformer. The
review labels were assigned by Codex AI without seeing the local predictions,
but another annotation model was inspected on this review during Sprint 3.
Neither source is a human-gold test. The original human annotation remains
incomplete (24 of 1,000 pass-one labels when last audited). Conclusions about
real Kazakh–Russian sentiment performance require the human corpus and a
fresh video-disjoint human test.

## Reproduce

From the repository root, with `requirements.txt` installed:

```bash
python -m src.eval.transformer_audit
python -m src.models.run_cv --labeled data/labeled/corpus_silver_train.csv --folds data/labeled/folds_silver.json --reps tfidf_word tfidf_char fasttext xlmr --out data/labeled/predictions_sprint4_silver_cv.csv
python -m src.models.run_review_holdout --reps tfidf_word tfidf_char fasttext xlmr --out data/labeled/predictions_sprint4_ai_review.csv
python -m src.eval.baseline_report --pred data/labeled/predictions_sprint4_silver_cv.csv --out reports/sprint4_cv_metrics.md --title "Sprint 4 representation metrics: silver CV"
python -m src.eval.baseline_report --pred data/labeled/predictions_sprint4_ai_review.csv --protocol blind_ai_review --out reports/sprint4_review_metrics.md --title "Sprint 4 representation metrics: blind AI review"
python -m src.eval.compare --pred data/labeled/predictions_sprint4_silver_cv.csv --out reports/sprint4_cv_comparison.md
python -m src.eval.compare --pred data/labeled/predictions_sprint4_ai_review.csv --out reports/sprint4_review_comparison.md
```

Large model files and unsampled comment data remain local in `models_cache/`
and `data/interim/`. Reproducing the fastText baseline requires the original
unsampled clean corpus and sampling manifest; the pinned XLM-R checkpoint is
downloaded on first use.
