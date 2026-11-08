# Sprint 5 — comparative model experiments

## Evaluation

The grid pairs word TF-IDF, character TF-IDF, fastText and frozen multilingual XLM-R
with class-balanced logistic regression and linear SVM (`C=1`, seed 2026).
TF-IDF vocabularies are fitted within each training split. XLM-R uses mean pooling
and a 256-token limit. fastText fits on unsampled comments from the training
videos of each split. Its five CV training pools contain 16,712–17,260 such
comments and its final training pool contains 25,136.

The frozen five-fold video-grouped CV uses 475 out-of-fold silver comments and
reserves 82 more. The final models train on all 557 silver comments and are compared on
98 usable comments from 75 other videos. Training sentiment comes from a Cardiff
sentiment model. Review sentiment comes from Codex. An alternative labeler was
checked against this review during Sprint 3. Treat the comparison as exploratory.

## Results

| Representation | Silver CV macro-F1, logreg / SVM | Blind AI review macro-F1, logreg / SVM |
|---|---:|---:|
| Word TF-IDF | 0.432 / 0.402 | 0.462 / 0.426 |
| Character TF-IDF | 0.561 / 0.533 | **0.552 / 0.537** |
| fastText | **0.574 / 0.571** | 0.478 / 0.473 |
| Frozen XLM-R | 0.505 / 0.524 | 0.371 / 0.352 |

The CV leader is fastText and the review leader is character TF-IDF. With logistic
regression, the review gap between them is 0.074 macro-F1 with a 95% video
bootstrap interval of −0.018 to +0.162. Character TF-IDF reaches 0.571 accuracy on
the review. Frozen XLM-R with logistic regression reaches 0.398, compared with a
0.459 majority-class accuracy reference. H1 does not receive support in the
tested frozen-encoder setup.

The primary review analysis compares paired differences in macro-F1. It swaps
predictions at the video level in 10,000 seeded two-sided permutations. Holm
correction covers six representation pairs per classifier and four pairs between
classifiers, for 16 tests. The 95% intervals use 10,000 video-level bootstrap
resamples. McNemar scores summarize paired accuracy as unadjusted diagnostics.
CV is descriptive because training folds overlap.

**Zero of 16 comparisons pass Holm correction at 0.05.** The lowest adjusted
p-value is 0.0544 for character TF-IDF versus XLM-R with SVM. The earlier positive
result for fastText versus XLM-R with SVM changes from adjusted p=0.0336 to
0.1764 once fastText excludes text from evaluation videos.

For the subgroup tables, both cohorts use one deterministic count of Kazakh and
Russian words. The analysis retains 441 CV comments (54 high) and 87 review
comments (12 high) with at least five counted words. The all-comment metrics retain
the original 475 and 98 comments. The subgroup sizes, uncertain word assignments,
and differences in comment lengths limit conclusions about H2 and H3.

## Reproduce

Run from the root with the existing cleaned corpus and sample manifest. In
PowerShell, set `$env:PYTHONHASHSEED = '2026'` before starting Python to reproduce
fastText training. The local XLM-R checkpoint is pinned at
`e73636d4f797dec63c3081bb6ed5c7b0bb3f2089`. Set
`$env:HF_HUB_OFFLINE = '1'` when using the populated local cache.

```powershell
$env:PYTHONHASHSEED = '2026'
python -m src.models.run_cv --fasttext-training-videos --labeled data/labeled/corpus_silver_train.csv --folds data/labeled/folds_silver.json --reps tfidf_word tfidf_char fasttext xlmr --clf logreg svm --out data/labeled/predictions_sprint5_grouped_silver_cv.csv
python -m src.models.run_review_holdout --fasttext-training-videos --reps tfidf_word tfidf_char fasttext xlmr --clf logreg svm --out data/labeled/predictions_sprint5_grouped_ai_review.csv
python -m src.eval.sprint5
python -m src.eval.baseline_report --pred data/labeled/predictions_sprint5_grouped_silver_cv.csv --out reports/sprint5_cv_metrics.md --title "Sprint 5 silver CV metrics"
python -m src.eval.baseline_report --pred data/labeled/predictions_sprint5_grouped_ai_review.csv --protocol blind_ai_review --out reports/sprint5_review_metrics.md --title "Sprint 5 blind AI review metrics"
python -m unittest discover -s tests -v
```

The score reports use the frozen `density_word_counts.csv`. To regenerate those
counts with the local density lexicon, run `python -m src.eval.density_groups`.
The historical prediction files keep the previous fastText results.
The rerun matched every TF-IDF and XLM-R prediction from those files.
[Paired tests and subgroup tables](sprint5_comparison.md) ·
[CV metrics](sprint5_cv_metrics.md) · [Review metrics](sprint5_review_metrics.md)

The experiment completes weeks 9–10 under the adopted AI-label protocol.
[Week-by-week audit](week10_audit_ru.md)

One human subsequently annotated the 100 review comments. The eight systems used
the same 557 silver-labeled training items and were evaluated on 96 usable human
labels. Character TF-IDF with logistic regression led with macro-F1 0.564.
None of the 16 paired comparisons passed Holm correction at 0.05.
Cohen's κ between the human and blind Codex reviewers was 0.533 across all 100
including skips. On the 96 usable human labels, Cardiff matched 62 with κ = 0.455.
[Human review results](sprint5_human_review_ru.md)
