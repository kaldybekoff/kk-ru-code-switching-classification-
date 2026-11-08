# Sprint 5 comparative experiments

Four frozen representations × two class-balanced linear classifiers. The same examples and labels are paired within each evaluation set. All systems use C=1. In the corrected experiment fastText fits unsampled text from the training videos separately for each split.

CV uses silver labels from a sentiment model; the video-disjoint review uses blind Codex AI labels. Neither set is human gold. The review was inspected with another annotation model during Sprint 3, so even its inferential results are exploratory. Density-enriched sampling means scores do not estimate natural YouTube prevalence.

CV: 475 comments, 101 videos; review: 98 comments, 75 videos.

### Silver out-of-fold CV (descriptive)

| Representation | Classifier | Macro-F1 | Accuracy | Negative recall |
|---|---|---:|---:|---:|
| tfidf_word | logreg | 0.432 | 0.436 | 0.379 |
| tfidf_word | svm | 0.402 | 0.408 | 0.300 |
| tfidf_char | logreg | 0.561 | 0.564 | 0.493 |
| tfidf_char | svm | 0.533 | 0.539 | 0.457 |
| fasttext | logreg | 0.574 | 0.583 | 0.407 |
| fasttext | svm | 0.571 | 0.589 | 0.350 |
| xlmr | logreg | 0.505 | 0.522 | 0.300 |
| xlmr | svm | 0.524 | 0.556 | 0.250 |

Density analysis includes 441 of 475 comments with at least five words assigned to Kazakh or Russian. Both sets use the same deterministic word counts. Median length uses the corpus word-token count.

| Density | n | Median length | Majority accuracy | tfidf_word | tfidf_char | fasttext | xlmr |
|---|---:|---:|---:|---:|---:|---:|---:|
| mono | 162 | 12 | 0.407 | 0.466 | 0.600 | 0.618 | 0.520 |
| low | 130 | 19 | 0.346 | 0.486 | 0.555 | 0.570 | 0.490 |
| med | 95 | 10 | 0.400 | 0.313 | 0.517 | 0.542 | 0.429 |
| high | 54 | 11 | 0.389 | 0.454 | 0.599 | 0.464 | 0.477 |

Density table uses logistic regression, the prespecified representation comparison. Cells with fewer than 20 examples are descriptive only.

### Blind AI review

| Representation | Classifier | Macro-F1 | Accuracy | Negative recall |
|---|---|---:|---:|---:|
| tfidf_word | logreg | 0.462 | 0.480 | 0.483 |
| tfidf_word | svm | 0.426 | 0.449 | 0.414 |
| tfidf_char | logreg | 0.552 | 0.571 | 0.345 |
| tfidf_char | svm | 0.537 | 0.551 | 0.414 |
| fasttext | logreg | 0.478 | 0.551 | 0.103 |
| fasttext | svm | 0.473 | 0.531 | 0.138 |
| xlmr | logreg | 0.371 | 0.398 | 0.138 |
| xlmr | svm | 0.352 | 0.418 | 0.034 |

Density analysis includes 87 of 98 comments with at least five words assigned to Kazakh or Russian. Both sets use the same deterministic word counts. Median length uses the corpus word-token count.

| Density | n | Median length | Majority accuracy | tfidf_word | tfidf_char | fasttext | xlmr |
|---|---:|---:|---:|---:|---:|---:|---:|
| mono | 24 | 8.5 | 0.625 | 0.378 | 0.536 | 0.503 | 0.451 |
| low | 30 | 11.5 | 0.433 | 0.479 | 0.506 | 0.436 | 0.365 |
| med | 21 | 14 | 0.476 | 0.434 | 0.396 | 0.329 | 0.264 |
| high | 12 | 7.5 | 0.333 | 0.367 | 0.653 | 0.500 | 0.283 |

Density table uses logistic regression, the prespecified representation comparison. Cells with fewer than 20 examples are descriptive only.

### Paired comparisons on the blind AI review

98 comments in 75 videos. Difference is first system minus second. 95% percentile intervals resample videos (10000 draws; seed 2026). The two-sided macro-F1 permutation swaps predictions for whole videos; Holm correction covers all 16 comparisons below. Exact McNemar p-values test paired accuracy and are unadjusted diagnostics, not the primary test.

| First | Second | Δ macro-F1 | 95% video CI | permutation p | Holm p | McNemar n01/n10 | McNemar p |
|---|---|---:|---:|---:|---:|---:|---:|
| tfidf_word+logreg | tfidf_char+logreg | -0.090 | [-0.196, +0.012] | 0.1064 | 1.0000 | 17/8 | 0.1078 |
| tfidf_word+logreg | fasttext+logreg | -0.017 | [-0.138, +0.094] | 0.8024 | 1.0000 | 22/15 | 0.3240 |
| tfidf_word+logreg | xlmr+logreg | +0.091 | [-0.033, +0.210] | 0.1674 | 1.0000 | 17/25 | 0.2800 |
| tfidf_char+logreg | fasttext+logreg | +0.074 | [-0.018, +0.162] | 0.1438 | 1.0000 | 9/11 | 0.8238 |
| tfidf_char+logreg | xlmr+logreg | +0.181 | [+0.064, +0.293] | 0.0037 | 0.0555 | 9/26 | 0.0060 |
| fasttext+logreg | xlmr+logreg | +0.107 | [+0.006, +0.212] | 0.0517 | 0.6720 | 6/21 | 0.0059 |
| tfidf_word+svm | tfidf_char+svm | -0.111 | [-0.221, -0.003] | 0.0589 | 0.7067 | 20/10 | 0.0987 |
| tfidf_word+svm | fasttext+svm | -0.046 | [-0.161, +0.064] | 0.4658 | 1.0000 | 23/15 | 0.2559 |
| tfidf_word+svm | xlmr+svm | +0.075 | [-0.043, +0.188] | 0.2826 | 1.0000 | 20/23 | 0.7608 |
| tfidf_char+svm | fasttext+svm | +0.065 | [-0.033, +0.157] | 0.2259 | 1.0000 | 11/13 | 0.8388 |
| tfidf_char+svm | xlmr+svm | +0.185 | [+0.078, +0.290] | 0.0034 | 0.0544 | 11/24 | 0.0410 |
| fasttext+svm | xlmr+svm | +0.121 | [+0.032, +0.212] | 0.0126 | 0.1764 | 8/19 | 0.0522 |
| tfidf_word+logreg | tfidf_word+svm | +0.035 | [-0.010, +0.083] | 0.1413 | 1.0000 | 3/6 | 0.5078 |
| tfidf_char+logreg | tfidf_char+svm | +0.015 | [-0.044, +0.077] | 0.6519 | 1.0000 | 4/6 | 0.7539 |
| fasttext+logreg | fasttext+svm | +0.006 | [-0.037, +0.046] | 0.9200 | 1.0000 | 1/3 | 0.6250 |
| xlmr+logreg | xlmr+svm | +0.019 | [-0.044, +0.085] | 0.6361 | 1.0000 | 5/3 | 0.7266 |

Holm-adjusted p < 0.05: 0 of 16.

CV results are descriptive: overlapping training folds make a naive McNemar test on pooled out-of-fold rows anti-conservative. The eligible high-density review group contains 12 comments. Subgroup scores are descriptive.
