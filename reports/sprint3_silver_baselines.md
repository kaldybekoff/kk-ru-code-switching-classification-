# Sprint 3 baseline results

Video-disjoint blind AI review: **98 comments**. Training sentiment labels came from a local AI model; test labels came from Codex without seeing those predictions. This is not human-gold accuracy or human inter-annotator agreement. An alternate annotation model was inspected on this review before final model selection, so these scores are exploratory. The corpus is density-enriched; these scores describe this sample, not natural YouTube prevalence.

| Representation | Classifier | n | Accuracy | Macro precision | Macro recall | Macro F1 |
|---|---|---:|---:|---:|---:|---:|
| fasttext | logreg | 98 | 0.571 | 0.569 | 0.551 | 0.532 |
| tfidf_char | logreg | 98 | 0.571 | 0.586 | 0.579 | 0.552 |
| tfidf_word | logreg | 98 | 0.480 | 0.465 | 0.464 | 0.462 |

## fasttext + logreg

### Per-class metrics

| Class | Support | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| pos | 45 | 0.842 | 0.711 | 0.771 |
| neg | 29 | 0.500 | 0.276 | 0.356 |
| neu | 24 | 0.364 | 0.667 | 0.471 |

### Confusion matrix

Rows are reference labels; columns are predictions.

| Reference \ predicted | pos | neg | neu |
|---|---:|---:|---:|
| pos | 32 | 4 | 9 |
| neg | 2 | 8 | 19 |
| neu | 4 | 4 | 16 |

### By density

| Density | n | Accuracy | Macro F1 |
|---|---:|---:|---:|
| mono | 64 | 0.625 | 0.569 |
| low | 13 | 0.538 | 0.502 |
| med | 14 | 0.429 | 0.321 |
| high | 7 | 0.429 | 0.300 |

## tfidf_char + logreg

### Per-class metrics

| Class | Support | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| pos | 45 | 0.900 | 0.600 | 0.720 |
| neg | 29 | 0.435 | 0.345 | 0.385 |
| neu | 24 | 0.422 | 0.792 | 0.551 |

### Confusion matrix

Rows are reference labels; columns are predictions.

| Reference \ predicted | pos | neg | neu |
|---|---:|---:|---:|
| pos | 27 | 9 | 9 |
| neg | 2 | 10 | 17 |
| neu | 1 | 4 | 19 |

### By density

| Density | n | Accuracy | Macro F1 |
|---|---:|---:|---:|
| mono | 64 | 0.547 | 0.524 |
| low | 13 | 0.615 | 0.605 |
| med | 14 | 0.643 | 0.500 |
| high | 7 | 0.571 | 0.417 |

## tfidf_word + logreg

### Per-class metrics

| Class | Support | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| pos | 45 | 0.632 | 0.533 | 0.578 |
| neg | 29 | 0.452 | 0.483 | 0.467 |
| neu | 24 | 0.310 | 0.375 | 0.340 |

### Confusion matrix

Rows are reference labels; columns are predictions.

| Reference \ predicted | pos | neg | neu |
|---|---:|---:|---:|
| pos | 24 | 11 | 10 |
| neg | 5 | 14 | 10 |
| neu | 9 | 6 | 9 |

### By density

| Density | n | Accuracy | Macro F1 |
|---|---:|---:|---:|
| mono | 64 | 0.516 | 0.472 |
| low | 13 | 0.462 | 0.456 |
| med | 14 | 0.357 | 0.288 |
| high | 7 | 0.429 | 0.378 |
