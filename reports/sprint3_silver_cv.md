# Sprint 3 baseline results

Out-of-fold predictions: **475 comments**. Frozen held-out test items are excluded. The corpus is density-enriched; these scores describe this sample, not natural YouTube prevalence.

| Representation | Classifier | n | Accuracy | Macro precision | Macro recall | Macro F1 |
|---|---|---:|---:|---:|---:|---:|
| fasttext | logreg | 475 | 0.602 | 0.597 | 0.592 | 0.593 |
| tfidf_char | logreg | 475 | 0.564 | 0.565 | 0.560 | 0.561 |
| tfidf_word | logreg | 475 | 0.436 | 0.432 | 0.431 | 0.432 |

## fasttext + logreg

### Per-class metrics

| Class | Support | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| pos | 171 | 0.779 | 0.743 | 0.760 |
| neg | 140 | 0.484 | 0.429 | 0.455 |
| neu | 164 | 0.527 | 0.604 | 0.562 |

### Confusion matrix

Rows are reference labels; columns are predictions.

| Reference \ predicted | pos | neg | neu |
|---|---:|---:|---:|
| pos | 127 | 18 | 26 |
| neg | 17 | 60 | 63 |
| neu | 19 | 46 | 99 |

### By density

| Density | n | Accuracy | Macro F1 |
|---|---:|---:|---:|
| mono | 172 | 0.628 | 0.614 |
| low | 132 | 0.644 | 0.645 |
| med | 112 | 0.536 | 0.509 |
| high | 59 | 0.559 | 0.502 |

## tfidf_char + logreg

### Per-class metrics

| Class | Support | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| pos | 171 | 0.697 | 0.620 | 0.656 |
| neg | 140 | 0.476 | 0.493 | 0.484 |
| neu | 164 | 0.522 | 0.567 | 0.544 |

### Confusion matrix

Rows are reference labels; columns are predictions.

| Reference \ predicted | pos | neg | neu |
|---|---:|---:|---:|
| pos | 106 | 30 | 35 |
| neg | 21 | 69 | 50 |
| neu | 25 | 46 | 93 |

### By density

| Density | n | Accuracy | Macro F1 |
|---|---:|---:|---:|
| mono | 172 | 0.587 | 0.583 |
| low | 132 | 0.568 | 0.569 |
| med | 112 | 0.509 | 0.490 |
| high | 59 | 0.593 | 0.580 |

## tfidf_word + logreg

### Per-class metrics

| Class | Support | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| pos | 171 | 0.557 | 0.544 | 0.550 |
| neg | 140 | 0.346 | 0.379 | 0.362 |
| neu | 164 | 0.394 | 0.372 | 0.382 |

### Confusion matrix

Rows are reference labels; columns are predictions.

| Reference \ predicted | pos | neg | neu |
|---|---:|---:|---:|
| pos | 93 | 37 | 41 |
| neg | 34 | 53 | 53 |
| neu | 40 | 63 | 61 |

### By density

| Density | n | Accuracy | Macro F1 |
|---|---:|---:|---:|
| mono | 172 | 0.483 | 0.471 |
| low | 132 | 0.492 | 0.494 |
| med | 112 | 0.295 | 0.286 |
| high | 59 | 0.441 | 0.431 |
