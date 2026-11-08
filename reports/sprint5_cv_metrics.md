# Sprint 5 silver CV metrics

Out-of-fold predictions: **475 comments**. Frozen held-out test items are excluded. The corpus is density-enriched; these scores describe this sample, not natural YouTube prevalence.

| Representation | Classifier | n | Accuracy | Macro precision | Macro recall | Macro F1 |
|---|---|---:|---:|---:|---:|---:|
| fasttext | logreg | 475 | 0.583 | 0.583 | 0.573 | 0.574 |
| fasttext | svm | 475 | 0.589 | 0.578 | 0.576 | 0.571 |
| tfidf_char | logreg | 475 | 0.564 | 0.565 | 0.560 | 0.561 |
| tfidf_char | svm | 475 | 0.539 | 0.534 | 0.534 | 0.533 |
| tfidf_word | logreg | 475 | 0.436 | 0.432 | 0.431 | 0.432 |
| tfidf_word | svm | 475 | 0.408 | 0.402 | 0.402 | 0.402 |
| xlmr | logreg | 475 | 0.522 | 0.525 | 0.510 | 0.505 |
| xlmr | svm | 475 | 0.556 | 0.573 | 0.538 | 0.524 |

## fasttext + logreg

### Per-class metrics

| Class | Support | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| pos | 171 | 0.742 | 0.673 | 0.706 |
| neg | 140 | 0.491 | 0.407 | 0.445 |
| neu | 164 | 0.515 | 0.640 | 0.571 |

### Confusion matrix

Rows are reference labels; columns are predictions.

| Reference \ predicted | pos | neg | neu |
|---|---:|---:|---:|
| pos | 115 | 22 | 34 |
| neg | 18 | 57 | 65 |
| neu | 22 | 37 | 105 |

### By density

Groups use the deterministic Kazakh and Russian word counts. At least five counted words are required.

| Density | n | Accuracy | Macro F1 |
|---|---:|---:|---:|
| mono | 162 | 0.623 | 0.618 |
| low | 130 | 0.569 | 0.570 |
| med | 95 | 0.579 | 0.542 |
| high | 54 | 0.537 | 0.464 |

## fasttext + svm

### Per-class metrics

| Class | Support | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| pos | 171 | 0.720 | 0.737 | 0.728 |
| neg | 140 | 0.490 | 0.350 | 0.408 |
| neu | 164 | 0.525 | 0.640 | 0.577 |

### Confusion matrix

Rows are reference labels; columns are predictions.

| Reference \ predicted | pos | neg | neu |
|---|---:|---:|---:|
| pos | 126 | 17 | 28 |
| neg | 24 | 49 | 67 |
| neu | 25 | 34 | 105 |

### By density

Groups use the deterministic Kazakh and Russian word counts. At least five counted words are required.

| Density | n | Accuracy | Macro F1 |
|---|---:|---:|---:|
| mono | 162 | 0.623 | 0.612 |
| low | 130 | 0.592 | 0.583 |
| med | 95 | 0.516 | 0.460 |
| high | 54 | 0.630 | 0.532 |

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

Groups use the deterministic Kazakh and Russian word counts. At least five counted words are required.

| Density | n | Accuracy | Macro F1 |
|---|---:|---:|---:|
| mono | 162 | 0.605 | 0.600 |
| low | 130 | 0.554 | 0.555 |
| med | 95 | 0.537 | 0.517 |
| high | 54 | 0.611 | 0.599 |

## tfidf_char + svm

### Per-class metrics

| Class | Support | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| pos | 171 | 0.637 | 0.637 | 0.637 |
| neg | 140 | 0.444 | 0.457 | 0.451 |
| neu | 164 | 0.519 | 0.506 | 0.512 |

### Confusion matrix

Rows are reference labels; columns are predictions.

| Reference \ predicted | pos | neg | neu |
|---|---:|---:|---:|
| pos | 109 | 31 | 31 |
| neg | 30 | 64 | 46 |
| neu | 32 | 49 | 83 |

### By density

Groups use the deterministic Kazakh and Russian word counts. At least five counted words are required.

| Density | n | Accuracy | Macro F1 |
|---|---:|---:|---:|
| mono | 162 | 0.580 | 0.568 |
| low | 130 | 0.554 | 0.555 |
| med | 95 | 0.442 | 0.432 |
| high | 54 | 0.648 | 0.639 |

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

Groups use the deterministic Kazakh and Russian word counts. At least five counted words are required.

| Density | n | Accuracy | Macro F1 |
|---|---:|---:|---:|
| mono | 162 | 0.481 | 0.466 |
| low | 130 | 0.485 | 0.486 |
| med | 95 | 0.326 | 0.313 |
| high | 54 | 0.463 | 0.454 |

## tfidf_word + svm

### Per-class metrics

| Class | Support | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| pos | 171 | 0.524 | 0.520 | 0.522 |
| neg | 140 | 0.316 | 0.300 | 0.308 |
| neu | 164 | 0.366 | 0.384 | 0.375 |

### Confusion matrix

Rows are reference labels; columns are predictions.

| Reference \ predicted | pos | neg | neu |
|---|---:|---:|---:|
| pos | 89 | 34 | 48 |
| neg | 37 | 42 | 61 |
| neu | 44 | 57 | 63 |

### By density

Groups use the deterministic Kazakh and Russian word counts. At least five counted words are required.

| Density | n | Accuracy | Macro F1 |
|---|---:|---:|---:|
| mono | 162 | 0.414 | 0.398 |
| low | 130 | 0.454 | 0.453 |
| med | 95 | 0.368 | 0.354 |
| high | 54 | 0.389 | 0.384 |

## xlmr + logreg

### Per-class metrics

| Class | Support | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| pos | 171 | 0.601 | 0.643 | 0.621 |
| neg | 140 | 0.519 | 0.300 | 0.380 |
| neu | 164 | 0.455 | 0.585 | 0.512 |

### Confusion matrix

Rows are reference labels; columns are predictions.

| Reference \ predicted | pos | neg | neu |
|---|---:|---:|---:|
| pos | 110 | 10 | 51 |
| neg | 34 | 42 | 64 |
| neu | 39 | 29 | 96 |

### By density

Groups use the deterministic Kazakh and Russian word counts. At least five counted words are required.

| Density | n | Accuracy | Macro F1 |
|---|---:|---:|---:|
| mono | 162 | 0.537 | 0.520 |
| low | 130 | 0.500 | 0.490 |
| med | 95 | 0.474 | 0.429 |
| high | 54 | 0.556 | 0.477 |

## xlmr + svm

### Per-class metrics

| Class | Support | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| pos | 171 | 0.595 | 0.731 | 0.656 |
| neg | 140 | 0.625 | 0.250 | 0.357 |
| neu | 164 | 0.498 | 0.634 | 0.558 |

### Confusion matrix

Rows are reference labels; columns are predictions.

| Reference \ predicted | pos | neg | neu |
|---|---:|---:|---:|
| pos | 125 | 5 | 41 |
| neg | 41 | 35 | 64 |
| neu | 44 | 16 | 104 |

### By density

Groups use the deterministic Kazakh and Russian word counts. At least five counted words are required.

| Density | n | Accuracy | Macro F1 |
|---|---:|---:|---:|
| mono | 162 | 0.580 | 0.554 |
| low | 130 | 0.523 | 0.502 |
| med | 95 | 0.505 | 0.441 |
| high | 54 | 0.611 | 0.483 |
