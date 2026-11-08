# Sprint 5 blind AI review metrics

Video-disjoint blind AI review: **98 comments**. Training sentiment labels came from a local AI model; test labels came from Codex without seeing those predictions. This is not human-gold accuracy or human inter-annotator agreement. An alternate annotation model was inspected on this review before final model selection, so these scores are exploratory. The corpus is density-enriched; these scores describe this sample, not natural YouTube prevalence.

| Representation | Classifier | n | Accuracy | Macro precision | Macro recall | Macro F1 |
|---|---|---:|---:|---:|---:|---:|
| fasttext | logreg | 98 | 0.551 | 0.536 | 0.529 | 0.478 |
| fasttext | svm | 98 | 0.531 | 0.514 | 0.505 | 0.473 |
| tfidf_char | logreg | 98 | 0.571 | 0.586 | 0.579 | 0.552 |
| tfidf_char | svm | 98 | 0.551 | 0.558 | 0.553 | 0.537 |
| tfidf_word | logreg | 98 | 0.480 | 0.465 | 0.464 | 0.462 |
| tfidf_word | svm | 98 | 0.449 | 0.427 | 0.427 | 0.426 |
| xlmr | logreg | 98 | 0.398 | 0.443 | 0.402 | 0.371 |
| xlmr | svm | 98 | 0.418 | 0.375 | 0.405 | 0.352 |

## fasttext + logreg

### Per-class metrics

| Class | Support | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| pos | 45 | 0.892 | 0.733 | 0.805 |
| neg | 29 | 0.375 | 0.103 | 0.162 |
| neu | 24 | 0.340 | 0.750 | 0.468 |

### Confusion matrix

Rows are reference labels; columns are predictions.

| Reference \ predicted | pos | neg | neu |
|---|---:|---:|---:|
| pos | 33 | 1 | 11 |
| neg | 2 | 3 | 24 |
| neu | 2 | 4 | 18 |

### By density

Groups use the deterministic Kazakh and Russian word counts. At least five counted words are required.

| Density | n | Accuracy | Macro F1 |
|---|---:|---:|---:|
| mono | 24 | 0.667 | 0.503 |
| low | 30 | 0.467 | 0.436 |
| med | 21 | 0.429 | 0.329 |
| high | 12 | 0.500 | 0.500 |

## fasttext + svm

### Per-class metrics

| Class | Support | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| pos | 45 | 0.889 | 0.711 | 0.790 |
| neg | 29 | 0.333 | 0.138 | 0.195 |
| neu | 24 | 0.320 | 0.667 | 0.432 |

### Confusion matrix

Rows are reference labels; columns are predictions.

| Reference \ predicted | pos | neg | neu |
|---|---:|---:|---:|
| pos | 32 | 3 | 10 |
| neg | 1 | 4 | 24 |
| neu | 3 | 5 | 16 |

### By density

Groups use the deterministic Kazakh and Russian word counts. At least five counted words are required.

| Density | n | Accuracy | Macro F1 |
|---|---:|---:|---:|
| mono | 24 | 0.625 | 0.481 |
| low | 30 | 0.500 | 0.471 |
| med | 21 | 0.381 | 0.305 |
| high | 12 | 0.500 | 0.499 |

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

Groups use the deterministic Kazakh and Russian word counts. At least five counted words are required.

| Density | n | Accuracy | Macro F1 |
|---|---:|---:|---:|
| mono | 24 | 0.667 | 0.536 |
| low | 30 | 0.500 | 0.506 |
| med | 21 | 0.429 | 0.396 |
| high | 12 | 0.667 | 0.653 |

## tfidf_char + svm

### Per-class metrics

| Class | Support | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| pos | 45 | 0.812 | 0.578 | 0.675 |
| neg | 29 | 0.462 | 0.414 | 0.436 |
| neu | 24 | 0.400 | 0.667 | 0.500 |

### Confusion matrix

Rows are reference labels; columns are predictions.

| Reference \ predicted | pos | neg | neu |
|---|---:|---:|---:|
| pos | 26 | 9 | 10 |
| neg | 3 | 12 | 14 |
| neu | 3 | 5 | 16 |

### By density

Groups use the deterministic Kazakh and Russian word counts. At least five counted words are required.

| Density | n | Accuracy | Macro F1 |
|---|---:|---:|---:|
| mono | 24 | 0.583 | 0.456 |
| low | 30 | 0.600 | 0.597 |
| med | 21 | 0.381 | 0.362 |
| high | 12 | 0.583 | 0.561 |

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

Groups use the deterministic Kazakh and Russian word counts. At least five counted words are required.

| Density | n | Accuracy | Macro F1 |
|---|---:|---:|---:|
| mono | 24 | 0.583 | 0.378 |
| low | 30 | 0.500 | 0.479 |
| med | 21 | 0.429 | 0.434 |
| high | 12 | 0.417 | 0.367 |

## tfidf_word + svm

### Per-class metrics

| Class | Support | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| pos | 45 | 0.571 | 0.533 | 0.552 |
| neg | 29 | 0.414 | 0.414 | 0.414 |
| neu | 24 | 0.296 | 0.333 | 0.314 |

### Confusion matrix

Rows are reference labels; columns are predictions.

| Reference \ predicted | pos | neg | neu |
|---|---:|---:|---:|
| pos | 24 | 11 | 10 |
| neg | 8 | 12 | 9 |
| neu | 10 | 6 | 8 |

### By density

Groups use the deterministic Kazakh and Russian word counts. At least five counted words are required.

| Density | n | Accuracy | Macro F1 |
|---|---:|---:|---:|
| mono | 24 | 0.542 | 0.319 |
| low | 30 | 0.433 | 0.417 |
| med | 21 | 0.429 | 0.440 |
| high | 12 | 0.417 | 0.367 |

## xlmr + logreg

### Per-class metrics

| Class | Support | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| pos | 45 | 0.741 | 0.444 | 0.556 |
| neg | 29 | 0.333 | 0.138 | 0.195 |
| neu | 24 | 0.254 | 0.625 | 0.361 |

### Confusion matrix

Rows are reference labels; columns are predictions.

| Reference \ predicted | pos | neg | neu |
|---|---:|---:|---:|
| pos | 20 | 4 | 21 |
| neg | 2 | 4 | 23 |
| neu | 5 | 4 | 15 |

### By density

Groups use the deterministic Kazakh and Russian word counts. At least five counted words are required.

| Density | n | Accuracy | Macro F1 |
|---|---:|---:|---:|
| mono | 24 | 0.500 | 0.451 |
| low | 30 | 0.400 | 0.365 |
| med | 21 | 0.333 | 0.264 |
| high | 12 | 0.250 | 0.283 |

## xlmr + svm

### Per-class metrics

| Class | Support | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| pos | 45 | 0.714 | 0.556 | 0.625 |
| neg | 29 | 0.143 | 0.034 | 0.056 |
| neu | 24 | 0.268 | 0.625 | 0.375 |

### Confusion matrix

Rows are reference labels; columns are predictions.

| Reference \ predicted | pos | neg | neu |
|---|---:|---:|---:|
| pos | 25 | 3 | 17 |
| neg | 4 | 1 | 24 |
| neu | 6 | 3 | 15 |

### By density

Groups use the deterministic Kazakh and Russian word counts. At least five counted words are required.

| Density | n | Accuracy | Macro F1 |
|---|---:|---:|---:|
| mono | 24 | 0.542 | 0.432 |
| low | 30 | 0.400 | 0.365 |
| med | 21 | 0.333 | 0.255 |
| high | 12 | 0.250 | 0.200 |
