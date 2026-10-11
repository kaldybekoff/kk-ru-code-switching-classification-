# Silver annotation audit

Annotator model: `cardiffnlp/twitter-xlm-roberta-base-sentiment` at revision `f2f1202b1bdeb07342385c3f807f9c07cd8f5cf8`. Predictions: 1000.
Label counts: {'neu': 350, 'pos': 358, 'neg': 292}.

These are AI-generated silver labels. The 100-row blind review was done by Codex without seeing the local model's predictions. Neither comparison is inter-annotator agreement between humans, and neither estimates gold-label accuracy.

## Existing sparse labels vs local AI

Comparable: **24**; exact agreement: **0.667**; Cohen's κ: **0.491**.

| existing \ local AI | pos | neg | neu | skip |
|---|---:|---:|---:|---:|
| pos | 9 | 0 | 2 | 0 |
| neg | 0 | 2 | 2 | 0 |
| neu | 0 | 3 | 5 | 0 |
| skip | 0 | 0 | 1 | 0 |

## Blind Codex review vs local AI

Comparable: **100**; exact agreement: **0.670**; Cohen's κ: **0.510**.

| Codex \ local AI | pos | neg | neu | skip |
|---|---:|---:|---:|---:|
| pos | 32 | 3 | 10 | 0 |
| neg | 2 | 17 | 10 | 0 |
| neu | 1 | 5 | 18 | 0 |
| skip | 0 | 0 | 2 | 0 |

## Blind Codex density review vs deterministic density

Comparable: **98**; exact agreement: **0.459**; Cohen's κ: **0.254**.

| Codex \ algorithm | mono | low | med | high |
|---|---:|---:|---:|---:|
| mono | 27 | 23 | 12 | 2 |
| low | 1 | 5 | 6 | 1 |
| med | 0 | 2 | 7 | 5 |
| high | 0 | 0 | 1 | 6 |

## Blind review agreement by density

| Density | n | Exact agreement | Cohen's κ |
|---|---:|---:|---:|
| mono | 28 | 0.714 | 0.454 |
| low | 30 | 0.733 | 0.614 |
| med | 26 | 0.577 | 0.382 |
| high | 16 | 0.625 | 0.442 |

Small cells, including the 24 existing labels, are exploratory. Disagreements need human adjudication before this becomes a gold corpus.
