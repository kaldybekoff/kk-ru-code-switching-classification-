# Sprint 3 status — corpus finalization and baseline models

These results document the original Sprint 3 run. Week 10 recomputes fastText
using only unsampled comments on training videos and uses updated density groups.
[Current results](sprint5_status.md)

## Delivered

- Audited the frozen 1,000-comment sample (150 natural, 850 enriched): all IDs
  are present in their expected annotation sheets, with no duplicate IDs or
  invalid filled labels. The original annotation sheets remain incomplete:
  24 sentiment cells filled (including one `skip`), 976 empty. The 100-row
  retest has only 2 filled cells. See `sprint3_corpus_audit.md`.
- Created and blind-reviewed a 100-comment validation subset. The review was
  performed by Codex AI without seeing the local model's predictions. Two
  review rows were marked `skip`, leaving 98 test items. No second human was
  involved, so **human inter-annotator agreement is unavailable**.
- Predicted sentiment for all 1,000 sampled comments using
  `cardiffnlp/twitter-xlm-roberta-base-sentiment`, revision
  `f2f1202b1bdeb07342385c3f807f9c07cd8f5cf8`. Three comments already
  known as `skip` were excluded, leaving **997 usable silver rows**:
  358 positive, 292 negative, 347 neutral. Predictions for all 1,000 include
  probabilities, model revision, and a SHA-256 hash of each input text. This
  does not turn the original corpus into human gold.
- Validated silver labels against the 24 previously filled annotations:
  exact agreement 0.667, Cohen's κ 0.491. Against the 100-row blind AI review:
  exact agreement 0.670, κ 0.510. Codex density review versus the deterministic
  density algorithm: 98 comparable rows, exact agreement 0.459, κ 0.254.
  Confusion matrices and density-stratified agreement are in
  `sprint3_silver_audit.md`. These are model-to-review and model-to-algorithm
  agreements, **not agreement between two people**.
- Trained word TF-IDF, character TF-IDF, and fastText embeddings with logistic
  regression. fastText embeddings were learned only from 41,497 unsampled
  comments (100 dimensions, 39,597-word vocabulary). The blind test uses 98
  Codex-reviewed comments from videos excluded from the 557-row silver
  training pool. A separate frozen five-fold protocol yields 475 out-of-fold
  predictions and reserves 82 additional silver-labeled comments.

## Results

| Representation | 5-fold silver CV macro-F1 (n=475) | Blind AI review accuracy (n=98) | Blind AI review macro-F1 |
|---|---:|---:|---:|
| Word TF-IDF | 0.432 | 0.480 | 0.462 |
| Character TF-IDF | 0.561 | 0.571 | **0.552** |
| fastText | **0.593** | 0.571 | 0.532 |

Full accuracy, macro and per-class precision, recall, F1, confusion matrices,
and density breakdowns are in `sprint3_silver_cv.md` and
`sprint3_silver_baselines.md`. The blind AI review is an exploratory reference,
not a human-gold test. High-density test results have only seven items.

## Corpus quality and scope

The silver labels disagree with the blind AI review on 33/100 comments. The
density algorithm disagrees with the Codex density review on 53/98. These
differences are material for research conclusions. The available scores support
an exploratory Sprint 3 baseline comparison, not a claim of validated human
annotation quality or publication-ready corpus quality. The original strict
human-corpus finalization gate remains intentionally closed.

The Codex labels were assigned without access to either model's outputs, but
the alternate small annotation model was inspected on this review before the
final Cardiff model was selected. Treat the 98-row comparison as exploratory,
not an untouched model-selection test.

The silver sample is enriched for code switching. Overall scores describe
that sample; they do not estimate natural YouTube prevalence. The model used
to make silver labels was sentiment-tuned on eight other languages and should
not be assumed to generalize perfectly to Kazakh–Russian comments.

Run environment: Python 3.13.0, scikit-learn 1.9.1, gensim 4.4.0,
transformers 4.57.6, PyTorch 2.11.0+cu128, NumPy 2.5.3, SciPy 1.18.1.
