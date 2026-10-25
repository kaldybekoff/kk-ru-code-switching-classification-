# Sprint 4 transformer corpus audit

Encoder: `FacebookAI/xlm-roberta-base@e73636d4f797dec63c3081bb6ed5c7b0bb3f2089`; max sequence length: 256 tokens.
The tokenizer receives the same cleaned comment text used by all baselines. Padding is per batch; truncation is at the stated maximum.

| Partition | n | label counts (pos/neg/neu) | density counts (mono/low/med/high) | median tokens | p95 tokens | truncated |
|---|---:|---|---|---:|---:|---:|
| Silver training corpus | 557 | 198/166/193 | 204/156/131/66 | 27 | 137 | 7 |
| Blind AI review | 98 | 45/29/24 | 64/13/14/7 | 23 | 87 | 0 |

Frozen split: 5 video-grouped CV folds, 475 out-of-fold predictions, 82 reserved silver-label test rows.
The blind AI review is video-disjoint from the silver training corpus.

Silver sentiment labels were assigned by a sentiment-tuned XLM-R teacher. The pretrained encoder evaluated here shares an architecture family with that teacher, so agreement with silver labels can favor XLM-R. The blind AI review is exploratory and is not human gold.
