# Kazakh–Russian code-switched sentiment classification

This project compares word and character TF-IDF, fastText and a frozen XLM-R encoder on Kazakh–Russian YouTube comments. Each representation is evaluated with logistic regression and a linear SVM. Comments from the same video stay together when splitting the data.

## Run the tests

From the repository root in a Python virtual environment, run

```sh
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
```

## Reproduce the model comparison

```sh
python -m src.eval.sprint5
```

This command uses the frozen splits and predictions in `data/labeled/` and writes the comparison to `reports/sprint5_comparison.md` locally. The five-fold evaluation covers 475 comments with automatic training labels. A separate evaluation uses 98 comments reviewed by an independent AI annotator. Recomputing the human-review metrics requires the local human annotations and predictions.

The [annotation guide](../annotation/guideline.md) defines sentiment and code-switching density. The [sampling design](../docs/sampling_design.md) describes the uniform and density-enriched samples. To collect new comments, copy `.env.example` to `.env` and set `YOUTUBE_API_KEY`.
