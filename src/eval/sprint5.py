"""Reproducible Sprint 5 grid analysis on paired predictions.

The blind AI review is the inferential set. CV remains descriptive because its
out-of-fold predictions come from overlapping training folds. All inference is
exploratory until independent human labels are available.
"""

import argparse
import csv
import json
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score

from src.eval.compare import holm, mcnemar
from src.eval.density_groups import COUNTS, DATA, load_groups
from src.eval.folds import load_labeled
from src.models.run_cv import validate_inputs

LABELS = ["pos", "neg", "neu"]
BUCKETS = ["mono", "low", "med", "high"]
REPS = ["tfidf_word", "tfidf_char", "fasttext", "xlmr"]
CLFS = ["logreg", "svm"]
ROOT = Path(__file__).resolve().parents[2]


def read_grid(path, expected=None, groups=None):
    with open(path, encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows or "video_id" not in rows[0]:
        raise ValueError(f"{path}: empty predictions or missing video_id")
    systems = defaultdict(dict)
    for row in rows:
        key = (row["representation"], row["classifier"])
        cid = row["comment_id"]
        if key[0] not in REPS or key[1] not in CLFS or cid in systems[key]:
            raise ValueError(f"{path}: unexpected system or duplicate prediction")
        if (row["gold"] not in LABELS or row["pred"] not in LABELS
                or row["density"] not in BUCKETS or not row["video_id"]):
            raise ValueError(f"{path}: invalid label, density, or video ID")
        systems[key][cid] = row
    if set(systems) != {(r, c) for r in REPS for c in CLFS}:
        raise ValueError(f"{path}: incomplete representation × classifier grid")
    def signature(items):
        return {cid: (r["gold"], r["density"], r["fold"], r["video_id"], int(r["n_tokens"]))
                for cid, r in items.items()}

    ref = signature(systems[(REPS[0], CLFS[0])])
    for key, items in systems.items():
        if signature(items) != ref:
            raise ValueError(f"{path}: unpaired predictions in {key}")
    if expected is not None and ref != expected:
        raise ValueError(f"{path}: predictions do not match the frozen evaluation rows")
    ids = sorted(ref)
    y = np.array([ref[cid][0] for cid in ids])
    density = np.array([groups[cid] if groups is not None else ref[cid][1] for cid in ids])
    video = np.array([ref[cid][3] for cid in ids])
    pred = {key: np.array([items[cid]["pred"] for cid in ids])
            for key, items in systems.items()}
    lengths = np.array([ref[cid][4] for cid in ids])
    return y, density, video, pred, lengths


def macro(y, pred):
    # This is sklearn's fixed-label macro-F1, computed directly because the
    # video permutation test calls it tens of thousands of times.
    total = 0.0
    for label in LABELS:
        actual = y == label
        guessed = pred == label
        denominator = np.count_nonzero(actual) + np.count_nonzero(guessed)
        if denominator:
            total += 2 * np.count_nonzero(actual & guessed) / denominator
    return total / len(LABELS)


def video_indices(video):
    return [np.flatnonzero(video == v) for v in sorted(set(video))]


def cluster_interval(y, a, b, groups, draws, seed):
    rng = np.random.default_rng(seed)
    differences = np.empty(draws)
    for i in range(draws):
        picked = rng.integers(len(groups), size=len(groups))
        indices = np.concatenate([groups[j] for j in picked])
        differences[i] = macro(y[indices], a[indices]) - macro(y[indices], b[indices])
    return np.quantile(differences, [0.025, 0.975])


def cluster_swap_p(y, a, b, groups, draws, seed):
    """Two-sided paired randomization: swap complete video prediction blocks."""
    rng = np.random.default_rng(seed)
    observed = abs(macro(y, a) - macro(y, b))
    extreme = 0
    for _ in range(draws):
        swap = np.zeros(len(y), dtype=bool)
        for group, choice in zip(groups, rng.integers(2, size=len(groups))):
            if choice:
                swap[group] = True
        left = np.where(swap, b, a)
        right = np.where(swap, a, b)
        extreme += abs(macro(y, left) - macro(y, right)) >= observed - 1e-12
    return (extreme + 1) / (draws + 1)


def score_table(y, density, pred, lengths, title):
    lines = [f"### {title}", "", "| Representation | Classifier | Macro-F1 | Accuracy | Negative recall |",
             "|---|---|---:|---:|---:|"]
    for rep in REPS:
        for clf in CLFS:
            p = pred[(rep, clf)]
            neg = y == "neg"
            lines.append(f"| {rep} | {clf} | {macro(y, p):.3f} | "
                         f"{accuracy_score(y, p):.3f} | {np.mean(p[neg] == 'neg'):.3f} |")
    lines += ["", f"Density analysis includes {np.count_nonzero(density != '')} of {len(y)} "
              "comments with at least five words assigned to Kazakh or Russian. Both sets use "
              "the same deterministic word counts. Median length uses the corpus word-token count.", "",
              "| Density | n | Median length | Majority accuracy | " + " | ".join(REPS) + " |",
              "|---|---:|---:|---:|" + "---:|" * len(REPS)]
    for bucket in BUCKETS:
        use = density == bucket
        if not np.any(use):
            continue
        floor = max(Counter(y[use]).values()) / sum(use)
        lines.append(f"| {bucket} | {sum(use)} | {np.median(lengths[use]):g} | {floor:.3f} | " +
                     " | ".join(f"{macro(y[use], pred[(rep, 'logreg')][use]):.3f}"
                                for rep in REPS) + " |")
    lines += ["", "Density table uses logistic regression, the prespecified representation comparison. "
              "Cells with fewer than 20 examples are descriptive only.", ""]
    return lines


def compare_review(y, video, pred, draws, seed, title="Paired comparisons on the blind AI review"):
    groups = video_indices(video)
    pairs = []
    for clf in CLFS:
        pairs.extend(((a, clf), (b, clf)) for a, b in combinations(REPS, 2))
    pairs.extend(((rep, "logreg"), (rep, "svm")) for rep in REPS)
    tests = []
    for number, (left, right) in enumerate(pairs):
        a, b = pred[left], pred[right]
        delta = macro(y, a) - macro(y, b)
        lo, hi = cluster_interval(y, a, b, groups, draws, seed + number)
        p_swap = cluster_swap_p(y, a, b, groups, draws, seed + 1000 + number)
        n01, n10, p_mc = mcnemar(a == y, b == y)
        tests.append((left, right, delta, lo, hi, p_swap, n01, n10, p_mc))
    adjusted = holm([t[5] for t in tests])
    lines = [f"### {title}", "",
             f"{len(y)} comments in {len(groups)} videos. Difference is first system minus second. "
             f"95% percentile intervals resample videos ({draws} draws; seed {seed}). "
             "The two-sided macro-F1 permutation swaps predictions for whole videos; "
             "Holm correction covers all 16 comparisons below. Exact McNemar p-values "
             "test paired accuracy and are unadjusted diagnostics, not the primary test.", "",
             "| First | Second | Δ macro-F1 | 95% video CI | permutation p | Holm p | McNemar n01/n10 | McNemar p |",
             "|---|---|---:|---:|---:|---:|---:|---:|"]
    for t, adj in zip(tests, adjusted):
        left, right, delta, lo, hi, pv, n01, n10, pmc = t
        lines.append(f"| {'+'.join(left)} | {'+'.join(right)} | {delta:+.3f} | "
                     f"[{lo:+.3f}, {hi:+.3f}] | {pv:.4f} | {adj:.4f} | "
                     f"{n01}/{n10} | {pmc:.4f} |")
    lines += ["", f"Holm-adjusted p < 0.05: {sum(p < 0.05 for p in adjusted)} of {len(tests)}.", ""]
    return lines


def frozen_references(train_path, test_path, folds_path):
    train, test = load_labeled(train_path), load_labeled(test_path)
    split = json.loads(folds_path.read_text(encoding="utf-8"))
    validate_inputs(train, split)
    if (not test or len({r["comment_id"] for r in test}) != len(test)
            or {r["comment_id"] for r in train} & {r["comment_id"] for r in test}
            or {r["video_id"] for r in train} & {r["video_id"] for r in test}):
        raise ValueError("Invalid or overlapping training and review partitions")
    folds = {cid: str(i) for i, ids in enumerate(split["folds"], 1) for cid in ids}

    def entry(row, fold):
        return row["sentiment"], row["density"], fold, row["video_id"], int(row["n_tokens"])

    cv_ref = {r["comment_id"]: entry(r, folds[r["comment_id"]]) for r in train
              if r["comment_id"] in folds}
    return cv_ref, {r["comment_id"]: entry(r, "blind_ai_review") for r in test}


def render(cv_path, review_path, draws=10000, seed=2026, counts_path=COUNTS,
           train_path=DATA / "corpus_silver_train.csv", test_path=DATA / "corpus_ai_review_test.csv",
           folds_path=DATA / "folds_silver.json"):
    groups = load_groups(counts_path)
    cv_ref, review_ref = frozen_references(train_path, test_path, folds_path)
    cv_y, cv_density, cv_video, cv_pred, cv_lengths = read_grid(cv_path, cv_ref, groups)
    y, density, video, pred, lengths = read_grid(review_path, review_ref, groups)
    lines = ["# Sprint 5 comparative experiments", "",
             "Four frozen representations × two class-balanced linear classifiers. "
             "The same examples and labels are paired within each evaluation set. "
             "All systems use C=1. In the corrected experiment fastText fits unsampled "
             "text from the training videos separately for each split.", "",
             "CV uses silver labels from a sentiment model; the video-disjoint review uses "
             "blind Codex AI labels. Neither set is human gold. The review was inspected "
             "with another annotation model during Sprint 3, so even its inferential "
             "results are exploratory. Density-enriched sampling means scores do not "
             "estimate natural YouTube prevalence.", "",
             f"CV: {len(cv_y)} comments, {len(set(cv_video))} videos; "
             f"review: {len(y)} comments, {len(set(video))} videos.", ""]
    lines += score_table(cv_y, cv_density, cv_pred, cv_lengths, "Silver out-of-fold CV (descriptive)")
    lines += score_table(y, density, pred, lengths, "Blind AI review")
    lines += compare_review(y, video, pred, draws, seed)
    lines += ["CV results are descriptive: overlapping training folds make a naive "
              "McNemar test on pooled out-of-fold rows anti-conservative. "
              f"The eligible high-density review group contains {sum(density == 'high')} comments. "
              "Subgroup scores are descriptive.", ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cv", type=Path, default=DATA / "predictions_sprint5_grouped_silver_cv.csv")
    parser.add_argument("--review", type=Path, default=DATA / "predictions_sprint5_grouped_ai_review.csv")
    parser.add_argument("--out", type=Path, default=ROOT / "reports/sprint5_comparison.md")
    parser.add_argument("--draws", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    if args.draws < 100:
        parser.error("--draws must be at least 100")
    result = render(args.cv, args.review, args.draws, args.seed)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(result, encoding="utf-8")
    print(args.out)


if __name__ == "__main__":
    main()
