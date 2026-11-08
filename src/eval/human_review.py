"""Evaluate the frozen Sprint 5 systems against one human annotator."""

import argparse
import csv
import hashlib
import json
from collections import Counter

import numpy as np
from sklearn.metrics import accuracy_score, cohen_kappa_score, confusion_matrix
from sklearn.metrics import precision_recall_fscore_support

from src.eval.folds import load_labeled
from src.eval.sprint5 import (CLFS, DATA, LABELS, REPS, ROOT, compare_review, macro,
                              read_grid)
from src.annotate.xlmr_silver import MODEL, MODEL_REVISION, OUT as SILVER_PREDICTIONS
from src.preprocess.density import Density

WORK = ROOT / "data/interim/human_review"
HUMAN = WORK / "reviewer_A.csv"
TEST = WORK / "holdout.csv"
PREDICTIONS = WORK / "predictions.csv"
OUTPUT = ROOT / "reports/sprint5_human_review_ru.md"
AI = DATA / "validation_ai_review.csv"
AI_TEST = DATA / "corpus_ai_review_test.csv"
AI_PREDICTIONS = DATA / "predictions_sprint5_grouped_ai_review.csv"
TRAIN = DATA / "corpus_silver_train.csv"
MANIFEST = ROOT / "data/interim/batches/manifest.csv"
ANNOTATIONS = [*LABELS, "skip"]


def index(rows):
    result = {row["comment_id"]: row for row in rows}
    if len(result) != len(rows) or "" in result:
        raise ValueError("Duplicate or empty comment IDs")
    return result


def references():
    human = index(load_labeled(HUMAN))
    ai = index(load_labeled(AI))
    if human.keys() != ai.keys():
        raise ValueError("The reviews cover different comments")
    for cid, row in human.items():
        if row["text"] != ai[cid]["text"]:
            raise ValueError(f"Review text changed for {cid}")
        if row["sentiment"] not in ANNOTATIONS or ai[cid]["sentiment"] not in ANNOTATIONS:
            raise ValueError(f"Missing or invalid review label for {cid}")
    return human, ai


def holdout(human, ai):
    frozen = index(load_labeled(AI_TEST))
    if frozen.keys() != {cid for cid, row in ai.items() if row["sentiment"] != "skip"}:
        raise ValueError("The AI holdout does not match its reviewed comments")
    for cid, row in frozen.items():
        if row["text"] != ai[cid]["text"] or row["sentiment"] != ai[cid]["sentiment"]:
            raise ValueError(f"AI holdout text or label differs for {cid}")
    manifest = index([r for r in load_labeled(MANIFEST) if r["sample"] != "retest"])
    train = index(load_labeled(TRAIN))
    train_videos = {r["video_id"] for r in train.values()}
    density = Density.load()
    output = {}
    for cid, row in human.items():
        if row["sentiment"] == "skip":
            continue
        source = frozen.get(cid)
        if source is None:
            source = manifest[cid]
            bucket = density.measure(row["text"])["density"]
        else:
            bucket = source["density"]
        if cid in train or source["video_id"] in train_videos:
            raise ValueError(f"Human review intersects training for {cid}")
        output[cid] = {"comment_id": cid, "text": row["text"], "sentiment": row["sentiment"],
                       "video_id": source["video_id"], "density": bucket,
                       "n_tokens": source["n_tokens"]}
    if not output:
        raise ValueError("No comments with usable human labels")
    return output


def signatures(items):
    return {cid: (r["sentiment"], r["density"], "blind_ai_review", r["video_id"], int(r["n_tokens"]))
            for cid, r in items.items()}


def teacher_agreement(human):
    with SILVER_PREDICTIONS.open(encoding="utf-8") as handle:
        silver = index([json.loads(line) for line in handle if line.strip()])
    human_labels, silver_labels = [], []
    for cid, row in human.items():
        prediction = silver[cid]
        text_hash = hashlib.sha256(row["text"].encode("utf-8")).hexdigest()
        if (prediction["model"] != MODEL or prediction["revision"] != MODEL_REVISION
                or prediction["text_sha256"] != text_hash or prediction["label"] not in LABELS):
            raise ValueError(f"Invalid silver prediction for {cid}")
        if row["sentiment"] != "skip":
            human_labels.append(row["sentiment"])
            silver_labels.append(prediction["label"])
    matches = sum(a == b for a, b in zip(human_labels, silver_labels))
    return matches, len(human_labels), cohen_kappa_score(human_labels, silver_labels, labels=LABELS)


def compare(human, ai, rows, draws):
    frozen = index(load_labeled(AI_TEST))
    y, _, videos, predictions, _ = read_grid(PREDICTIONS, signatures(rows))
    ai_y, _, _, ai_predictions, _ = read_grid(AI_PREDICTIONS, signatures(frozen))
    shared = rows.keys() & frozen.keys()
    common_human = [cid in shared for cid in sorted(rows)]
    common_ai = [cid in shared for cid in sorted(frozen)]
    for key in predictions:
        if not np.array_equal(predictions[key][common_human], ai_predictions[key][common_ai]):
            raise ValueError(f"Reconstructed predictions differ from frozen predictions for {key}")

    human_all = [human[cid]["sentiment"] for cid in human]
    ai_all = [ai[cid]["sentiment"] for cid in human]
    y_shared = y[common_human]
    ai_shared = ai_y[common_ai]
    matches = sum(a == b for a, b in zip(human_all, ai_all))
    common_matches = int((y_shared == ai_shared).sum())
    counts = Counter(human_all)
    teacher_matches, teacher_n, teacher_kappa = teacher_agreement(human)
    lines = [
        "# Спринт 5. Проверка по человеческой разметке", "",
        f"Один человек оценил тональность {len(human)} комментариев вслепую. "
        f"Положительных — {counts['pos']}, отрицательных — {counts['neg']}, "
        f"нейтральных — {counts['neu']}, пропущенных — {counts['skip']}. "
        f"В проверку моделей вошли {len(y)} комментариев с {len(set(videos))} видео.", "",
        f"Все системы обучались на исходных {len(load_labeled(TRAIN))} комментариях с ИИ-метками. "
        f"Для {len(rows) - len(shared)} комментариев, которые ИИ ранее пропустил, "
        "предсказания восстановлены с прежними параметрами и обучающими данными. "
        "На общей выборке все предсказания совпали с сохранёнными.", "",
        "## Согласие человека и ИИ", "",
        f"На всех комментариях, включая `skip`, совпали {matches} из {len(human)} меток "
        f"(κ Коэна = {cohen_kappa_score(human_all, ai_all, labels=ANNOTATIONS):.3f}). "
        f"На комментариях с валидной оценкой у обоих разметчиков совпали "
        f"{common_matches} из {len(shared)} меток "
        f"(κ = {cohen_kappa_score(y_shared, ai_shared, labels=LABELS):.3f}).", "",
        "| Человек / ИИ | pos | neg | neu | skip |", "|---|---:|---:|---:|---:|",
    ]
    for label, values in zip(ANNOTATIONS, confusion_matrix(human_all, ai_all, labels=ANNOTATIONS)):
        lines.append(f"| {label} | " + " | ".join(map(str, values)) + " |")
    lines += [
        "",
        f"Модель Cardiff, которая дала обучающие метки, совпала с человеком в "
        f"{teacher_matches} из {teacher_n} комментариев (κ = {teacher_kappa:.3f}). "
        "В этом сравнении учитываются комментарии, которым человек присвоил "
        "один из трёх классов тональности.",
    ]

    majority_label, majority_count = Counter(y).most_common(1)[0]
    lines += [
        "", "## Метрики восьми систем", "",
        f"Macro-F1 на общей выборке из {len(shared)} комментариев сравнивает один и тот же "
        "набор предсказаний при двух источниках проверочных меток. "
        f"Для ориентира постоянная метка `{majority_label}` даёт accuracy "
        f"{majority_count / len(y):.3f} на {len(y)} человеческих оценках.", "",
        "| Представление | Классификатор | Macro-F1, человек, все | Accuracy, человек, все | "
        "Macro-F1, человек, общая | Macro-F1, ИИ, общая |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for rep in REPS:
        for clf in CLFS:
            pred = predictions[rep, clf]
            common = pred[common_human]
            lines.append(f"| {rep} | {clf} | {macro(y, pred):.3f} | "
                         f"{accuracy_score(y, pred):.3f} | {macro(y_shared, common):.3f} | "
                         f"{macro(ai_shared, common):.3f} |")

    lines += [""] + compare_review(y, videos, predictions, draws, 2026,
                                   title="Парные сравнения по человеческим меткам")
    lines += [
        "Протокол проверки использует оценки одного человека и сравнение с ИИ-метками.", "",
        "## Метрики по классам", "",
        "| Представление | Классификатор | Класс | n | Precision | Recall | F1 |",
        "|---|---|---|---:|---:|---:|---:|",
    ]
    for (rep, clf), pred in predictions.items():
        p, r, f1, n = precision_recall_fscore_support(y, pred, labels=LABELS, zero_division=0)
        for label, count, precision, recall, score in zip(LABELS, n, p, r, f1):
            lines.append(f"| {rep} | {clf} | {label} | {count} | {precision:.3f} | "
                         f"{recall:.3f} | {score:.3f} |")
    lines += [
        "", "## Воспроизведение", "",
        "Запустите из корня проекта в PowerShell с заполненным `reviewer_A.csv`.", "",
        "```powershell",
        "$env:PYTHONHASHSEED = '2026'",
        "python -m src.eval.human_review --prepare",
        "python -m src.models.run_review_holdout --fasttext-training-videos "
        "--reps tfidf_word tfidf_char fasttext xlmr --clf logreg svm "
        "--test data/interim/human_review/holdout.csv --out data/interim/human_review/predictions.csv",
        "python -m src.eval.human_review",
        "```", "",
    ]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare", action="store_true", help="write the human-labeled holdout")
    parser.add_argument("--draws", type=int, default=10000)
    args = parser.parse_args()
    if args.draws < 100:
        parser.error("--draws must be at least 100")
    human, ai = references()
    rows = holdout(human, ai)
    if args.prepare:
        with TEST.open("w", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=next(iter(rows.values())))
            writer.writeheader()
            writer.writerows(rows.values())
        print(f"{len(rows)} comments -> {TEST}")
        return
    OUTPUT.write_text(compare(human, ai, rows, args.draws), encoding="utf-8")
    print(OUTPUT)


if __name__ == "__main__":
    main()
