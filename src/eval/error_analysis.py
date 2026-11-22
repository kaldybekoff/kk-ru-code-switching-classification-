"""Reproduce the Week 11–12 analysis from the frozen human review predictions."""

from collections import Counter

import numpy as np
from sklearn.metrics import confusion_matrix

from src.eval.density_groups import COUNTS, load_groups
from src.eval.human_review import PREDICTIONS, holdout, references, signatures
from src.eval.sprint5 import BUCKETS, CLFS, LABELS, REPS, ROOT, macro, read_grid

OUT = ROOT / "reports/sprint6_error_analysis_ru.md"
NAMES = {"tfidf_word": "Словный TF-IDF", "tfidf_char": "Символьный TF-IDF",
         "fasttext": "fastText", "xlmr": "XLM-R"}

# These are qualitative observations of selected errors, not a coded prevalence sample.
# The case number is the 1-based index when the 96 human holdout IDs are sorted.
CASES = [
    (2, "pos", "neg", "Отрицание и контраст", "Автор называет слёзы проявлением смелости. Фраза содержит «не слабость»."),
    (6, "pos", "neu", "Смешение языков", "Автор хвалит подкаст за отсутствие рекламы. Положительную оценку выражает «тыңдауға кайф»."),
    (9, "neu", "neg", "Разговорная пунктуация", "Вопрос о том, где купить одежду, завершается «?!» и обращением на казахском."),
    (13, "pos", "neu", "Цитата и неоднозначность", "Автор приводит цитату о казахах, а отношение выражает преимущественно через контекст."),
    (17, "neg", "neu", "Неформальное написание", "Критика ведущего содержит удлинённые гласные в словах «бәлеее» и «қайтааа»."),
    (21, "pos", "neu", "Граница языкового охвата", "Слова в основном на узбекском. Это одна из строк, ранее пропущенных Codex."),
    (26, "pos", "neg", "Сленг", "«ЖЁСТКООО» и смех сопровождают явную положительную оценку."),
    (48, "neu", "pos", "Смешанная оценка", "Комплимент внешности соседствует с критикой макияжа в конкретном выпуске."),
    (51, "pos", "neg", "Эмодзи и контекст", "Слёзы и разбитое сердце сопровождают слова «так мило»."),
    (60, "neg", "neu", "Смешение и морфология", "Критика содержит русское «хайповать» в казахской конструкции."),
    (87, "pos", "neg", "Короткий текст", "В пяти словах автор описывает человека как простого и комфортного, добавляя 😍."),
    (89, "pos", "neg", "Короткий текст и вариативность", "Казахское пожелание счастья записано с пропусками характерных букв."),
]


def confusion_rows(y, prediction):
    return confusion_matrix(y, prediction, labels=LABELS)


def group_table(y, video, groups, predictions):
    lines = [
        "| Группа | n | Видео | pos/neg/neu | " + " | ".join(NAMES[r] for r in REPS) + " |",
        "|---|---:|---:|---:|" + "---:|" * len(REPS),
    ]
    for name, mask in groups:
        n = int(mask.sum())
        if not n:
            continue
        counts = Counter(y[mask])
        scores = " | ".join(f"{macro(y[mask], predictions[rep, 'logreg'][mask]):.3f}"
                            for rep in REPS)
        class_counts = "/".join(str(counts[label]) for label in LABELS)
        lines.append(f"| {name} | {n} | {len(set(video[mask]))} | {class_counts} | {scores} |")
    return lines


def render():
    human, ai = references()
    rows = holdout(human, ai)
    y, _, video, predictions, lengths = read_grid(PREDICTIONS, signatures(rows))
    counts = load_groups(COUNTS)
    density = np.array([counts.get(cid, "") for cid in sorted(rows)])
    lines = [
        "# Спринт 6. Анализ моделей и ошибок", "",
        f"Анализ использует основную проверочную выборку из {len(y)} комментариев с "
        f"{len(set(video))} видео. Во всех строках macro-F1 "
        "усредняется по положительному, отрицательному и нейтральному классам.", "",
        "## Профили восьми систем", "",
        "| Представление | Классификатор | Macro-F1 | Ошибки | Отрицательных распознано |",
        "|---|---|---:|---:|---:|",
    ]
    for rep in REPS:
        for clf in CLFS:
            prediction = predictions[rep, clf]
            matrix = confusion_rows(y, prediction)
            lines.append(f"| {NAMES[rep]} | {clf} | {macro(y, prediction):.3f} | "
                         f"{int((prediction != y).sum())}/{len(y)} | "
                         f"{matrix[1, 1]}/{int(matrix[1].sum())} |")

    lines += [
        "", "### Матрицы ошибок для логистической регрессии", "",
        "В каждой ячейке число прогнозов `pos / neg / neu` для указанного класса разметки.", "",
        "| Представление | Метка pos | Метка neg | Метка neu |",
        "|---|---:|---:|---:|",
    ]
    for rep in REPS:
        matrix = confusion_rows(y, predictions[rep, "logreg"])
        cells = [" / ".join(str(n) for n in row) for row in matrix]
        lines.append(f"| {NAMES[rep]} | " + " | ".join(cells) + " |")

    covered = int(np.count_nonzero(density != ""))
    missing = sum(cid not in counts for cid in rows)
    lines += [
        "", "## Плотность смешения", "",
        f"Замороженные подсчёты дают группу плотности для {covered} из {len(y)} текстов. "
        f"Для {missing} строк нет замороженного подсчёта, в остальных исключённых "
        "строках зачтено менее пяти слов. Все эти строки остаются в общих метриках. "
        "Подгрупповые результаты описательные. Доля языков оценена словарём, "
        "а отдельные группы малы и различаются по составу классов.", "",
    ]
    lines += group_table(y, video, [(band, density == band) for band in BUCKETS], predictions)
    lines += [
        "", "## Длина комментария", "",
        "Длина берётся из корпуса до группировки по плотности. Границы групп служат "
        "для описательного сравнения. Состав классов указан для каждой группы.", "",
    ]
    lines += group_table(y, video, [("≤6 слов", lengths <= 6),
                                    ("7–15 слов", (lengths >= 7) & (lengths <= 15)),
                                    ("≥16 слов", lengths >= 16)], predictions)
    lines += [
        "", "## Качественный разбор", "",
        "Разбор охватывает 12 отобранных ошибок символьного TF-IDF с логистической "
        "регрессией. Исходная метка и прогноз сохранены. Примеры показывают механизмы "
        "ошибок. Подсчёт частот этих механизмов потребовал бы разметить все ошибки "
        "по единому протоколу. Публичное описание сокращает исходные комментарии.", "",
        "| Случай | Метка | Прогноз | Наблюдение | Что видно в тексте |",
        "|---:|---|---|---|---|",
    ]
    char = predictions["tfidf_char", "logreg"]
    for case, actual, predicted, tag, explanation in CASES:
        if (y[case - 1], char[case - 1]) != (actual, predicted):
            raise ValueError(f"Reviewed case {case} does not match frozen predictions")
        lines.append(f"| {case} | {actual} | {predicted} | {tag} | {explanation} |")
    lines += [
        "", "Номер случая соответствует позиции в списке отсортированных ID "
        "из 96 строк проверки. [Метрики и статистические тесты](sprint5_human_review_ru.md)", "",
        "Воспроизведение из корня проекта", "",
        "```powershell",
        "python -m src.eval.error_analysis",
        "```", "",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    OUT.write_text(render(), encoding="utf-8")
    print(OUT)
