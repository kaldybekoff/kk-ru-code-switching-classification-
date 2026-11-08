# Спринт 5. Проверка по человеческой разметке

Один человек оценил тональность 100 комментариев вслепую. Положительных — 46, отрицательных — 16, нейтральных — 34, пропущенных — 4. В проверку моделей вошли 96 комментариев с 74 видео.

Все системы обучались на исходных 557 комментариях с ИИ-метками. Для 2 комментариев, которые ИИ ранее пропустил, предсказания восстановлены с прежними параметрами и обучающими данными. На общей выборке все предсказания совпали с сохранёнными.

## Согласие человека и ИИ

На всех комментариях, включая `skip`, совпали 69 из 100 меток (κ Коэна = 0.533). На комментариях с валидной оценкой у обоих разметчиков совпали 69 из 94 меток (κ = 0.589).

| Человек / ИИ | pos | neg | neu | skip |
|---|---:|---:|---:|---:|
| pos | 38 | 3 | 4 | 1 |
| neg | 0 | 14 | 1 | 1 |
| neu | 5 | 12 | 17 | 0 |
| skip | 2 | 0 | 2 | 0 |

Модель Cardiff, которая дала обучающие метки, совпала с человеком в 62 из 96 комментариев (κ = 0.455). В этом сравнении учитываются комментарии, которым человек присвоил один из трёх классов тональности.

## Метрики восьми систем

Macro-F1 на общей выборке из 94 комментариев сравнивает один и тот же набор предсказаний при двух источниках проверочных меток. Для ориентира постоянная метка `pos` даёт accuracy 0.479 на 96 человеческих оценках.

| Представление | Классификатор | Macro-F1, человек, все | Accuracy, человек, все | Macro-F1, человек, общая | Macro-F1, ИИ, общая |
|---|---|---:|---:|---:|---:|
| tfidf_word | logreg | 0.466 | 0.490 | 0.468 | 0.481 |
| tfidf_word | svm | 0.407 | 0.438 | 0.408 | 0.444 |
| tfidf_char | logreg | 0.564 | 0.594 | 0.559 | 0.565 |
| tfidf_char | svm | 0.550 | 0.573 | 0.546 | 0.549 |
| fasttext | logreg | 0.427 | 0.562 | 0.435 | 0.479 |
| fasttext | svm | 0.445 | 0.552 | 0.454 | 0.472 |
| xlmr | logreg | 0.431 | 0.490 | 0.440 | 0.360 |
| xlmr | svm | 0.425 | 0.531 | 0.434 | 0.342 |

### Парные сравнения по человеческим меткам

96 comments in 74 videos. Difference is first system minus second. 95% percentile intervals resample videos (10000 draws; seed 2026). The two-sided macro-F1 permutation swaps predictions for whole videos; Holm correction covers all 16 comparisons below. Exact McNemar p-values test paired accuracy and are unadjusted diagnostics, not the primary test.

| First | Second | Δ macro-F1 | 95% video CI | permutation p | Holm p | McNemar n01/n10 | McNemar p |
|---|---|---:|---:|---:|---:|---:|---:|
| tfidf_word+logreg | tfidf_char+logreg | -0.098 | [-0.203, +0.009] | 0.0818 | 0.9815 | 20/10 | 0.0987 |
| tfidf_word+logreg | fasttext+logreg | +0.039 | [-0.061, +0.133] | 0.5848 | 1.0000 | 23/16 | 0.3368 |
| tfidf_word+logreg | xlmr+logreg | +0.035 | [-0.092, +0.165] | 0.6047 | 1.0000 | 22/22 | 1.0000 |
| tfidf_char+logreg | fasttext+logreg | +0.137 | [+0.038, +0.226] | 0.0400 | 0.5999 | 9/12 | 0.6636 |
| tfidf_char+logreg | xlmr+logreg | +0.133 | [-0.002, +0.265] | 0.0655 | 0.9169 | 11/21 | 0.1102 |
| fasttext+logreg | xlmr+logreg | -0.003 | [-0.104, +0.094] | 0.9611 | 1.0000 | 8/15 | 0.2100 |
| tfidf_word+svm | tfidf_char+svm | -0.143 | [-0.260, -0.023] | 0.0243 | 0.3888 | 23/10 | 0.0351 |
| tfidf_word+svm | fasttext+svm | -0.038 | [-0.150, +0.075] | 0.5782 | 1.0000 | 26/15 | 0.1173 |
| tfidf_word+svm | xlmr+svm | -0.018 | [-0.160, +0.119] | 0.8210 | 1.0000 | 28/19 | 0.2430 |
| tfidf_char+svm | fasttext+svm | +0.105 | [-0.019, +0.219] | 0.1492 | 1.0000 | 12/14 | 0.8450 |
| tfidf_char+svm | xlmr+svm | +0.125 | [-0.012, +0.250] | 0.1056 | 1.0000 | 16/20 | 0.6177 |
| fasttext+svm | xlmr+svm | +0.020 | [-0.086, +0.121] | 0.7088 | 1.0000 | 11/13 | 0.8388 |
| tfidf_word+logreg | tfidf_word+svm | +0.059 | [+0.002, +0.118] | 0.0664 | 0.9169 | 2/7 | 0.1797 |
| tfidf_char+logreg | tfidf_char+svm | +0.014 | [-0.068, +0.095] | 0.7199 | 1.0000 | 5/7 | 0.7744 |
| fasttext+logreg | fasttext+svm | -0.018 | [-0.072, +0.023] | 0.8564 | 1.0000 | 1/2 | 1.0000 |
| xlmr+logreg | xlmr+svm | +0.005 | [-0.066, +0.085] | 0.8463 | 1.0000 | 6/2 | 0.2891 |

Holm-adjusted p < 0.05: 0 of 16.

Протокол проверки использует оценки одного человека и сравнение с ИИ-метками.

## Метрики по классам

| Представление | Классификатор | Класс | n | Precision | Recall | F1 |
|---|---|---|---:|---:|---:|---:|
| tfidf_word | logreg | pos | 46 | 0.684 | 0.565 | 0.619 |
| tfidf_word | logreg | neg | 16 | 0.300 | 0.562 | 0.391 |
| tfidf_word | logreg | neu | 34 | 0.429 | 0.353 | 0.387 |
| tfidf_word | svm | pos | 46 | 0.595 | 0.543 | 0.568 |
| tfidf_word | svm | neg | 16 | 0.259 | 0.438 | 0.326 |
| tfidf_word | svm | neu | 34 | 0.370 | 0.294 | 0.328 |
| tfidf_char | logreg | pos | 46 | 0.867 | 0.565 | 0.684 |
| tfidf_char | logreg | neg | 16 | 0.348 | 0.500 | 0.410 |
| tfidf_char | logreg | neu | 34 | 0.535 | 0.676 | 0.597 |
| tfidf_char | svm | pos | 46 | 0.812 | 0.565 | 0.667 |
| tfidf_char | svm | neg | 16 | 0.346 | 0.562 | 0.429 |
| tfidf_char | svm | neu | 34 | 0.526 | 0.588 | 0.556 |
| fasttext | logreg | pos | 46 | 0.838 | 0.674 | 0.747 |
| fasttext | logreg | neg | 16 | 0.000 | 0.000 | 0.000 |
| fasttext | logreg | neu | 34 | 0.442 | 0.676 | 0.535 |
| fasttext | svm | pos | 46 | 0.833 | 0.652 | 0.732 |
| fasttext | svm | neg | 16 | 0.091 | 0.062 | 0.074 |
| fasttext | svm | neu | 34 | 0.449 | 0.647 | 0.530 |
| xlmr | logreg | pos | 46 | 0.769 | 0.435 | 0.556 |
| xlmr | logreg | neg | 16 | 0.250 | 0.188 | 0.214 |
| xlmr | logreg | neu | 34 | 0.414 | 0.706 | 0.522 |
| xlmr | svm | pos | 46 | 0.765 | 0.565 | 0.650 |
| xlmr | svm | neg | 16 | 0.143 | 0.062 | 0.087 |
| xlmr | svm | neu | 34 | 0.436 | 0.706 | 0.539 |

## Воспроизведение

Запустите из корня проекта в PowerShell с заполненным `reviewer_A.csv`.

```powershell
$env:PYTHONHASHSEED = '2026'
python -m src.eval.human_review --prepare
python -m src.models.run_review_holdout --fasttext-training-videos --reps tfidf_word tfidf_char fasttext xlmr --clf logreg svm --test data/interim/human_review/holdout.csv --out data/interim/human_review/predictions.csv
python -m src.eval.human_review
```
