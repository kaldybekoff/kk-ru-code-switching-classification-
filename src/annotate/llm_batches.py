"""Build paste-ready LLM annotation prompts from the batch CSVs.

    python -m src.annotate.llm_batches
    python -m src.annotate.llm_batches --per-prompt 40 --only enriched_01

Writes data/interim/llm/prompt_XX.txt - each one a complete, self-contained
prompt: the sentiment rules from annotation/guideline.md, then N comments, then
a strict JSON output contract. Paste one into the model, paste the JSON reply
into data/interim/llm/reply_XX.json, then run:

    python -m src.annotate.llm_ingest

## What the model is and is not asked for

Asked: `sentiment`, `base_lang`, a `confidence` flag and a short `note`.

**Not asked: `density`.** That label is produced by a deterministic algorithm
(src/preprocess/density.py), not by a language model, and the reason is not
procedural. H2 tests whether a multilingual transformer degrades as mixing
rises. A large multilingual LLM shares that architecture's blind spots, so the
comments it fails to recognise as mixed are disproportionately the ones XLM-R
also mishandles. Letting it set the density buckets would make the
stratification variable correlate with the very effect being measured, and no
amount of spot-checking afterwards could separate them.

`confidence: low` is what drives the audit queue required by the sprint plan -
those rows are surfaced for human review rather than trusted.
"""

import argparse
import csv
import glob
import json
import textwrap
from pathlib import Path

from src.preprocess.textnorm import utf8_stdout

ROOT = Path(__file__).resolve().parents[2]
BATCHES = ROOT / "data" / "interim" / "batches"
OUTDIR = ROOT / "data" / "interim" / "llm"

INSTRUCTIONS = """\
Ты размечаешь тональность комментариев YouTube для научного корпуса
казахско-русского code-switching (Academic Beta Career 2026).

Комментарии смешивают казахский и русский, часто в одном предложении, с
неформальной орфографией и пропущенными казахскими буквами (ә ғ қ ң ө ұ ү һ і).
Размечай СМЫСЛ, а не язык.

## Метка 1 — sentiment

  pos   похвала, благодарность, восхищение, поддержка, симпатия
  neg   критика, злость, насмешка, оскорбление, разочарование, отвращение
  neu   вопросы, фактические утверждения, ремарки без оценочной окраски
  skip  спам/самореклама, голая ссылка, нечитаемый текст, один символ,
        текст не на казахском и не на русском, только тайм-код

Правила:
1. Оценивай тональность САМОГО комментария, а не темы. Ровное упоминание
   тяжёлой темы — это neu, а не neg.
2. Мишень не важна: негатив к ведущему, гостю, другому комментатору или
   стране — всё neg.
3. Эмодзи считаются наравне с текстом: "Керемет 🔥🔥🔥" = pos,
   "Ведущий 🤡" = neg.
4. Сарказм размечай по ПОДРАЗУМЕВАЕМОМУ смыслу. Если ирония не
   восстанавливается уверенно — neu.
5. Смешанная тональность: бери доминирующую часть. Если действительно
   сбалансировано — neu.
6. Религиозные формулы ("Аллаға шүкір", "Иншалла") — pos, если поддерживают
   содержание, иначе neu.
7. Сочувствие человеку без оценки чего-либо — neu.
8. Когда искренне колеблешься между двумя метками — ставь neu. Это значение
   по умолчанию.

## Метка 2 — base_lang

Язык, дающий БОЛЬШИНСТВО значимых токенов: kk (казахский) или ru (русский).
Имена собственные и заимствования, общие для обоих языков (интернет, видео,
телефон), не считаются. Если skip — оставь пустой строкой "".

## Метка 3 — confidence

  high  правило применяется однозначно
  low   ты колебался, текст неоднозначен, сарказм неясен, смысл размыт

Ставь low честно. Эти строки уходят человеку на аудит, поэтому занижение
уверенности ничего не ломает, а завышение — ломает.

## Метка 4 — note

Если confidence = low, напиши 3-8 слов почему. Иначе пустая строка "".

## Формат ответа

Верни ТОЛЬКО валидный JSON-массив, без markdown-обёртки, без пояснений.
Ровно один объект на каждый входной комментарий, в том же порядке:

[
  {"id": "<id>", "sentiment": "pos", "base_lang": "kk", "confidence": "high", "note": ""},
  {"id": "<id>", "sentiment": "skip", "base_lang": "", "confidence": "high", "note": ""},
  {"id": "<id>", "sentiment": "neu", "base_lang": "ru", "confidence": "low", "note": "сарказм неясен"}
]

Никаких других полей. Не пропускай и не объединяй строки.
"""


def read_batch(path):
    with open(path, encoding="utf-8-sig") as f:
        return [r for r in csv.DictReader(f) if (r.get("comment_id") or "").strip()]


def main():
    utf8_stdout()
    p = argparse.ArgumentParser()
    p.add_argument("--per-prompt", type=int, default=40,
                   help="comments per prompt; keep it small enough that the "
                        "model's reply is not truncated mid-array")
    p.add_argument("--only", nargs="*", default=[],
                   help="restrict to these batch names, e.g. enriched_01")
    p.add_argument("--include-annotated", action="store_true",
                   help="also send rows that already carry a sentiment")
    args = p.parse_args()

    files = sorted(glob.glob(str(BATCHES / "*.csv")))
    files = [f for f in files if "manifest" not in Path(f).name]
    if args.only:
        files = [f for f in files if Path(f).stem in args.only]
    if not files:
        raise SystemExit("No batch CSVs found. Run make_batches first.")

    todo = []
    for f in files:
        for r in read_batch(f):
            if not args.include_annotated and (r.get("sentiment") or "").strip():
                continue
            todo.append({"id": r["comment_id"], "text": r["text"],
                         "batch": Path(f).stem})

    if not todo:
        raise SystemExit("Every row already has a sentiment. Nothing to send.")

    OUTDIR.mkdir(parents=True, exist_ok=True)
    for old in OUTDIR.glob("prompt_*.txt"):
        old.unlink()

    n_prompts = 0
    index = []
    for i in range(0, len(todo), args.per_prompt):
        chunk = todo[i:i + args.per_prompt]
        n_prompts += 1
        name = "prompt_{:02d}".format(n_prompts)

        body = ["", "## Комментарии ({} штук)".format(len(chunk)), ""]
        for c in chunk:
            # One line per comment; newlines inside a comment would break the
            # visual boundary between items when pasted.
            flat = " ".join(c["text"].split())
            body.append('{{"id": "{}", "text": {}}}'.format(
                c["id"], json.dumps(flat, ensure_ascii=False)))
        body += ["", "Верни JSON-массив из ровно {} объектов.".format(len(chunk))]

        (OUTDIR / (name + ".txt")).write_text(
            INSTRUCTIONS + "\n".join(body) + "\n", encoding="utf-8")
        index.append({"prompt": name, "n": len(chunk),
                      "ids": [c["id"] for c in chunk]})

    (OUTDIR / "index.json").write_text(
        json.dumps(index, ensure_ascii=False, indent=1), encoding="utf-8")

    print("{} comments awaiting annotation -> {} prompts of up to {}".format(
        len(todo), n_prompts, args.per_prompt))
    print("\n-> {}".format(OUTDIR))
    print(textwrap.dedent("""
        Loop, one prompt at a time:
          1. open data/interim/llm/prompt_01.txt, copy all of it
          2. paste into the model, let it answer
          3. save the JSON reply as data/interim/llm/reply_01.json
          4. repeat for prompt_02, prompt_03, ...

        Then:
          python -m src.annotate.llm_ingest

        Density is NOT asked for here and must not be added by hand from the
        model - it is computed by src/preprocess/density.py. See this module's
        docstring for why that separation is load-bearing.
        """))


if __name__ == "__main__":
    main()
