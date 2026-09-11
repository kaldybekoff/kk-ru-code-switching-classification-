"""Rough code-switching screen — used to decide WHICH SOURCES to collect from.

    python -m src.preprocess.cs_screen                  # screen everything in data/raw/
    python -m src.preprocess.cs_screen --by-video       # per-video breakdown
    python -m src.preprocess.cs_screen --show 20        # print sample mixed comments

This is a *screening heuristic*, not the density metric from the annotation
guideline. It exists to answer one question cheaply: is a given channel worth
annotating, or is its audience writing monolingually?

Method: word-level voting with two closed word lists. A word counts as Kazakh if
it contains a Kazakh-specific letter or is a frequent Kazakh function word; as
Russian if it is a frequent Russian function word. Shared borrowings (интернет,
видео, телефон) are in a stoplist and vote for neither, per the borrowing vs.
switching distinction in Dogruoz et al. (ACL 2021).

Known limits, do not paper over them in the report:
  - Recall is bounded by the word lists. Content words absent from both lists are
    invisible, so the true mixed share is HIGHER than what this prints.
  - Kazakh typed without its special letters (keremet for keremet) is missed
    unless the word is in the Kazakh function list.
  - English insertions are ignored entirely.
Treat the output as a lower bound and a ranking tool, nothing more.
"""

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"

KK_LETTERS = set("әғқңөұүһі")

# Frequent Kazakh function words, including the common spellings that drop the
# Kazakh-specific letters.
KK_WORDS = {
    "мен", "сен", "ол", "біз", "сіз", "олар", "бұл", "осы", "сол", "бул", "осынау",
    "және", "жане", "бірақ", "бирак", "үшін", "ушин", "сияқты", "сиякты", "деп",
    "ғой", "гой", "го", "ма", "ме", "ба", "бе", "па", "пе", "қой", "кой",
    "бар", "жоқ", "жок", "көп", "коп", "аз", "жақсы", "жаксы", "жаман", "керемет",
    "тым", "өте", "оте", "тағы", "тагы", "әлі", "али", "енді", "енди", "әрине",
    "болды", "болған", "болган", "екен", "еді", "еди", "ғой", "емес", "керек",
    "қалай", "калай", "қайда", "кайда", "неге", "кім", "ким", "не", "қашан",
    "рахмет", "рақмет", "ракмет", "сәлем", "салем", "ассалаумағалейкум",
    "аға", "ага", "әпке", "апке", "апа", "ата", "тәте", "тате", "бауырым",
    "сынып", "оқимын", "окимын", "көрдім", "кордим", "айтты", "деген", "дейді",
    # Kazakh particles that are homographs of Russian function words. Listing them
    # on both sides puts them in AMBIGUOUS, so they vote for neither.
    "да", "де", "та", "те", "ал", "о",
}

# Frequent Russian function words and high-frequency adverbs/particles.
RU_WORDS = {
    "и", "в", "не", "на", "я", "быть", "он", "с", "что", "а", "по", "это", "она",
    "этот", "к", "но", "они", "мы", "как", "из", "у", "который", "то", "за", "свой",
    "весь", "год", "от", "так", "о", "для", "ты", "же", "все", "всё", "тот", "мочь",
    "вы", "человек", "такой", "его", "сказать", "только", "или", "ещё", "еще", "бы",
    "себя", "один", "как", "уже", "до", "когда", "вот", "кто", "да", "нет", "очень",
    "если", "нас", "них", "меня", "тебя", "нам", "вам", "им", "её", "ее", "их",
    "сразу", "конечно", "просто", "тоже", "также", "потом", "теперь", "почему",
    "давай", "давайте", "можно", "нужно", "надо", "хорошо", "плохо", "круто",
    "спасибо", "пожалуйста", "привет", "пока", "лучше", "больше", "меньше",
    "два", "три", "четыре", "пять", "сто", "двести", "тысяча", "первый", "второй",
    "где", "куда", "зачем", "сколько", "какой", "чтобы", "потому", "хотя", "ведь",
}

# Borrowings and internationalisms shared by both languages - vote for neither.
SHARED = {
    "интернет", "видео", "видос", "видио", "телефон", "телевизор", "компьютер",
    "лайк", "лайкь", "репост", "подписка", "канал", "стрим", "контент", "блог",
    "комментарий", "коммент", "топ", "супер", "окей", "ок", "пж", "плиз",
    "майнкрафт", "роблокс", "гта", "ютуб", "ютубер", "тикток", "инстаграм",
    "мама", "папа", "школа", "класс", "серия", "фото", "клип", "музыка",
}

# Words present in both lists (kk "да/не/о" vs ru "да/не") carry no evidence -
# counting them as Russian was producing false "mixed" hits on pure Kazakh text.
AMBIGUOUS = KK_WORDS & RU_WORDS
NEUTRAL = SHARED | AMBIGUOUS

WORD_RE = re.compile(r"[Ѐ-ӿa-zA-Z]+")


def classify(text):
    """-> (n_kk, n_ru, n_counted). Words voting for neither are excluded."""
    kk = ru = 0
    for w in WORD_RE.findall(text.lower()):
        if w in NEUTRAL:
            continue
        if w in RU_WORDS:
            ru += 1
        elif w in KK_WORDS or (set(w) & KK_LETTERS):
            kk += 1
    return kk, ru, kk + ru


def bucket(kk, ru):
    """Guideline buckets applied to the screen's own counts."""
    total = kk + ru
    if total == 0:
        return "unknown"
    if kk == 0 or ru == 0:
        return "mono"
    share = min(kk, ru) / total
    if share <= 0.20:
        return "low"
    if share <= 0.40:
        return "med"
    return "high"


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    p = argparse.ArgumentParser()
    p.add_argument("--by-video", action="store_true")
    p.add_argument("--show", type=int, default=0, help="print N sample mixed comments")
    p.add_argument("--infile")
    args = p.parse_args()

    files = [Path(args.infile)] if args.infile else sorted(RAW.glob("*.jsonl"))
    if not files:
        raise SystemExit("No raw JSONL in data/raw/.")

    overall = Counter()
    per_video = defaultdict(Counter)
    mixed_samples = []

    for fp in files:
        for line in open(fp, encoding="utf-8"):
            if not line.strip():
                continue
            c = json.loads(line)
            kk, ru, _ = classify(c.get("text", ""))
            b = bucket(kk, ru)
            overall[b] += 1
            per_video[c.get("video_id", "?")][b] += 1
            if b in ("low", "med", "high") and len(mixed_samples) < 200:
                mixed_samples.append((b, c["text"][:100]))

    total = sum(overall.values())
    mixed = overall["low"] + overall["med"] + overall["high"]
    print("screened {} comments\n".format(total))
    for b in ("mono", "low", "med", "high", "unknown"):
        n = overall[b]
        print("  {:<8} {:>6}  {:>5.1f}%".format(b, n, n / total * 100 if total else 0))
    print("\nMIXED (low+med+high): {} = {:.1f}% of all, "
          "{:.1f}% of comments with a language signal".format(
              mixed, mixed / total * 100 if total else 0,
              mixed / (total - overall["unknown"]) * 100 if total - overall["unknown"] else 0))
    print("\nThis is a LOWER BOUND - the word lists miss content words.")

    if args.by_video:
        print("\n=== per video (ranked by mixed share) ===")
        rows = []
        for vid, c in per_video.items():
            t = sum(c.values())
            m = c["low"] + c["med"] + c["high"]
            rows.append((m / t if t else 0, vid, t, m, c["mono"], c["unknown"]))
        rows.sort(reverse=True)
        print("{:<14} {:>6} {:>7} {:>8} {:>6} {:>8}".format(
            "video_id", "total", "mixed", "mixed%", "mono", "unknown"))
        for share, vid, t, m, mono, unk in rows:
            print("{:<14} {:>6} {:>7} {:>7.1f}% {:>6} {:>8}".format(
                vid, t, m, share * 100, mono, unk))

    if args.show:
        print("\n=== sample mixed comments ===")
        for b, txt in mixed_samples[: args.show]:
            print("[{:<4}] {}".format(b, txt))


if __name__ == "__main__":
    main()
