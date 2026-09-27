"""Deterministic code-switching density, computed rather than judged.

    python -m src.preprocess.density --build      # fit the lexicon, then label
    python -m src.preprocess.density --show 25    # inspect what it decided
    python -m src.preprocess.density --eval data/interim/llm/audit_queue.csv

Writes `density` and `base_lang` into data/interim/batches/*.csv, and the
fitted lexicon to models_cache/density_lexicon.json so a run is reproducible.

## Why this is not done by a language model

The density bucket is the variable H2 is measured against: it asks whether a
multilingual transformer's advantage shrinks as mixing rises. If the buckets
were assigned by a large multilingual model, the comments it failed to see as
mixed would be disproportionately the ones XLM-R also mishandles - the
stratification variable would correlate with the effect under test, and the
result would be uninterpretable in a way no spot-check could repair.

So density comes from a transparent, deterministic procedure with a different
failure mode, and its errors are measured against human labels rather than
assumed away.

## Method

Kazakh and Russian share the Cyrillic alphabet, so there is no script signal.
The lexicon is bootstrapped from the corpus itself in four stages:

1. **Document-level seeds.** Comments that are confidently monolingual are
   identified first - Kazakh ones by a density of Kazakh-specific letters with
   no Russian function words, Russian ones by the absence of any Kazakh letter
   plus several Russian function words. Every word is then scored by how it
   distributes between the two halves. Seeding at the document level rather
   than the word level is what keeps the two sides balanced: Kazakh words
   announce themselves through ә/ғ/қ/ң/ө/ұ/ү/һ/і and Russian has no equivalent
   marker, so a letter-based seed yields roughly ten times more Kazakh types
   and skews every model fitted on it.

2. **Character n-gram models.** Order-3 models with add-k smoothing are fitted
   to each seed set. This is what generalises beyond the seeds - it learns that
   `-дағы`, `-ымен`, `-тар` look Kazakh and `-ость`, `-ение`, `-ами` look
   Russian, which is exactly the morphology the closed word lists in
   cs_screen.py cannot reach.

3. **Classification.** Every vocabulary word above a frequency floor is scored
   by log-likelihood ratio. Words inside a margin around zero stay `ambiguous`
   and vote for neither - guessing them would manufacture switches.

4. **Diacritic-free Kazakh.** 59% of this corpus contains no Kazakh-specific
   letter at all, and the document-level seeds above require those letters, so
   Kazakh typed without them is invisible to the seeding stage and the
   character models score it as Russian. `ұры` (thief) written `уры` was being
   filed as a Russian word, which turns a monolingual Kazakh insult into a
   `high`-density comment. Every confident Kazakh word is therefore also
   indexed under its stripped form (ұры -> уры, сілімтігі -> силимтиги), and an
   unmarked word matching that index is Kazakh - unless the stripped form is
   also a real Russian word (боқ -> бок), in which case it is ambiguous and
   votes for neither.

5. **Shared-word detection.** A word used freely on both sides is a borrowing
   (телефон, подкаст), not evidence of a switch. Loans too rare in the Kazakh
   half for a ratio to be meaningful are listed explicitly in `SHARED_SEED`.
   Per annotation/guideline.md these count as the base language, so they are
   excluded from the minority count entirely.

Proper names, numbers, emoji, URLs and @mentions are excluded throughout, as
the guideline requires.
"""

import argparse
import csv
import glob
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path

from src.preprocess.textnorm import utf8_stdout

ROOT = Path(__file__).resolve().parents[2]
CLEAN = ROOT / "data" / "interim" / "corpus_clean.jsonl"
BATCHES = ROOT / "data" / "interim" / "batches"
CACHE = ROOT / "models_cache"
LEXICON = CACHE / "density_lexicon.json"

WORD_RE = re.compile(r"[Ѐ-ӿa-zA-Z]+")
KK_LETTERS = set("әғқңөұүһі")
# Letters that do not occur in native Kazakh words. в/ф/ц/ч are excluded from
# this list: modern Kazakh uses them in established loans, so they are weak
# evidence and would poison the Russian seed.
RU_ONLY_LETTERS = set("щъё")

RU_FUNCTION = {
    "и", "в", "не", "на", "я", "он", "с", "что", "по", "это", "она", "этот",
    "к", "но", "они", "мы", "как", "из", "у", "который", "за", "свой", "весь",
    "от", "так", "для", "ты", "же", "все", "всё", "тот", "вы", "такой", "его",
    "только", "или", "ещё", "еще", "бы", "себя", "один", "уже", "до", "когда",
    "вот", "кто", "нет", "очень", "если", "нас", "них", "меня", "тебя", "нам",
    "вам", "им", "её", "ее", "их", "сразу", "конечно", "просто", "тоже",
    "также", "потом", "теперь", "почему", "можно", "нужно", "надо", "где",
    "куда", "зачем", "сколько", "какой", "чтобы", "потому", "хотя", "ведь",
    "был", "была", "было", "были", "есть", "будет", "быть", "мне", "тебе",
}
# Kazakh function words that are homographs of Russian ones, or that are
# frequent enough to seed reliably without a special letter.
KK_FUNCTION = {
    "бул", "осы", "сол", "жане", "бирак", "ушин", "сиякты", "деп", "гой",
    "жок", "коп", "жаксы", "жаман", "керемет", "оте", "енди", "болды",
    "болган", "екен", "еди", "емес", "керек", "калай", "кайда", "неге",
    "рахмет", "ракмет", "салем", "бар", "мен", "сен", "биз", "олар", "кыз",
}
# Homographs that carry no evidence either way.
STOP_BOTH = {"да", "де", "та", "те", "ал", "о", "не", "а", "и", "бар", "мен"}

# Internationalisms and platform vocabulary shared by both languages. The
# frequency test below catches borrowings that are common on both sides, but a
# loan can be rare in the Kazakh half of this corpus and still not be a switch
# when it appears there - `интернет` occurs 18 times in total, far too few for
# a ratio to mean anything. Per annotation/guideline.md these count as the base
# language, never as evidence of mixing, so they are listed explicitly.
SHARED_SEED = {
    "интернет", "видео", "видос", "видио", "телефон", "телевизор", "компьютер",
    "лайк", "репост", "подписка", "канал", "стрим", "контент", "блог", "влог",
    "комментарий", "коммент", "топ", "супер", "окей", "ок", "пж", "плиз",
    "ютуб", "ютюб", "тикток", "инстаграм", "фейсбук", "телеграм", "вайб",
    "подкаст", "интервью", "эфир", "выпуск", "сезон", "серия", "клип",
    "музыка", "фото", "камера", "микрофон", "монтаж", "студия", "продюсер",
    "менеджер", "бизнес", "маркетинг", "проект", "старт", "финал", "гость",
    "школа", "класс", "университет", "институт", "доктор", "автобус", "такси",
    "банк", "кредит", "телеграмм", "мем", "хайп", "кринж", "респект",
}

# Kazakh letters mapped to the bare Cyrillic people type instead of them, plus
# the й/и confusion that comes with typing fast without a Kazakh layout.
DEACCENT = str.maketrans({
    "ә": "а", "ғ": "г", "қ": "к", "ң": "н", "ө": "о",
    "ұ": "у", "ү": "у", "һ": "х", "і": "и", "й": "и",
})


def deaccent(word):
    return word.translate(DEACCENT)


NGRAM = 3
SMOOTH = 0.5
MIN_FREQ = 3            # vocabulary floor
LLR_MARGIN = 0.35       # per-character log-ratio below this stays ambiguous
SHARED_MARGIN = 0.30    # context balance that marks a word as shared


def words(text):
    return [w.lower() for w in WORD_RE.findall(text)]


class CharLM:
    """Order-n character model over padded words, add-k smoothed."""

    def __init__(self, n=NGRAM):
        self.n = n
        self.ctx = Counter()
        self.gram = Counter()
        self.alphabet = set()

    def add(self, word, weight=1):
        s = "^" * (self.n - 1) + word + "$"
        self.alphabet.update(word)
        for i in range(self.n - 1, len(s)):
            self.ctx[s[i - self.n + 1:i]] += weight
            self.gram[s[i - self.n + 1:i + 1]] += weight

    def logp(self, word):
        """Mean log-probability per character - length-normalised so long words
        are not automatically scored as less likely."""
        s = "^" * (self.n - 1) + word + "$"
        v = max(len(self.alphabet), 1)
        total = 0.0
        steps = 0
        for i in range(self.n - 1, len(s)):
            c, g = s[i - self.n + 1:i], s[i - self.n + 1:i + 1]
            total += math.log((self.gram[g] + SMOOTH) /
                              (self.ctx[c] + SMOOTH * v))
            steps += 1
        return total / max(steps, 1)

    def to_dict(self):
        return {"n": self.n, "ctx": dict(self.ctx), "gram": dict(self.gram),
                "alphabet": "".join(sorted(self.alphabet))}

    @classmethod
    def from_dict(cls, d):
        lm = cls(d["n"])
        lm.ctx = Counter(d["ctx"])
        lm.gram = Counter(d["gram"])
        lm.alphabet = set(d["alphabet"])
        return lm


def seed_of(word):
    """-> 'kk' | 'ru' | None, for the bootstrap stage only."""
    if word in STOP_BOTH:
        return None
    if set(word) & KK_LETTERS:
        return "kk"
    if word in KK_FUNCTION:
        return "kk"
    if set(word) & RU_ONLY_LETTERS:
        return "ru"
    if word in RU_FUNCTION:
        return "ru"
    return None


def classify_docs(texts):
    """Split the corpus into confidently-monolingual halves.

    Word-level seeding alone is badly unbalanced on this corpus: Kazakh words
    announce themselves through ә/ғ/қ/ң/ө/ұ/ү/һ/і while Russian has no
    equivalent marker, so a letter-based seed yields ~10x more Kazakh types and
    the resulting models are skewed toward Kazakh. Seeding at the *document*
    level fixes that - a comment with no Kazakh-specific letter anywhere and
    several Russian function words is confidently Russian, and every word in it
    is Russian evidence, marker or not.
    """
    kk_docs, ru_docs = [], []
    for t in texts:
        ws = words(t)
        if len(ws) < 4:
            continue
        kk_marked = sum(1 for w in ws if set(w) & KK_LETTERS)
        ru_marked = sum(1 for w in ws if w in RU_FUNCTION or set(w) & RU_ONLY_LETTERS)
        if kk_marked >= 2 and kk_marked / len(ws) >= 0.25 and ru_marked == 0:
            kk_docs.append(ws)
        elif kk_marked == 0 and ru_marked >= 2:
            ru_docs.append(ws)
    return kk_docs, ru_docs


def build_lexicon(texts, verbose=True):
    kk_docs, ru_docs = classify_docs(texts)
    if verbose:
        print("confidently monolingual: {} kk docs / {} ru docs".format(
            len(kk_docs), len(ru_docs)))
    if len(kk_docs) < 100 or len(ru_docs) < 100:
        raise SystemExit("Too few monolingual documents to bootstrap from.")

    kk_freq, ru_freq = Counter(), Counter()
    for ws in kk_docs:
        kk_freq.update(ws)
    for ws in ru_docs:
        ru_freq.update(ws)
    # Normalise for corpus imbalance: raw counts would let the larger side win
    # every contested word.
    kk_total = max(sum(kk_freq.values()), 1)
    ru_total = max(sum(ru_freq.values()), 1)

    freq = Counter()
    for t in texts:
        freq.update(words(t))

    seeds = {}
    for w in set(kk_freq) | set(ru_freq):
        if w in STOP_BOTH:
            continue
        a = kk_freq[w] / kk_total
        b = ru_freq[w] / ru_total
        if a + b == 0:
            continue
        p = a / (a + b)
        n = kk_freq[w] + ru_freq[w]
        if n < 2:
            continue
        if 0.25 <= p <= 0.75 and n >= 8:
            # Used freely on both sides - a borrowing, not a switch.
            seeds[w] = "shared"
        elif p >= 0.90:
            seeds[w] = "kk"
        elif p <= 0.10:
            seeds[w] = "ru"

    # The letter rule overrides the ratio: a Kazakh-specific character is
    # decisive evidence regardless of how the word distributes.
    for w in freq:
        if set(w) & KK_LETTERS:
            seeds[w] = "kk"
    # The curated borrowing list overrides everything, including the letter
    # rule - a loan written with a Kazakh letter is still a loan.
    for w in SHARED_SEED:
        if w in freq:
            seeds[w] = "shared"

    kk_lm, ru_lm = CharLM(), CharLM()
    n_kk = n_ru = 0
    for w, lang in seeds.items():
        if lang == "kk":
            kk_lm.add(w, min(freq[w], 50))
            n_kk += 1
        elif lang == "ru":
            ru_lm.add(w, min(freq[w], 50))
            n_ru += 1
    if verbose:
        print("seeds: {} kk / {} ru / {} shared word types".format(
            n_kk, n_ru, sum(1 for v in seeds.values() if v == "shared")))
    if n_kk < 200 or n_ru < 200:
        raise SystemExit("Not enough seed words to fit the models.")

    lex = dict(seeds)
    # Words the document-frequency evidence settled directly. Everything else
    # is a guess by the character models, and guesses are what the
    # stripped-form index is allowed to overrule.
    confident = set(seeds)
    unresolved = 0
    for w, c in freq.items():
        if c < MIN_FREQ or w in lex or w in STOP_BOTH:
            continue
        d = kk_lm.logp(w) - ru_lm.logp(w)
        lex[w] = "kk" if d > LLR_MARGIN else ("ru" if d < -LLR_MARGIN else "ambiguous")
        unresolved += 1

    # Stripped-form index, built only from words carrying a Kazakh-specific
    # letter - those are certain, so the index inherits their certainty.
    deacc_kk = {}
    for w, lang in lex.items():
        if lang == "kk" and set(w) & KK_LETTERS:
            deacc_kk.setdefault(deaccent(w), w)
    # A stripped form that is itself a confident Russian word is a collision,
    # not evidence.
    collisions = {k for k in deacc_kk if lex.get(k) == "ru"}
    for k in collisions:
        deacc_kk.pop(k, None)

    if verbose:
        c = Counter(lex.values())
        print("lexicon: {} types ({} resolved by n-gram model)".format(
            len(lex), unresolved))
        print("stripped-form index: {} entries ({} dropped as ru collisions)".format(
            len(deacc_kk), len(collisions)))
        print("  kk {}  ru {}  shared {}  ambiguous {}".format(
            c["kk"], c["ru"], c["shared"], c["ambiguous"]))
    return lex, kk_lm, ru_lm, deacc_kk, collisions, confident


class Density:
    def __init__(self, lex, kk_lm, ru_lm, deacc_kk=None, collisions=None,
                 confident=None):
        self.lex, self.kk_lm, self.ru_lm = lex, kk_lm, ru_lm
        self.deacc_kk = deacc_kk or {}
        self.collisions = set(collisions or ())
        self.confident = set(confident or ())

    def word_lang(self, w):
        known = self.lex.get(w)
        # The stripped-form index only gets a say where the lexicon has no
        # confident answer. Letting it override was a real bug: Russian `и`
        # strips to Kazakh `і` and `а` to `ә`, so 15,018 occurrences of `и`
        # were counted as Kazakh and 8,934 monolingual Russian comments came
        # out as code-switched. Frequent function words and proper names must
        # keep whatever the corpus-level evidence says about them.
        # Evidence beats a guess. `и` and `молдир` were settled by how they
        # distribute across monolingual comments, so the stripped-form index
        # does not get to relabel them; `уры` was only ever a character-model
        # guess, so it does.
        if w in self.confident:
            return known
        # Short words carry too little signal for a stripped-form match to mean
        # anything - almost every 2-3 letter Russian word has a Kazakh
        # near-homograph once the diacritics are gone.
        if len(w) >= 4 and w not in RU_FUNCTION and not (set(w) & KK_LETTERS):
            d = deaccent(w)
            if d in self.collisions:
                return "ambiguous"
            if d in self.deacc_kk:
                return "kk"
        if known is not None:
            return known
        if w in STOP_BOTH:
            return "ambiguous"
        s = seed_of(w)
        if s:
            return s
        # Out-of-vocabulary: fall back to the character models. This is the
        # step that makes the metric work on misspellings and rare morphology.
        d = self.kk_lm.logp(w) - self.ru_lm.logp(w)
        return "kk" if d > LLR_MARGIN else ("ru" if d < -LLR_MARGIN else "ambiguous")

    def measure(self, text):
        """-> dict with counts, minority_share, density bucket, base_lang.

        Two passes, because annotation/guideline.md does not drop borrowings -
        it counts them as *the base language of the surrounding clause*:

          "Керемет видео, рахмет!"  ->  видео is shared  ->  0/3  ->  mono (kk)

        The denominator keeps видео; it just never becomes a switch. So the base
        language has to be settled from the unambiguous words first, and the
        shared ones are then folded into it. Dropping them instead would shrink
        the denominator and inflate every minority share.
        """
        seq = []
        for w in words(text):
            lang = self.word_lang(w)
            if lang in ("kk", "ru", "shared"):
                seq.append(lang)

        kk = sum(1 for x in seq if x == "kk")
        ru = sum(1 for x in seq if x == "ru")
        shared = sum(1 for x in seq if x == "shared")

        if kk == 0 and ru == 0:
            # Nothing but borrowings and ambiguous tokens - no evidence either way.
            return {"kk": kk, "ru": ru, "shared": shared, "counted": 0,
                    "minority_share": 0.0, "density": "", "base_lang": ""}

        if kk > ru:
            base = "kk"
        elif ru > kk:
            base = "ru"
        else:
            # Tie: the guideline breaks it on the language opening the comment.
            base = next((x for x in seq if x in ("kk", "ru")), "kk")

        if base == "kk":
            kk += shared
        else:
            ru += shared

        total = kk + ru
        share = min(kk, ru) / total
        if kk == 0 or ru == 0:
            bucket = "mono"
        elif share <= 0.20:
            bucket = "low"
        elif share <= 0.40:
            bucket = "med"
        else:
            bucket = "high"
        return {"kk": kk, "ru": ru, "shared": shared, "counted": total,
                "minority_share": round(share, 4), "density": bucket,
                "base_lang": base}

    def save(self, path=LEXICON):
        path.parent.mkdir(exist_ok=True)
        path.write_text(json.dumps({
            "lexicon": self.lex, "kk_lm": self.kk_lm.to_dict(),
            "ru_lm": self.ru_lm.to_dict(),
            "deacc_kk": self.deacc_kk, "collisions": sorted(self.collisions),
            "confident": sorted(self.confident),
            "params": {"ngram": NGRAM, "smooth": SMOOTH, "min_freq": MIN_FREQ,
                       "llr_margin": LLR_MARGIN, "shared_margin": SHARED_MARGIN},
        }, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, path=LEXICON):
        if not path.exists():
            raise SystemExit(
                "{} not found. Run: python -m src.preprocess.density "
                "--build".format(path))
        d = json.loads(path.read_text(encoding="utf-8"))
        return cls(d["lexicon"], CharLM.from_dict(d["kk_lm"]),
                   CharLM.from_dict(d["ru_lm"]),
                   d.get("deacc_kk", {}), d.get("collisions", []),
                   d.get("confident", []))


def corpus_texts():
    if not CLEAN.exists():
        raise SystemExit("{} not found. Run build_corpus first.".format(CLEAN))
    with open(CLEAN, encoding="utf-8") as f:
        return [json.loads(line)["text"] for line in f if line.strip()]


def main():
    utf8_stdout()
    p = argparse.ArgumentParser()
    p.add_argument("--build", action="store_true", help="refit the lexicon")
    p.add_argument("--show", type=int, default=0,
                   help="print N measured example comments")
    p.add_argument("--eval", help="CSV with a human `density` column to score against")
    p.add_argument("--no-write", action="store_true",
                   help="do not touch the batch CSVs")
    args = p.parse_args()

    if args.build or not LEXICON.exists():
        texts = corpus_texts()
        print("fitting on {} comments ...".format(len(texts)))
        lex, kk_lm, ru_lm, deacc_kk, collisions, confident = build_lexicon(texts)
        dens = Density(lex, kk_lm, ru_lm, deacc_kk, collisions, confident)
        dens.save()
        print("-> {}".format(LEXICON))
    else:
        dens = Density.load()

    for ex in ("Керемет видео, рахмет!", "Очень интересный выпуск, рахмет сізге",
               "Ол айтты что это не так", "Согласен, дұрыс айтасың",
               "Мен келісемін толығымен"):
        m = dens.measure(ex)
        print("  {:<6} {:<4} kk{:>3} ru{:>3}  share {:.2f}  | {}".format(
            m["density"], m["base_lang"], m["kk"], m["ru"],
            m["minority_share"], ex))

    if args.show:
        print("\n=== measured examples from the corpus ===")
        shown = Counter()
        for t in corpus_texts():
            m = dens.measure(t)
            b = m["density"]
            if b in ("low", "med", "high") and shown[b] < args.show // 3 + 1:
                shown[b] += 1
                print("[{:<4} {:.2f}] {}".format(b, m["minority_share"], t[:110]))

    if args.eval:
        rows = list(csv.DictReader(open(args.eval, encoding="utf-8-sig")))
        pairs = [(r["density"].strip().lower(), dens.measure(r["text"])["density"])
                 for r in rows if (r.get("density") or "").strip()
                 and (r.get("text") or "").strip()]
        if not pairs:
            print("\n{} has no human density labels yet.".format(args.eval))
        else:
            agree = sum(a == b for a, b in pairs)
            mixed = {"low", "med", "high"}
            tp = sum(1 for a, b in pairs if a in mixed and b in mixed)
            fn = sum(1 for a, b in pairs if a in mixed and b not in mixed)
            fp = sum(1 for a, b in pairs if a not in mixed and b in mixed)
            print("\n=== vs human labels ({} comments) ===".format(len(pairs)))
            print("exact bucket agreement: {:.1f}%".format(agree / len(pairs) * 100))
            print("mixed/not-mixed  precision {:.2f}  recall {:.2f}".format(
                tp / (tp + fp) if tp + fp else 0, tp / (tp + fn) if tp + fn else 0))
        return

    if args.no_write:
        return

    files = [f for f in sorted(glob.glob(str(BATCHES / "*.csv")))
             if "manifest" not in Path(f).name]
    written = 0
    bucket_counts = Counter()
    for bf in files:
        with open(bf, encoding="utf-8-sig", newline="") as f:
            rows = list(csv.DictReader(f))
        for r in rows:
            if (r.get("sentiment") or "").strip() == "skip":
                continue
            m = dens.measure(r.get("text", ""))
            if not m["density"]:
                continue
            r["density"] = m["density"]
            if not (r.get("base_lang") or "").strip():
                r["base_lang"] = m["base_lang"]
            bucket_counts[m["density"]] += 1
            written += 1
        tmp = Path(str(bf) + ".tmp")
        with open(tmp, "w", encoding="utf-8-sig", newline="") as f:
            w = csv.DictWriter(f, fieldnames=[
                "comment_id", "text", "sentiment", "density", "base_lang", "notes"],
                extrasaction="ignore")
            w.writeheader()
            w.writerows(rows)
        tmp.replace(bf)

    print("\nwrote density for {} rows across {} batches".format(written, len(files)))
    for b in ("mono", "low", "med", "high"):
        print("  {:<6} {:>5}  {:>5.1f}%".format(
            b, bucket_counts[b], bucket_counts[b] / max(written, 1) * 100))


if __name__ == "__main__":
    main()
