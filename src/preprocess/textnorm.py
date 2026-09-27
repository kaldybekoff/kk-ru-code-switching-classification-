"""Shared text normalization, anonymization and filtering rules for the corpus.

Sprint 2. `make_pilot.py` deliberately keeps its own (older, looser) copies of
these rules so the Sprint 1 pilot batch stays byte-reproducible — do not
refactor it onto this module.

Every rule here was chosen after profiling the 7,832 raw Sprint 1 comments, not
from a generic spam-filter template. Notes on what the profiling showed are
inline, because several of the obvious rules turn out to be wrong on this data.
"""

import hashlib
import re
import sys

URL_RE = re.compile(r"https?://\S+|www\.\S+")
MENTION_RE = re.compile(r"@[\wЀ-ӿ.\-]+")
WS_RE = re.compile(r"\s+")
CYRILLIC_RE = re.compile(r"[Ѐ-ӿ]")
LATIN_RE = re.compile(r"[a-zA-Z]")
# A "meaningful token" per annotation/guideline.md axis 2: a word, not an emoji,
# a number, a punctuation run, a URL placeholder or a @mention placeholder.
WORD_RE = re.compile(r"[Ѐ-ӿa-zA-Z]+")
# Leading "12:45" / "1:29:47" timestamp references.
TIMESTAMP_RE = re.compile(r"\b\d{1,2}:\d{2}(?::\d{2})?\b")
REPEAT_RE = re.compile(r"(.)\1{3,}")

# Promotional keywords. NOTE: `подпис*` is NOT on this list. Profiling found 36
# occurrences and almost all were genuine praise ("Подписалась на Александра",
# "Лайк, подписка") — filtering on it would delete real positive comments.
# A keyword here only marks spam when a URL is present as well.
PROMO_RE = re.compile(
    r"билет|концерт|sollt|промокод|скидк|розыгрыш|giveaway|telegram|whatsapp|"
    r"tiktok|instagram|инстаграм|телеграм|курс|обучени|заработ",
    re.I,
)

# Density is a ratio, so it is only defined once there are enough tokens to form
# one. Below 5 meaningful tokens the `low` band (minority_share <= 0.20) is
# arithmetically unreachable — 1/4 = 0.25 already lands in `med`. Profiling
# confirmed the artifact: 0.0% of screened-`low` comments have <= 5 words, while
# 9.4% of `mono` and 15.2% of `high` do. Comments below this threshold are kept
# for the sentiment corpus but excluded from the density-stratified analysis.
MIN_DENSITY_TOKENS = 5
# Below this a comment carries almost no signal at all ("❤", "1", "😂").
MIN_CORPUS_TOKENS = 3
MAX_CHARS = 1500


def utf8_stdout():
    """Windows consoles default to a legacy codepage and mangle Cyrillic."""
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


def anonymize(text):
    """Strip direct identifiers. Author names are dropped by the caller, never
    carried into any interim file."""
    text = URL_RE.sub("<URL>", text)
    text = MENTION_RE.sub("<USER>", text)
    return WS_RE.sub(" ", text).strip()


def collapse_repeats(text):
    """'❤❤❤❤❤❤' -> '❤❤❤'. Keeps the emphasis as a signal without letting a
    single comment's character runs dominate a TF-IDF vocabulary."""
    return REPEAT_RE.sub(lambda m: m.group(1) * 3, text)


def n_tokens(text):
    return len(WORD_RE.findall(text))


def norm_key(text):
    """Exact-duplicate key: lowercased, punctuation and emoji stripped."""
    k = re.sub(r"[^\w\s]", "", text.lower())
    return hashlib.md5(WS_RE.sub(" ", k).strip().encode()).hexdigest()


def token_key(text):
    """Reordering-tolerant duplicate key: the sorted multiset of word tokens.
    Catches 'Керемет видео' vs 'Видео керемет!!!', which norm_key misses."""
    toks = sorted(w.lower() for w in WORD_RE.findall(text))
    if len(toks) < 3:  # too short for this to be evidence of duplication
        return None
    return hashlib.md5(" ".join(toks).encode()).hexdigest()


def script_of(text):
    """'cyrillic' | 'latin' | 'none'.

    Latin-script comments are NOT junk here. Profiling found 434 (5.5%), and a
    real share of them are romanized Kazakh-Russian code-switching:

        "Mne kajetsya Aldik otirik istoryalar aytatin syaqti"
        (ru 'мне кажется' + kk 'өтірік историялар айтатын сияқты')

    That is exactly the phenomenon this project studies, in a different
    orthography. It is routed to its own file rather than silently dropped —
    see docs/sampling_design.md for why it stays out of the main corpus.
    """
    has_cyr = bool(CYRILLIC_RE.search(text))
    has_lat = bool(LATIN_RE.search(text))
    if has_cyr:
        return "cyrillic"
    if has_lat:
        return "latin"
    return "none"


def classify_row(text):
    """-> (status, reason). status: ok | romanized | too_short | drop."""
    if len(text) > MAX_CHARS:
        return "drop", "too_long"

    script = script_of(text)
    if script == "none":
        # Emoji / punctuation / digits only. 4.0% of the raw pull.
        return "drop", "no_letters"

    stripped = TIMESTAMP_RE.sub(" ", text)
    if n_tokens(stripped) < MIN_CORPUS_TOKENS and TIMESTAMP_RE.search(text):
        # "9:36 🤣🤣🤣" — a pointer into the video, not an opinion. 1.6% of raw.
        return "drop", "timestamp_only"

    if URL_RE.search(text) or "<URL>" in text:
        if PROMO_RE.search(text):
            return "drop", "promo"

    if n_tokens(text) < MIN_CORPUS_TOKENS:
        return "drop", "too_few_tokens"

    if script == "latin":
        return "romanized", "latin_script"

    if n_tokens(text) < MIN_DENSITY_TOKENS:
        # Kept for the sentiment corpus, excluded from density-stratified analysis.
        return "too_short", "below_density_threshold"

    return "ok", ""
