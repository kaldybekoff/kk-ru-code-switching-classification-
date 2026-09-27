"""Detect neighbouring languages that are not Kazakh or Russian.

    python -m src.preprocess.langfilter --scan        # measure the corpus
    python -m src.preprocess.langfilter --show 20     # inspect what it flags

Annotating the first 160 sampled comments turned up a contamination rate of
**10.6%** — 11 Kyrgyz, 2 Uzbek, 2 Ukrainian out of 160. The keyword screen used
before that estimated 1.1%, six times too low.

This matters well beyond the wasted annotation budget:

- `density.py` has no concept of Kyrgyz, so it scores Kyrgyz words against the
  Kazakh and Russian character models and files them as Kazakh.
- Kyrgyz-Russian code-switching therefore enters the corpus as Kazakh-Russian
  code-switching. That is not noise around the phenomenon under study, it is a
  different phenomenon being counted as it.

## How the three are told apart

**Kyrgyz.** Shares ө ү ң with Kazakh but has **none** of ә ғ қ һ і, so a single
Kazakh-specific letter is strong evidence against it. Positive evidence comes
from the present tense: Kyrgyz `-Vт` (болот, келет, көрөт, айтат, турат) against
Kazakh `-ады/-еді` (болады, келеді). Long vowels (оо, өө, уу, үү) are frequent
in Kyrgyz and rare in Kazakh, and the genitive is -нын/-нүн against Kazakh
-ның/-нің.

**Uzbek in Cyrillic.** Marked by ў and қ+ғ together with Uzbek grammar, and by
a small set of very frequent forms (ҳамма, гунох, Аллохим, -миз, -ми).

**Ukrainian.** Trivial: і, ї, є, ґ plus Ukrainian function words. Kazakh also
uses і, so the other three letters and the function words carry the decision.

Scores are additive and the threshold is deliberately conservative: a comment
is only flagged when the evidence is positive *and* Kazakh-specific letters are
absent or nearly so. Borderline cases stay in the corpus and are caught by the
annotator's `skip`, which is the safer direction to err in.
"""

import argparse
import json
import re
from collections import Counter
from pathlib import Path

from src.preprocess.textnorm import utf8_stdout

ROOT = Path(__file__).resolve().parents[2]
CLEAN = ROOT / "data" / "interim" / "corpus_clean.jsonl"

WORD_RE = re.compile(r"[Ѐ-ӿ]+")

# Present in Kazakh, absent from Kyrgyz. Each occurrence is evidence against Kyrgyz.
KK_ONLY = set("әғқһ")
# Ukrainian-specific; Kazakh has і but not ї/є/ґ.
UK_ONLY = set("їєґ")
UZ_ONLY = set("ў")

# Kyrgyz present-tense and function forms whose Kazakh equivalents differ.
KY_WORDS = {
    "болот", "келет", "кетет", "көрөт", "жүрөт", "айтат", "берет", "алат",
    "дейт", "турат", "жатат", "кылат", "кылган", "кылып", "кылуу", "болуу",
    "эле", "беле", "эч", "жана", "менен", "экөө", "силер", "эмне", "ошол",
    "ушул", "тигил", "бирөө", "бирөөсү", "окшойт", "ойлобойсузбу", "болчу",
    "бактылуу", "болгула", "жакшылык", "кыргыз", "кыргызмын", "кыргызстан",
    "сүйлөбөдү", "жебегиле", "белениз", "жөнөкөй", "бөбөктү", "карытбайт",
    "гүлдөй", "аласыңбы", "кылууда", "кетбейт", "жолдошунуздун", "коромин",
    "бүтүпсүнөр", "бутупсунор", "эчтеке", "эчнерсе", "албайм", "тилинерди",
    "укугун", "тепселебеген", "карытбайт", "гулдой", "озунузду", "ойлобойсуз",
}
# Removed after validation: `тилейм` collided with Kazakh тілеймiз/тилеймиз
# ("we wish"), and `жазып`, `атын`, `бизге`, `жаштар`, `кыла` are ordinary
# Kazakh words. Each cost real Kazakh comments to a false positive, which is
# the expensive direction of error here - a missed Kyrgyz comment is caught by
# the annotator, a deleted Kazakh one is gone.
# Uzbek Cyrillic markers.
UZ_WORDS = {
    "ҳамма", "хаммамизга", "гунох", "гунохларимизни", "аллохим", "астагфируллох",
    "аллох", "аллохга", "аллохим", "козок", "боурлар",
    "мусулмон", "мусилмонмисан", "исломни", "исломми", "зертла", "жавоби",
    "чиройли", "утказип", "домла", "тогамни", "дининг", "гунохлар",
}
# `сабр`, `берсин`, `керак`, `яхши`, `бор` were dropped: all are ordinary
# Kazakh or shared Turkic forms.
UK_WORDS = {
    "дякую", "щиро", "зустріч", "приємна", "пізнавальна", "маєш",
    "можливості", "знайти", "необхідне", "оточення", "дійти", "найприємніше",
    "найкорисніше", "використання", "висновок", "розуму", "щастя", "немає",
    "нещастя", "захваті", "казахського", "передивилася", "знайшла",
    "виступи", "черзі", "чарівні", "розвитку", "вашому", "імпонує",
    "кількість", "нецензурної", "лексики",
}
# Ukrainian is the easy case - ї/є/ґ alone decide most of it. The word list
# drops anything a Russian comment could contain (часу, горе, миру, але, цей).

LONG_VOWEL_RE = re.compile(r"(оо|өө|уу|үү)")
KY_GENITIVE_RE = re.compile(r"(нын|нин|нун|нүн)$")


def score(text):
    """-> dict of per-language scores plus the Kazakh-evidence count."""
    ws = [w.lower() for w in WORD_RE.findall(text)]
    if not ws:
        return {"ky": 0, "uz": 0, "uk": 0, "kk_evidence": 0, "n": 0}

    kk_evidence = sum(1 for w in ws if set(w) & KK_ONLY)

    def marked(w, lexicon):
        """Exact match, or the word begins with a marker. Kyrgyz is
        agglutinative like Kazakh, so `силерге` and `силерди` are the same
        evidence as `силер` and a plain set lookup misses them."""
        if w in lexicon:
            return True
        return any(w.startswith(m) and len(m) >= 4 and len(w) - len(m) <= 4
                   for m in lexicon)

    ky = 2 * sum(1 for w in ws if marked(w, KY_WORDS))
    ky += sum(1 for w in ws if LONG_VOWEL_RE.search(w))
    ky += sum(1 for w in ws if KY_GENITIVE_RE.search(w))

    uz = 2 * sum(1 for w in ws if marked(w, UZ_WORDS))
    uz += 3 * sum(1 for w in ws if set(w) & UZ_ONLY)

    uk = 2 * sum(1 for w in ws if marked(w, UK_WORDS))
    uk += 3 * sum(1 for w in ws if set(w) & UK_ONLY)

    return {"ky": ky, "uz": uz, "uk": uk, "kk_evidence": kk_evidence, "n": len(ws)}


def detect(text):
    """-> 'ky' | 'uz' | 'uk' | None.

    Ukrainian and Uzbek are decided on their own letters, which Kazakh does not
    use, so Kazakh evidence does not override them. Kyrgyz shares its entire
    alphabet with Kazakh, so it is only called when Kazakh-specific letters are
    absent or vanishingly rare relative to the Kyrgyz evidence.
    """
    s = score(text)
    if s["n"] < 3:
        return None
    if s["uk"] >= 3:
        return "uk"
    if s["uz"] >= 3 and s["kk_evidence"] <= 1:
        return "uz"
    if s["ky"] >= 2 and s["kk_evidence"] == 0:
        return "ky"
    if s["ky"] >= 6 and s["kk_evidence"] <= 1:
        return "ky"
    return None


def main():
    utf8_stdout()
    p = argparse.ArgumentParser()
    p.add_argument("--scan", action="store_true", help="measure the whole corpus")
    p.add_argument("--show", type=int, default=0, help="print N flagged examples")
    p.add_argument("--infile", default=str(CLEAN))
    args = p.parse_args()

    rows = [json.loads(l) for l in open(args.infile, encoding="utf-8") if l.strip()]
    flagged = [(detect(r["text"]), r) for r in rows]
    hits = [(lang, r) for lang, r in flagged if lang]

    c = Counter(lang for lang, _ in hits)
    print("{} comments scanned".format(len(rows)))
    for lang in ("ky", "uz", "uk"):
        print("  {}  {:>5}  {:>5.2f}%".format(lang, c[lang], c[lang] / len(rows) * 100))
    print("  ---------------------")
    print("  all {:>5}  {:>5.2f}%".format(len(hits), len(hits) / len(rows) * 100))

    if args.show:
        print("\n=== flagged examples ===")
        shown = Counter()
        for lang, r in hits:
            if shown[lang] >= args.show // 3 + 1:
                continue
            shown[lang] += 1
            print("[{}] {}".format(lang, r["text"][:110]))


if __name__ == "__main__":
    main()
