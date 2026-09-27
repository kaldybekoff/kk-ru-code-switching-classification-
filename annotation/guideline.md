# Annotation Guideline v0.3

Kazakh–Russian code-switched YouTube comments
**Still to be revised after the pilot round (100 comments, Cohen's kappa) — v0.2 only
adds the corpus-scope rules settled during Sprint 2 cleaning. The edge-case tables below
are unchanged and remain untested.**

Every comment gets **two independent labels**: a sentiment label (axis 1) and a
code-switch density label (axis 2). They are annotated at the same time but must
not influence each other — a heavily mixed comment is not more or less positive
because of the mixing.

---

## Axis 1 — Sentiment

| Label | Code | When to use |
|-------|------|-------------|
| Positive | `pos` | Praise, thanks, affection, admiration, enthusiasm, support. |
| Negative | `neg` | Criticism, anger, mockery, insult, disappointment, disgust. |
| Neutral | `neu` | Questions, factual statements, on-topic remarks with no evaluative charge. |

### Rules

1. **Judge the comment's own sentiment**, not the topic's. "Опять про войну" discussing
   a sad topic in a flat tone is `neu`, not `neg`.
2. **Target does not matter.** Negative toward the host, a guest, another commenter, or
   the country all count as `neg`.
3. **Emoji count as evidence** alongside text: "Керемет 🔥🔥🔥" is `pos`, "Ведущий 🤡" is `neg`. Emoji-*only* comments are `skip` (see edge cases).
4. **When genuinely torn between two labels, prefer `neu`** — it is the default.
5. **If the comment is unusable, do not force a label** — mark it `skip` (see below).

### Edge cases

| Case | Example type | Decision |
|------|-------------|----------|
| Sarcasm / irony | "Өте жақсы, рахмет 🙃" praising something the comment clearly mocks | Label the **intended** meaning → `neg`. If the irony is not clearly recoverable, `neu`. |
| Mixed sentiment | "Ведущий керемет, но монтаж ужасный" | Label the **dominant** part. If genuinely balanced → `neu`. |
| Emoji-only | "😂😂😂" | `skip`. Density is undefined with zero word tokens, so these cannot enter the corpus. The pre-filter drops them automatically. |
| Question only | "Бұл қай жерде түсірілген?" | `neu`. |
| Greeting / filler | "Салем всем" | `neu`. |
| Religious formula | "Аллаға шүкір", "Иншалла" | `pos` if supportive of the content, else `neu`. |
| Negative word, positive intent | "Ой как жалко его", sympathy | Sympathy toward a person is `pos` toward them → but if it evaluates nothing, `neu`. Prefer `neu`. |
| Toward another commenter | Insulting a commenter, not the video | `neg`. Target does not matter. |
| Copypasta / self-promo | "Подпишись на мой канал" | `skip` (spam). |

### `skip` — exclude from the corpus

Use `skip` for: pure spam/self-promotion, a bare link, fully unintelligible text,
a single character, text in neither Kazakh nor Russian (e.g. fully English or Turkish),
or pure timestamp comments ("12:45").

**Romanized Kazakh/Russian (v0.2).** Comments written in Latin script — *"Mne kajetsya
Aldik otirik istoryalar aytatin syaqti"* — are **out of scope for this corpus** and are
removed by the pipeline before you see them, so they should not appear in your batches.
If one does slip through, mark it `skip` and note `romanized`.

This is not a judgement that they are uninteresting; they are a different regime. The
project's premise is that Kazakh and Russian share the Cyrillic alphabet so the script
carries no signal, whereas in transliteration the spelling conventions themselves do.
See `docs/sampling_design.md` §7.

---

## Axis 2 — Code-switch density

Measured over **meaningful word tokens**. Do not count: emoji, punctuation, numbers,
@mentions, URLs, or names of people/places.

Let `minority_share` = (tokens of the less frequent of the two languages) / (all
meaningful tokens).

**Minimum length (v0.2).** Density is a ratio, so it needs enough tokens to form one.
With 4 meaningful tokens a single switch is already 0.25 — `low` (≤ 0.20) is
arithmetically unreachable. Comments with **fewer than 5 meaningful tokens** are still
annotated normally, but the pipeline flags them `density_eligible = False` and leaves
them out of the density-stratified analysis. Do not try to compensate by rounding them
toward `mono`; label what you see.

| Label | Code | Rule |
|-------|------|------|
| None / monolingual | `mono` | `minority_share` = 0 — the comment is entirely Kazakh or entirely Russian. |
| Low | `low` | `minority_share` > 0 and ≤ 0.20 — mostly one language, a few inserted words. |
| Medium | `med` | 0.20 < `minority_share` ≤ 0.40 — clear mixing, one language still dominant. |
| High | `high` | `minority_share` > 0.40 — genuinely mixed, no clear dominant language. |

Also record `base_lang` = `kk` / `ru` (whichever language supplies the majority of
tokens; for `high`, whichever supplies the first clause).

### Token-level language rules

Kazakh and Russian share the Cyrillic alphabet, so the script gives no signal. Decide
by the **word**, not the letters:

1. **Kazakh-specific letters (ә ғ қ ң ө ұ ү һ і) → Kazakh.** But their absence proves
   nothing — many people type Kazakh without them ("керемет" is Kazakh either way).
2. **Shared/borrowed words** that exist in both languages with the same form
   (интернет, видео, телефон, компьютер) → count as the **base language** of the
   surrounding clause. Do not count them as switches.
3. **Russian words with Kazakh morphology** (e.g. a Russian stem + Kazakh suffix) →
   count as **one Kazakh token** (this is morphological integration, not a switch).
4. **English insertions** (respect, cringe, ok, thanks) → do **not** count toward either
   language, and do not count in the denominator. Note them in the `notes` field if
   frequent; English is out of scope for the density metric.
5. **Proper names** are not counted at all.

### Examples

| Comment | Counting | minority_share | Label |
|---------|----------|---------------|-------|
| "Керемет видео, рахмет!" | керемет kk · видео borrowed → base language · рахмет kk | 0 / 3 | `mono` (kk) |
| "Очень интересный выпуск, рахмет сізге" | выпуск borrowed → ru; 3 ru, 2 kk | 2 / 5 = 0.40 | `med` (ru) |
| "Ол айтты что это не так" | `не` is a kk/ru homograph → counts for neither; 3 ru, 2 kk | 2 / 5 = 0.40 | `med` (ru) |
| "Согласен, дұрыс айтасың" | 1 ru, 2 kk | 1 / 3 = 0.33 | `med` (kk) |
| "Мен келісемін толығымен" | `мен` is a homograph → counts for neither; 2 kk | 0 / 2 = 0 | `mono` (kk) |

> **v0.3 correction.** Three of these five examples were wrong in v0.1–v0.2, and the
> errors have been recomputed above by checking every token with the deterministic
> implementation in `src/preprocess/density.py`:
>
> - *"Очень интересный выпуск, рахмет сізге"* was given a denominator of 6 for 5 counted
>   tokens. The label `med` (ru) was right; the arithmetic was not.
> - *"Ол айтты что это не так"* was given as `high` (kk). With `не` excluded as a
>   homograph the count is 2 kk against 3 ru, which is `med` (ru).
> - *"Согласен, дұрыс айтасың"* was given as `high` (ru) on a count of 4 tokens. It has 3,
>   Kazakh supplies 2 of them, and it is `med` (kk).
>
> This matters more than a typo would: worked examples are what an annotator calibrates
> against, so an error here propagates into every label.
>
> Reproduce the table at any time with `python -m src.preprocess.density`, which prints
> these five examples with their token counts before doing anything else.

## Annotation record format

The sheets produced by `make_pilot.py` already have these columns:

```
comment_id,sentiment,density,base_lang,notes
```

- `sentiment`: `pos` | `neg` | `neu` | `skip`
- `density`: `mono` | `low` | `med` | `high` (leave blank if `skip`)
- `base_lang`: `kk` | `ru` (leave blank if `skip`)
- `notes`: free text — **always fill this in when you hesitated**. These comments are
  the input to the next guideline revision.

## Procedure (solo — intra-annotator agreement)

With one annotator, inter-annotator kappa is not available. The substitute is
**test-retest reliability**: annotate the same batch twice, far enough apart that you are
re-applying the guideline rather than recalling your earlier answers. If the guideline is
vague, you will disagree with yourself — which is exactly what the pilot is meant to find.

1. Annotate `pilot_100_pass1.csv` today.
2. **Wait at least one day.** Then annotate `pilot_100_pass2.csv` — the same comments in
   a different order — without looking at pass 1.
3. Compute kappa per axis: `python -m src.eval.kappa <pass1> <pass2>`.
4. Target: **kappa ≥ 0.60** on sentiment, **≥ 0.70** on density (density is more
   mechanical, so it should agree better). Intra-annotator agreement is normally *higher*
   than inter-annotator, so treat these as a floor, not a pass mark.
5. Below target → review every disagreement, add the pattern to the edge-case table
   above, bump the version, re-run on a **fresh** batch.
6. Log each round in `annotation/pilot_log.md`.

State plainly in the report that this is intra-annotator agreement, not inter-annotator.
It is a legitimate reliability measure, but it does not show that a *second* person would
read the guideline the same way — say so rather than letting a reader assume otherwise.

## Changelog

- **v0.3** — Corrected three arithmetically wrong worked examples in the density
  section (see the note under the examples table). Density is now produced by a
  deterministic metric (`src/preprocess/density.py`) rather than annotated by
  hand; this scale remains the specification that metric implements and the
  reference a human audit is scored against.
- **v0.2** — Sprint 2 corpus-scope rules, decided from profiling the 7,832 Sprint 1
  comments rather than from the pilot: a 5-token minimum for density eligibility, and
  romanized (Latin-script) comments placed out of scope. No edge case was changed —
  those still await the pilot.
- **v0.1** — Sprint 1 initial draft. Not yet pilot-tested.
