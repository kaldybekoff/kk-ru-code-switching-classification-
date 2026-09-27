# Kazakh–Russian Code-Switched Text Classification

**Research report · Academic Beta Career 2026 · Yeskendir Kaldybek**

Everything from Sprints 1 and 2 in one place: the literature review and the research gap,
how the sources were chosen, how the corpus was built, and what building it turned up.
Video production material lives in `reports/video/`; it is scaffolding, not research.

| | |
|---|---|
| Research question and hypotheses | [docs/research_framing.md](../docs/research_framing.md) |
| Annotation guideline | [annotation/guideline.md](../annotation/guideline.md) |
| Sampling design | [docs/sampling_design.md](../docs/sampling_design.md) |
| Sprint plan and status | [PROJECT_PLAN.md](../PROJECT_PLAN.md) |

---

## Where the project stands

| Sprint | Weeks | Status |
|---|---|---|
| 1 — Literature review and research framing | 1–2 | Complete |
| 2 — Dataset preparation and annotation | 3–4 | Corpus complete; annotation in progress |
| 3 — Corpus finalization and baseline models | 5–6 | Pipeline written and tested, waiting on labels |

### The question

> How do TF-IDF, fastText and multilingual transformer representations compare on
> Kazakh–Russian code-switched text classification, and does the gap between them depend
> on how heavily the text is mixed?

**H1** — multilingual transformer representations perform best overall.
**H2** — their advantage shrinks as code-switching density rises.
**H3** — fastText degrades less than TF-IDF on heavily mixed text.

H1 replicates a known result. H2 and H3 are the contribution: no published work on this
pair breaks performance down by how heavily the text is mixed.

---

# Part I — Literature review (Sprint 1)

Weeks 1–2 · Yeskendir Kaldybek

**Scope:** classification methods and text representations for code-switched NLP;
existing Kazakh/Turkic NLP resources; the gap this project fills.

> **Honesty note.** Summaries below are the substance of each work — what it argues and
> why it matters here. Claims tagged **⚠ verify** are specific numbers I have not
> confirmed against the paper. Do **not** put a ⚠ number into the final report or the
> video without opening the source first. Everything untagged is safe to state.

---

### 1. What the field looks like

#### 1.1 Surveys

**Sitaram et al. (2019), *A Survey of Code-switched Speech and Language Processing***

The field map. Code-switching is normal behaviour in multilingual communities, but NLP
systems are built and evaluated on monolingual data, so they degrade on it. The survey's
central diagnosis is that **data scarcity is the bottleneck, not modelling** — CS text is
rarely collected and even more rarely annotated, so most work recycles the same handful
of corpora. Work concentrates on Hindi–English, Spanish–English and Mandarin–English.
Tasks covered: language identification (LID), POS tagging, NER, language modelling, ASR,
MT, and sentiment.

*Takeaway for us:* the framing "no data exists for this pair, so we build it" is the
standard and accepted contribution shape in this field. We do not need to invent a new
model to make a contribution.

**Doğruöz, Sitaram, Bullock, Toribio (ACL 2021), *A Survey of Code-switching: Linguistic
and Social Perspectives for Language Technologies***

The linguistics-side critique of the NLP literature. Two points matter to us directly:

1. **Borrowing ≠ code-switching.** An established loanword that the recipient language
   has absorbed (Kazakh *интернет*, *видео*, *компьютер*) is not a switch — the speaker
   is not alternating languages, they are using a word their language already owns. A
   productive switch is different. Many NLP papers conflate the two and therefore
   overestimate how much switching their corpus contains.
2. Code-switching is **socially structured**, not random noise — it patterns by speaker,
   topic, and register.

*Takeaway for us:* point 1 is the direct justification for **rule #2 of our density
metric** (shared/borrowed words count as the base language, not as a switch). This is a
methodological choice we can now defend with a citation instead of arguing it from
intuition. Point 2 justifies sampling from several channel types rather than one.

#### 1.2 Benchmarks — where our task definition comes from

**SemEval-2020 Task 9, "SentiMix" (Patwa et al.)** — *the closest precedent to this
project.*

Sentiment classification on code-mixed tweets, two tracks: Hindi–English (Hinglish) and
Spanish–English (Spanglish). Labels: **positive / negative / neutral** — exactly the
three we chose. Organizers supplied word-level language tags alongside the text.
**Confirmed:** 20K Hinglish and 19K Spanglish examples, labels positive/negative/neutral,
89 submissions; best F1 **75.0% Hinglish / 80.6% Spanglish**. BERT-like models and
ensembles dominated. ([arXiv 2008.04277](https://arxiv.org/abs/2008.04277))

*Takeaway for us:* this legitimizes our entire setup. 3-class sentiment on code-mixed
social-media text is an established task formulation with a shared task behind it, so we
are not inventing a task — we are porting a recognized one to an unstudied language pair.
Cite this as the template in the report.

**Khanuja et al. (ACL 2020), GLUECoS**

A multi-task CS benchmark (LID, POS, NER, sentiment, QA, NLI) for Hindi–English and
Spanish–English. Two findings we use:

- mBERT is a strong general baseline across CS tasks, but
- **continuing pretraining on code-switched text improves it further**, which implies the
  off-the-shelf multilingual model is *not* already well-adapted to mixed input.
- CS task performance sits well below the monolingual equivalents of the same tasks.

*Takeaway for us:* this is indirect empirical support for **H2**. If plain multilingual
pretraining were sufficient for mixed text, CS-specific further pretraining would not
help as much as it does.

**Aguilar, Kar, Solorio (LREC 2020), LinCE**

A centralized CS benchmark and leaderboard covering four pairs — Spanish–English,
Nepali–English, Hindi–English, and Modern Standard Arabic–Egyptian Arabic — across LID,
POS, NER and sentiment.

*Takeaway for us:* (a) evidence for the "only a handful of pairs are covered" claim in
our gap statement — Kazakh–Russian is in none of these benchmarks; (b) a reference for
standard evaluation practice. Note that the Arabic pair is the closest analogue to ours:
two varieties sharing one script, where surface cues are weak.

---

### 2. The three representations we compare

**TF-IDF + linear classifier** — the reference baseline. No pretraining, no cross-lingual
knowledge; each surface word form is an independent feature. Two predicted weaknesses on
our data: Kazakh is agglutinative, so one lemma appears as many distinct word forms and
the feature space fragments; and informal spelling variation (Kazakh typed with or
without ә/қ/ң/ө/ұ/ү/і) splits the same word into several unrelated features. Cheap,
fast, fully interpretable — keep it as the floor everything else must beat.

**fastText — Bojanowski, Grave, Joulin, Mikolov (TACL 2017), *Enriching Word Vectors with
Subword Information***

Extends word2vec by representing a word as the sum of its character n-gram vectors. Two
consequences: out-of-vocabulary words still get a vector (built from their n-grams), and
morphologically related forms end up near each other because they share n-grams. The
paper reports the gains are largest on morphologically rich languages.

*Takeaway for us:* this is the **theoretical basis of H3**. Kazakh's agglutinative
morphology is exactly the case subword n-grams are designed for, and n-grams also degrade
gracefully under spelling variation. Prediction: fastText loses less than TF-IDF as
density and informality rise.

Companion resource: **Grave et al. (LREC 2018), *Learning Word Vectors for 157
Languages*** — pretrained vectors trained on Wikipedia + Common Crawl. Kazakh is expected
to be among the 157 ⚠ verify before relying on it.

**XLM-R — Conneau et al. (ACL 2020), *Unsupervised Cross-lingual Representation Learning
at Scale***

Transformer encoder pretrained with masked language modelling on **CC-100: 100 languages,
~2.5 TB of filtered CommonCrawl**. Kazakh is one of the 100 ⚠ verify exact data size.
Two findings we need:

- Low-resource languages gain substantially from cross-lingual transfer at scale.
- **The "curse of multilinguality":** at fixed model capacity, adding languages helps up
  to a point and then *degrades* per-language performance, because capacity is shared.

*Takeaway for us — this is the core argument for H2.* CC-100 is assembled as **monolingual
web text per language**. The model has seen a lot of Kazakh and a lot of Russian, but
essentially no text where the two alternate inside one sentence. Its cross-lingual
alignment is learned indirectly, not from mixed input. So we should expect strong
performance on monolingual comments and weaker relative performance as mixing increases —
which is precisely the hypothesis, and now it has a mechanism behind it rather than being
a guess.

**mBERT — Devlin et al. (NAACL 2019)** — 104 Wikipedia languages; the encoder most CS
papers benchmark against, hence the natural secondary comparison. Kazakh expected to be
included ⚠ verify.

**Winata et al. (CALCS 2021), *Are Multilingual Models Effective in Code-Switching?* —
the single most relevant paper to our hypothesis.**

Asks our exact question for other language pairs and answers it sceptically: large
multilingual models do **not** uniformly outperform simpler or more targeted approaches on
code-switched input, and their effectiveness varies by language pair. Multilingual
pretraining alone does not produce good representations of mixed text.

*Takeaway for us:* H2 is not a fringe guess — there is published evidence pointing the
same way for other pairs. Our contribution is to test it on a pair that shares a script,
and to test it **as a function of density** rather than as a single aggregate number,
which is the finer-grained version of their question.

---

### 3. Kazakh / Turkic resources — what exists and what doesn't

| Resource | What it is | Code-switched? |
|---|---|---|
| **KazNERD** (Yeshpanov, Khassanov, Varol, LREC 2022) | Kazakh NER dataset, ~112k sentences ⚠ verify, 25 entity classes, openly released | **No** — monolingual Kazakh |
| **KazParC** (Yeshpanov et al., 2024) | Kazakh–English–Russian–Turkish parallel corpus for MT, plus the Tilmash MT model | **No** — parallel, i.e. clean monolingual on each side |
| **Mirzakhalov et al. (EMNLP 2021)** | Large-scale MT study across 22 Turkic languages + parallel data | **No** |
| Apertium Kazakh pairs, Kazakh morphological analyzers | Rule-based morphology and translation tooling | **No** — but usable for token analysis if we need morphology |

The pattern is consistent: **Kazakh NLP resources exist and are of decent quality, but
every one of them is monolingual or parallel by construction.** A parallel corpus is the
opposite of what we need — it is specifically clean Kazakh on one side and clean Russian
on the other, never the two interleaved in one sentence.

⚠ **Required before the report:** a proper negative-result search. The gap claim rests on
it, so search ACL Anthology, Google Scholar and HuggingFace datasets for
"Kazakh code-switching", "Kazakh-Russian code-mixing", "қазақша орысша араласқан",
and record what you searched and found nothing.

---

### 4. Evaluation methodology

**Dietterich (1998), *Approximate Statistical Tests for Comparing Supervised
Classification Learning Algorithms***

Compares five tests for "is classifier A better than classifier B". **McNemar's test** is
recommended when each algorithm can only be trained once: it has acceptably low Type I
error, unlike a naive t-test over resampled runs, which is over-optimistic because the
samples are not independent.

*Takeaway for us:* justifies McNemar in the protocol. Because we make more than one
pairwise comparison, add **Holm–Bonferroni** correction and report raw and corrected
p-values. Also report the macro-F1 difference with a bootstrap CI — significance alone
does not tell the reader whether the difference is large enough to matter.

---

### 5. Related work on Kazakh–Russian — CORRECTED after the negative-result search

**The earlier draft of this section claimed no labeled Kazakh–Russian code-switched
corpus exists. That claim is false as of 2026 and has been removed.** A proper search
found four directly relevant works, three of them published in 2025–2026:

| Work | What it is | Overlap with this project |
|---|---|---|
| **Yeshpanov (2026)**, *100,000+ Movie Reviews from Kazakhstan: Russian, Kazakh, and Code-Switched Texts* ([arXiv 2605.08600](https://arxiv.org/abs/2605.08600), [ACL](https://aclanthology.org/2026.nlp4dh-1.4/)) | 100,502 kino.kz reviews, manually annotated for **language and sentiment**; 3-way polarity task; benchmarks BoW/TF-IDF against mBERT, XLM-R, RemBERT | **Very high.** Same language pair, same 3-class sentiment task, overlapping representations. Finds transformers consistently beat classical baselines. |
| **Savelyev (2026)**, *Loanword or Switch? The Annotation Boundary, Not the Model, Drives Kazakh–Russian Code-Switching Identification* ([arXiv 2608.00581](https://arxiv.org/abs/2608.00581)) | Gold LID set + a mixed-only sentiment pool, released on GitHub; tests fastText, Lingua, HeLI, char-trigram NB, XLM-R | **High.** Its central finding — the loanword-vs-switch annotation boundary drives results more than model class — is exactly our density rule #2. |
| **Borisov, Kozhirbayev, Malykh (2025)**, *Low-resource MT for Code-switched Kazakh–Russian* ([arXiv 2503.20007](https://arxiv.org/abs/2503.20007)) | First code-switched kk–ru **parallel** corpus, partly synthetic; 16.48 BLEU | Moderate. Different task (MT), but establishes the pair as an active research area. |
| **KazSAnDRA** ([arXiv 2403.19335](https://arxiv.org/abs/2403.19335), [GitHub](https://github.com/IS2AI/KazSAnDRA)) | 180,064 Kazakh reviews, 1–5 star ratings; naturally contains kk–ru switching | Moderate. Kazakh-first, switching incidental rather than annotated. |

#### What is actually still open

Yeshpanov (2026) is the closest work and it **already answers H1** — transformers beat
classical baselines on Kazakhstani code-switched sentiment. Three things it does not do:

1. **No breakdown by code-switching density.** The comparison is aggregate. Whether the
   transformer advantage *holds, shrinks, or reverses* as mixing increases is untested.
   **H2 and H3 survive intact, and they are now the entire contribution.**
2. **No fastText.** The comparison is BoW/TF-IDF vs. transformers, skipping the subword
   representation that H3 is about — and which Savelyev used only for language ID.
3. **Different domain and register.** kino.kz movie reviews are composed, moderately
   formal text. YouTube comments are shorter, more misspelled, and more informal, which
   is where subword robustness should matter most.

#### Revised positioning

> Recent work benchmarks classical and multilingual transformer representations on
> Kazakhstani code-switched sentiment and reports that transformers win overall
> (Yeshpanov 2026). That comparison is aggregate: it does not ask whether the advantage
> survives as the text becomes genuinely mixed. Savelyev (2026) shows, for language
> identification on the same pair, that where the loanword/switch boundary is drawn
> changes conclusions more than the model does — which makes a density-aware evaluation
> of downstream classification the obvious next question. This project answers it:
> TF-IDF, fastText and a multilingual transformer compared **per code-switch density
> bucket**, on informal YouTube comments rather than composed reviews.

This is a narrower claim than the original draft, and a defensible one. Positioning the
project as an extension of published results is stronger than claiming a void that a
reviewer can disprove with one search.

#### Consequence for the project

- H1 is now partly a **replication** in a new domain, not a novel claim. Say so.
- H2 and H3 are the contribution. Protect them: the density annotation has to be good.
- Savelyev's released data is worth examining — possibly as an external validation set.
- Yeshpanov's corpus gives a directly comparable reference point for our numbers.

### 6. Conclusions carried into the rest of the project

1. **The task definition is safe.** 3-class sentiment on code-mixed social text is exactly
   SemEval-2020 Task 9. We inherit a validated formulation.
2. **H2 has a mechanism, not just a hunch.** XLM-R is pretrained on monolingual CC-100
   with no mixed-sentence data; GLUECoS shows CS-specific pretraining still adds gains;
   Winata et al. report multilingual models underperforming on CS. Three independent
   lines point the same way.
3. **H3 has a mechanism too.** fastText's character n-grams are built for morphologically
   rich languages and absorb spelling variation — both of which describe informal Kazakh.
4. **The borrowing/switching distinction must be in the guideline**, per Doğruöz et al.,
   or the density metric will overcount switches. Already encoded as density rule #2.
5. **Shared script is our genuine novelty.** Every major CS benchmark gets surface cues
   from script or transliteration; Kazakh–Russian does not. Say this explicitly — it is
   the strongest "why is this hard" sentence available.
6. **Density stratification is the contribution.** Aggregate numbers would replicate known
   results; the per-density breakdown is what nobody has reported for this pair.

### 7. Fact-checks outstanding (all ⚠ items, in one list)

- [ ] Kazakh in XLM-R's 100 languages, and its share of CC-100
- [ ] Kazakh in mBERT's 104 Wikipedia languages
- [ ] Kazakh among fastText's 157 pretrained language vectors
- [x] SemEval-2020 Task 9: 20K/19K, pos/neg/neutral, best F1 75.0/80.6 — CONFIRMED
- [ ] KazNERD exact size; KazParC exact size
- [x] Negative-result search — DONE, and it came back POSITIVE. See section 5.

---

# Part II — Source selection (Sprint 1)

Deciding **which YouTube sources to annotate**, using measurement rather than
assumption. Tool: `src/preprocess/cs_screen.py` (a lower-bound heuristic screen, not
the annotation density metric).

### What was collected

| Batch | Source type | Comments | Screened mixed |
|---|---|---|---|
| `kids_gaming.jsonl` | Videos from YouTube's most-popular chart for Kazakhstan | 1,832 | **~3%** |
| `adult_v2.jsonl` | Long-form interviews, podcasts, stand-up (search by view count) | 6,000 | **~8%** |

Both figures are lower bounds — the screen only sees words in its two closed word
lists, so content-word switches are invisible to it.

### Finding 1 — YouTube's popular chart for Kazakhstan is the wrong sampling frame

The most-viewed videos in Kazakhstan are dominated by children's gaming channels
(SEGA KZ, Zhukonay, УСАТИК Roblox). Their comment sections are:

- **Almost entirely monolingual Kazakh** (~3% mixed by the screen)
- Written by children — "Мен 7 сынып оқимын", "Менде 2 сыныпын" recur constantly
- Heavily misspelled, which is interesting for robustness but useless for a
  **code-switching** corpus if there is no switching to observe

These channels are marked REJECTED in `data/sources.csv`. The secondary reason to avoid
them stands on its own: a corpus built mostly from comments written by young children is
not something to publish, even anonymized.

### Finding 2 — long-form adult content is where the code-switching is

Interviews, podcasts and stand-up produced roughly **2.5× the mixed rate**, and the
mixing is qualitatively different — genuine intra-sentential switching by adults, not
isolated borrowings:

> Единственное мое пожелание для Молдира — чтобы это трудное время поскорее забылось ❤ бұл қиын уақыт…

> НҮКТЕСІН ҚОЙҒАН ТАҚЫРЫП БОЛДЫ! Моля ты это сделала ✊🏻

> Осы интервьюден кейін жеңілденіп кететін сияқты, ей станет легче, и Мөлдір бәрі жақсы болады

This is exactly the phenomenon the project is about.

### Finding 3 — pull specific videos, not channel feeds

Pulling by channel ID mostly returned Shorts with comments disabled: three priority
channels yielded **81 comments across 18 videos**. Searching for long-form videos and
pulling those returned **6,000 from 30**. Collect by video ID, filtered to long-form.

### Consequence for Sprint 2

1. Collect from the priority-1 interview/podcast sources in `data/sources.csv`.
2. Screen every new batch with `cs_screen` **before** annotating, and drop sources
   below ~5% screened-mixed.
3. **Sampling is now a design decision that must be documented.** A uniform sample is
   ~90% monolingual, which would leave too few medium/high-density examples for the
   subgroup analysis that H2 and H3 depend on. Options:
   - stratified sampling with a documented target density distribution, or
   - uniform sampling and a much larger corpus.
   Decide in Sprint 2 and state it explicitly in the report — a stratified corpus whose
   sampling is undocumented would misrepresent how common code-switching actually is.
4. The **pilot batch is deliberately oversampled** to 50% screened-mixed
   (`--mixed-share 0.5`) so the density rules actually get exercised. This is correct
   for testing a guideline and wrong for the final corpus.

### Honest limitations

- `cs_screen` has recall bounded by its word lists; true mixed share is higher.
- Kazakh typed without ә/ғ/қ/ң/ө/ұ/ү/һ/і is often missed.
- Homographs shared by both languages (да, не, о) are excluded as ambiguous rather than
  guessed, which lowers recall further but avoids false "mixed" hits on pure Kazakh.
- "Popular in Kazakhstan" was one sampling attempt on one day; it is evidence about that
  chart, not a claim about all Kazakhstani YouTube.

---

# Part III — Corpus construction (Sprint 2)

## 1. Collection

Sources were found by script and then checked by hand. `discover_videos.py` searches the
channels in `data/sources.csv` for long-form videos and ranks them by how many comments
they actually carry, which is the number that decides whether a pull is worth the quota.

| | |
|---|---|
| Videos | 185 — 172 found by script, 13 added by hand |
| New comments pulled | 45,832 |
| Total with Sprint 1 | 53,664 |
| API quota used | ~1,560 of 10,000 daily units |

Comments were pulled with `--order time` rather than `--order relevance`. Relevance
ordering front-loads heavily-upvoted comments, which are not representative of what people
write — they are what people upvote, which skews long, quotable and emotionally marked.
Sprint 1 batches keep `relevance` so they stay reproducible, and the two live in separate
raw files so the difference can be checked rather than assumed.

### Code-switch rate per source, re-measured on the full pull

| Source | Screened mixed | med+high |
|---|---|---|
| TARTARIA FILMS | 13.8% | 214 |
| Qazaq Stand Up | 11.8% | 155 |
| kana beisekeyev | 9.3% | 149 |
| Marat Oralgazin | 8.1% | 91 |
| Anna Danchenko | 7.6% | 51 |
| ALIBEKOVKZ | 5.7% | 97 |
| AIRAN | 5.1% | 65 |
| Janar Baisemiz | 4.9% | 62 |
| Stand Up Astana | 4.5% | 63 |
| CMN KZ | 4.4% | 23 |
| Информбюро 31 | 2.7% | 3 |

Sprint 1 proposed dropping sources below ~5%. None were dropped. That rule was for deciding
where to spend collection effort before the data existed; now that it does, the stratified
sampler already concentrates annotation on mixed comments wherever they come from, so
excluding a source would discard usable high-density examples for no gain. The measured
rate is a lower bound in any case, and 4.5% against 5.0% is well inside its error. The one
source actually rejected is `Altyn Bala TV` — kids cartoons with comments disabled, on the
same grounds as the Sprint 1 gaming channels.

## 2. Cleaning

53,664 raw → **42,497** clean, across 196 videos.

| Removed | Count | Why |
|---|---|---|
| too short to carry meaning | 4,766 | fewer than 3 word tokens |
| emoji / punctuation only | 2,207 | density is undefined with zero word tokens |
| another language | 866 | see Part IV |
| exact duplicates | 269 | inflate agreement, leak between CV folds |
| over 1,500 characters | 245 | would dominate a TF-IDF vocabulary |
| timestamp pointers | 199 | "9:36 🤣" is a reference into the video, not an opinion |
| cross-video copypasta | 108 | same text under 3+ videos, bot or campaign text |
| word-reordered duplicates | 43 | "Керемет видео" vs "Видео керемет!!!" |
| promotional spam | 9 | a link **and** a promo keyword together |

Also held aside: **216 romanized comments**, Latin-script Kazakh–Russian mixing such as
*"Mne kajetsya Aldik otirik istoryalar aytatin syaqti"*. That is the same phenomenon in a
different orthography, and a different regime for the research question — the premise here
is that the shared Cyrillic alphabet gives no signal, whereas transliteration conventions
do. Mixing them would confound what is being measured, and there is not enough romanized
data to study separately.

### The spam rule was derived from this corpus, not from a template

Profiling `подпис` (subscribe) expecting spam returned 36 matches in 7,832 comments, and
almost all were genuine praise:

> "**Подписалась** на Александра совсем недавно. И теперь с упоением смотрю его подкасты."
>
> "Ведущий круто задает вопросы. **Подписалась.**"

A keyword filter would have deleted real positive examples, the class the corpus has least
of. The surviving rule marks a comment as spam only when a link and a promotional keyword
appear together, which catches 9 comments.

## 3. Anonymization

Author display names are dropped when the raw file is loaded and never reach any interim
file. URLs become `<URL>`, @mentions become `<USER>`. `data/raw/` and `data/interim/` are
gitignored; only the final labeled dataset is published.

---

# Part IV — What building the corpus turned up (Sprint 2)

## 1. Neighbouring languages, concentrated in the worst possible bucket

Annotating the first 160 sampled comments produced a 10.6% `skip` rate, and 15 of the 17
skips were not Kazakh at all: 11 Kyrgyz, 2 Uzbek, 2 Ukrainian.

The cost is not the wasted annotation. `density.py` has no model for these languages, so it
scores them against the Kazakh and Russian character models and files roughly 70% of them
as code-switched. Kyrgyz–Russian switching was entering the corpus **as** Kazakh–Russian
switching, which is a different phenomenon being counted as the one under study.

Measured concentration on the first draw:

| | Share that was not Kazakh |
|---|---|
| whole corpus | 2.0% |
| `mono` bucket | 0.3% |
| **`high` bucket** | **8.8%** |

The bucket H2 depends on was the most contaminated, by a factor of about 30 over `mono`,
precisely because foreign text looks mixed to an algorithm that only knows two languages.

**The filter.** Kyrgyz shares ө ү ң with Kazakh but has **none** of ә ғ қ һ, so a single
Kazakh-specific letter is strong evidence against it. Positive evidence comes from the
present tense — Kyrgyz `-Vт` (болот, келет, көрөт) where Kazakh has `-ады/-еді` — plus long
vowels and the genitive -нын against -ның. Uzbek is marked by ў and a small set of frequent
forms; Ukrainian by ї, є, ґ.

Validated on the 160 comments already labelled by hand: **recall 100%, precision 94%**.
Markers that collided with Kazakh were removed during validation: `тилейм` matched Kazakh
*тілеймiз*, and `берсин`, `сабр`, `керак` are ordinary Kazakh or shared Turkic forms. A
false positive is the expensive direction here — a missed Kyrgyz comment is caught by the
annotator, a deleted Kazakh one is gone.

Result: 866 comments routed to `corpus_otherlang.jsonl`, contamination in the sample
**5.1% → 0.00%**.

## 2. Diacritic-free Kazakh was being read as Russian

**59% of this corpus contains no Kazakh-specific letter at all.** The lexicon is seeded from
comments that are confidently monolingual, and that test relies on those letters, so Kazakh
typed without them never enters the Kazakh seed and the character models score it as
Russian.

The effect on a single comment: *"…уры назрбаевтын силимтиги…"* — Kazakh ұры (thief) and
сілімтігі, typed bare — was read as 3 Kazakh against 4 Russian tokens and filed as `high`
density with Russian as the base language. It is monolingual Kazakh with one Russian swear
word.

**The fix.** Every word carrying a Kazakh-specific letter is also indexed under its stripped
form (ұры → уры, сілімтігі → силимтиги), and an unmarked word matching that index is Kazakh,
unless the stripped form is itself a real Russian word (боқ → бок), in which case it is
ambiguous and votes for neither.

This needed two corrections before it worked. Applying the index ahead of the lexicon meant
Russian `и` mapped to Kazakh `і` and `а` to `ә`, which made 8,934 monolingual Russian
comments look code-switched. Restricting it to words of four letters or more, excluding
Russian function words, and letting it overrule only what the character models guessed
rather than what document frequency established, brought that down and left the corpus
distribution where it should be.

Effect on the sample: the `high` bucket fell from 215 comments to 136 once measured
correctly. The earlier metric had been inflating it by roughly half.

## 3. The guideline worked examples were wrong

Checking the density procedure with the guideline found three arithmetic errors in the five
worked examples in `annotation/guideline.md`:

| Example | Was | Is |
|---|---|---|
| "Очень интересный выпуск, рахмет сізге" | denominator 6 | 5 counted tokens; label was right |
| "Ол айтты что это не так" | `high` (kk) | `med` (ru), 2 kk against 3 ru |
| "Согласен, дұрыс айтасың" | `high` (ru) on 4 tokens | `med` (kk), 3 tokens, Kazakh supplies 2 |

Worked examples are what an annotator calibrates against, so an error there propagates into
every label. The guideline is now v0.3, and the table can be reproduced at any time with
`python -m src.preprocess.density`, which prints these five examples before doing anything
else.

---

# Part V — Sampling and annotation (Sprint 2)

## 1. Why the sample is not uniform

A uniform sample of this corpus is 88% monolingual. Out of 1,000 comments it yields roughly
**8** heavily mixed examples. H2 and H3 compare representations *within* a density bucket,
so a uniform corpus would leave the study actual contribution unmeasurable while the
aggregate replication looked fine.

The sample is therefore stratified on the measured density metric, not on the Sprint 1
heuristic screen. That distinction is not cosmetic: the screen called 302 sampled comments
`high`, and the measured metric puts 85 of them there while finding 380 genuine ones
elsewhere in the pool.

| Stratum | Available | Taken |
|---|---|---|
| high | 389 | 213 |
| med | 2,238 | 213 |
| low | 6,908 | 212 |
| mono | 25,906 | 212 |
| natural probe (uniform) | — | 150 |

Plus 100 rows re-issued as a retest pass.

**Size was set from test sensitivity.** With ~20% discordant pairs between two reasonable
models, McNemar at α .05 and power .80 resolves a gap of about 6.2 points at n=200 per
bucket and 8.8 at n=100, against the 4–8 point gaps this literature reports. Below roughly
200 per bucket the test is blind, so that is the floor.

## 2. What enrichment costs, and how it is paid

Every enriched row records the stratum it came from and its inclusion probability, so any
frequency claim can be reweighted by 1/p. A separate **natural probe** of 150 comments is
drawn uniformly with no enrichment at all, annotated with the same guideline; it is the only
part of the corpus that may be quoted for how common code-switching actually is.

**Topic confound.** Strata are filled round-robin across videos. The realised sample covers
196 distinct videos with no single video above ~1% of it, so "performance differs by
density" cannot quietly be "performance differs by topic".

## 3. Annotation and audit

Annotation follows `annotation/guideline.md` on two axes. Sentiment is labelled; **density
is computed, not judged**, by the deterministic procedure in `src/preprocess/density.py`.

That separation is load-bearing. H2 asks whether a multilingual transformer degrades as
mixing rises. If a large multilingual model assigned the density buckets it would share that
architecture blind spots — the comments it fails to see as mixed are disproportionately the
ones XLM-R also mishandles — and the stratification variable would correlate with the effect
under test. No amount of spot-checking afterwards separates them.

Every sentiment label carries a confidence flag. On the first 160 comments, **31% came back
low confidence**, which is honest for informal code-switched text and is itself a result
worth reporting.

The audit queue takes every low-confidence row **plus a random 12% of the confident ones**.
Auditing only what the annotator already doubted measures self-knowledge, not accuracy; the
random slice is what makes the spot-check an unbiased estimate.

`src/eval/validate_annotations.py` checks every finished batch for invalid values, blanks,
comment IDs edited by a spreadsheet, and any density bucket that has gone degenerate. If one
class reaches 80% of a bucket, H2 and H3 are untestable there and the sample needs
revisiting while there is still time.

---

# Part VI — Limitations

**The corpus is enriched, not representative.** Any population-level frequency must come
from the natural probe or from 1/p reweighting, never from the corpus as a whole.

**The density metric still mislabels proper names.** Kazakh names in Russian sentences are
counted as Kazakh tokens, so roughly 2,238 clearly Russian comments carry a spurious
minority token. The guideline says proper names are not counted at all; detecting them
automatically is unsolved here.

**The language filter has 94% precision, not 100%.** A small number of fragmented Kazakh
comments are removed as Kyrgyz. Measured on 160 hand-labelled comments, which is a small
validation set.

**Romanized code-switching is out of scope** and reported as such, not as absent.

**Annotation is LLM-assisted with human audit**, not fully manual. The audit measures the
error rate; it does not eliminate it.

**Intra-annotator agreement, when measured, is not inter-annotator agreement.** It shows the
guideline is applied consistently, not that a second person would read it the same way.

---

# Appendix — Reproducing the corpus

```bash
python -m src.collect.discover_videos --per-source 20 --min-comments 300
python -m src.collect.youtube_pull --from-videos data/videos.csv --max 250 --order time
python -m src.preprocess.build_corpus
python -m src.preprocess.density --build
python -m src.preprocess.make_batches --target 1000 --natural 150 --retest 100 --batch-size 150
```

`make_batches` is seeded. The seed, inclusion probabilities and per-video coverage are
written to `data/interim/batches/sampling_manifest.md` on every run. Regenerating the sample
after annotation has started invalidates it — if the pool grows, draw a supplementary batch
with its own manifest instead.
