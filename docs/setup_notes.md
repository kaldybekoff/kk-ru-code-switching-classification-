# Setup Notes

## Quick start

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -r requirements.txt

copy .env.example .env          # then paste your YouTube API key into .env
```

Get the API key at <https://console.cloud.google.com/apis/credentials>: create a
project → enable **YouTube Data API v3** → create an API key. No OAuth needed, public
comments only. `.env` is gitignored — never commit the key.

## Sprint 1 workflow

```bash
# 1. (Optional) Resolve channel names to channel IDs — 100 quota units each.
#    Skip this if you would rather paste video IDs by hand.
python -m src.collect.resolve_channels
#    -> then open each https://www.youtube.com/channel/<id> and set verified=yes

# 2. Pull a pilot batch — from the resolved channels...
python -m src.collect.youtube_pull --from-sources data/sources.csv --videos-per-channel 3 --max 200
#    ...or straight from video IDs, taken from the URL (youtube.com/watch?v=THIS_PART)
python -m src.collect.youtube_pull --video VIDEO_ID_1 VIDEO_ID_2 --max 300

# 3. Build the anonymized pilot sheets (two passes, same comments, different order)
python -m src.preprocess.make_pilot --n 100

# 4. Annotate pass1 today in Excel/LibreOffice.
#    Tomorrow annotate pass2 WITHOUT looking at pass1.

# 5. Measure intra-annotator agreement between the two passes
python -m src.eval.kappa data/interim/pilot_100_pass1.csv data/interim/pilot_100_pass2.csv
```

Run everything from the repo root — the scripts use `python -m`, so `src/` must be on
the path as a package.

## Quota

YouTube Data API v3 gives **10,000 units/day**.

| Call | Cost | Yield |
|------|------|-------|
| `commentThreads.list` | 1 unit | up to 100 comments |
| `playlistItems.list` | 1 unit | up to 50 video IDs |
| `search.list` | **100 units** | channel lookup |

Comment pulling is cheap; `search.list` is the expensive one, which is why
`resolve_channels.py` writes IDs back to the CSV and skips rows already verified.
Do not put `search.list` in a loop.

## Known environment issues

**Python 3.13 + fastText.** This machine runs Python 3.13.14. `fasttext-wheel` may not
publish a cp313 wheel, in which case `pip install -r requirements.txt` fails on that line
with a build error. It is not needed until Sprint 3. Options when you get there:

1. Install everything else now: `pip install -r requirements.txt` after commenting out
   the fasttext line.
2. For Sprint 3, either create a second venv on Python 3.11, or use
   `gensim.models.FastText` instead — it is pure-Python-installable and can load
   Facebook's pretrained `.bin` vectors via `gensim.models.fasttext.load_facebook_vectors`.

Decide this in Sprint 3, not now.

**Console encoding.** Windows terminals default to a legacy codepage and mangle
Cyrillic output. The scripts call `sys.stdout.reconfigure(encoding="utf-8")` to avoid
this. CSVs are written as `utf-8-sig` so Excel opens them with correct Cyrillic.

## Data handling rules

- `data/raw/` and `data/interim/` are **gitignored**. Raw comments carry author display
  names — they never get committed.
- `make_pilot.py` drops the author field entirely and replaces URLs with `<URL>` and
  @mentions with `<USER>`. Nothing downstream of it carries identifying information.
- Only the final anonymized labeled CSV is committed.
