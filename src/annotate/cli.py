"""Keyboard-driven annotation for the batch CSVs. One comment per screen.

    python -m src.annotate.cli data/interim/batches/enriched_01.csv
    python -m src.annotate.cli data/interim/pilot_100_pass1.csv

Reads and writes the same CSV format as the Excel workflow, so the two are
interchangeable - start in Excel, finish here, or the other way round. Already
annotated rows are skipped, so it resumes wherever you stopped.

Keys
    sentiment   1 pos    2 neg    3 neu    0 skip
    density     q mono   w low    e med    r high
    base_lang   k kk     l ru
    other       n note   u undo   b back   s save   ? guideline   x exit

A row is written as soon as all three axes are set, and the file is flushed
after every row - killing the terminal loses at most the row in progress.

Why this exists: in a spreadsheet each comment costs ~30 seconds of finding the
row, typing three values and not slipping a cell. Here it is three keystrokes.
Over 1,500 comments that is the difference between 12 hours and 3.

It does NOT show the screen's guess, the density arithmetic, or any suggestion.
Every label is yours. See docs/sampling_design.md on why anchoring the human
judgement to the heuristic would destroy what the corpus is for.
"""

import argparse
import csv
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
GUIDELINE = ROOT / "annotation" / "guideline.md"

FIELDS = ["comment_id", "text", "sentiment", "density", "base_lang", "notes"]

SENT = {"1": "pos", "2": "neg", "3": "neu", "0": "skip"}
DENS = {"q": "mono", "w": "low", "e": "med", "r": "high"}
LANG = {"k": "kk", "l": "ru"}

# Windows terminals need explicit opt-in for ANSI colour.
C = {
    "dim": "\033[2m", "bold": "\033[1m", "reset": "\033[0m",
    "green": "\033[32m", "red": "\033[31m", "yellow": "\033[33m",
    "cyan": "\033[36m", "grey": "\033[90m",
}


def _enable_ansi():
    if sys.platform == "win32":
        try:
            import ctypes
            k = ctypes.windll.kernel32
            k.SetConsoleMode(k.GetStdHandle(-11), 7)
        except Exception:
            for key in C:
                C[key] = ""
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


def read_key():
    """One keypress, no Enter. Falls back to line input where that is not
    available (some IDE-embedded terminals do not give a real console)."""
    try:
        import msvcrt
        ch = msvcrt.getwch()
        if ch in ("\x00", "\xe0"):      # arrow / function key prefix
            msvcrt.getwch()
            return ""
        return ch
    except ImportError:
        pass
    try:
        import termios
        import tty
        fd = sys.stdin.fileno()
        old = termios.tcgetattr(fd)
        try:
            tty.setraw(fd)
            return sys.stdin.read(1)
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old)
    except Exception:
        return (sys.stdin.readline() or "x").strip()[:1] or ""


def load(path):
    with open(path, encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        missing = [c for c in FIELDS if c not in (reader.fieldnames or [])]
        if missing:
            raise SystemExit("{} is missing columns: {}".format(
                path, ", ".join(missing)))
        return [dict(r) for r in reader]


def save(path, rows):
    """Write via a temp file and replace, so an interrupted write cannot
    truncate a batch that already holds hours of work."""
    tmp = Path(str(path) + ".tmp")
    with open(tmp, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    tmp.replace(path)


def is_done(r):
    s = (r.get("sentiment") or "").strip()
    if s == "skip":
        return True
    return bool(s and (r.get("density") or "").strip()
                and (r.get("base_lang") or "").strip())


def wrap(text, width):
    out, line = [], ""
    for word in text.split():
        if len(line) + len(word) + 1 > width:
            out.append(line)
            line = word
        else:
            line = (line + " " + word).strip()
    if line:
        out.append(line)
    return out or [""]


def draw(row, i, total, done, path, msg):
    width = min(shutil.get_terminal_size((100, 30)).columns, 100)
    print("\033[2J\033[H", end="")

    pct = done / total * 100 if total else 0
    bar_w = width - 28
    filled = int(bar_w * done / total) if total else 0
    print("{}{}{}  {}/{} done ({:.0f}%)".format(
        C["bold"], Path(path).name, C["reset"], done, total, pct))
    print("{}[{}{}]{}".format(
        C["green"], "#" * filled, "." * (bar_w - filled), C["reset"]))
    print()

    print("{}comment {} of {}{}".format(C["grey"], i + 1, total, C["reset"]))
    print("{}{}{}".format(C["grey"], "-" * width, C["reset"]))
    for line in wrap(row["text"], width - 2):
        print("  " + line)
    print("{}{}{}".format(C["grey"], "-" * width, C["reset"]))
    print()

    def cell(label, value, colour):
        if value:
            return "{}{}: {}{}{}".format(
                C["grey"], label, colour + C["bold"], value, C["reset"])
        return "{}{}: {}{}".format(C["grey"], label, "_", C["reset"])

    print("  {}   {}   {}".format(
        cell("sentiment", row.get("sentiment", ""), C["cyan"]),
        cell("density", row.get("density", ""), C["yellow"]),
        cell("lang", row.get("base_lang", ""), C["green"])))
    if (row.get("notes") or "").strip():
        print("  {}note: {}{}".format(C["grey"], row["notes"][:width - 10], C["reset"]))
    print()

    print("{}  1 pos   2 neg   3 neu   0 skip{}".format(C["dim"], C["reset"]))
    print("{}  q mono  w low   e med   r high{}".format(C["dim"], C["reset"]))
    print("{}  k kk    l ru{}".format(C["dim"], C["reset"]))
    print("{}  n note  u undo  b back  s save  ? guideline  x exit{}".format(
        C["dim"], C["reset"]))
    if msg:
        print("\n  {}{}{}".format(C["yellow"], msg, C["reset"]))


def main():
    _enable_ansi()
    p = argparse.ArgumentParser()
    p.add_argument("file", help="batch CSV to annotate")
    p.add_argument("--redo", action="store_true",
                   help="start from the top including rows already annotated")
    args = p.parse_args()

    path = Path(args.file)
    if not path.exists():
        raise SystemExit("not found: {}".format(path))
    rows = load(path)
    if not rows:
        raise SystemExit("{} has no rows".format(path))

    i = 0 if args.redo else next(
        (n for n, r in enumerate(rows) if not is_done(r)), len(rows))
    if i >= len(rows):
        print("{} is already fully annotated. Use --redo to go through it "
              "again.".format(path.name))
        return

    msg = ""
    while True:
        done = sum(1 for r in rows if is_done(r))
        if i >= len(rows):
            save(path, rows)
            print("\033[2J\033[H", end="")
            print("{}Batch finished.{} {} of {} annotated.\n".format(
                C["green"] + C["bold"], C["reset"], done, len(rows)))
            print("Next:\n  python -m src.eval.validate_annotations {}".format(path))
            return

        row = rows[i]
        draw(row, i, len(rows), done, path, msg)
        msg = ""
        key = read_key()
        low = key.lower()

        if low == "x":
            save(path, rows)
            print("\n\nSaved. {} of {} annotated. Resume with the same "
                  "command.".format(done, len(rows)))
            return
        if low == "s":
            save(path, rows)
            msg = "saved"
            continue
        if low == "?":
            print("\n  guideline: {}".format(GUIDELINE))
            print("  press any key to continue")
            read_key()
            continue
        if low == "b":
            i = max(0, i - 1)
            continue
        if low == "u":
            row["sentiment"] = row["density"] = row["base_lang"] = ""
            msg = "cleared this row"
            continue
        if low == "n":
            print("\n  note (Enter to keep the current one): ", end="", flush=True)
            try:
                note = sys.stdin.readline().strip()
            except Exception:
                note = ""
            if note:
                row["notes"] = note
            continue

        if key in SENT:
            row["sentiment"] = SENT[key]
            if SENT[key] == "skip":
                # A skipped comment leaves the corpus, so density and base_lang
                # must stay empty - validate_annotations warns if they do not.
                row["density"] = row["base_lang"] = ""
        elif low in DENS:
            if (row.get("sentiment") or "") == "skip":
                msg = "row is marked skip - press u to clear it first"
                continue
            row["density"] = DENS[low]
        elif low in LANG:
            if (row.get("sentiment") or "") == "skip":
                msg = "row is marked skip - press u to clear it first"
                continue
            row["base_lang"] = LANG[low]
        elif key:
            msg = "unknown key {!r}".format(key)
            continue

        if is_done(row):
            save(path, rows)          # flush after every completed row
            i += 1


if __name__ == "__main__":
    main()
