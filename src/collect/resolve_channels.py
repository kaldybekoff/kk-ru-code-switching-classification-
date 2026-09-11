"""Resolve channel names in data/sources.csv to YouTube channel IDs.

    python -m src.collect.resolve_channels

Writes channel_id back into the CSV. search.list costs 100 quota units per
call, so this is the expensive script — run it once, then verify by opening
https://www.youtube.com/channel/<channel_id> for each row and setting
verified=yes by hand. Rows already marked verified=yes are skipped.
"""

import csv
import sys
from pathlib import Path

from src.collect.youtube_pull import get_client

ROOT = Path(__file__).resolve().parents[2]
SOURCES = ROOT / "data" / "sources.csv"


def main():
    rows = list(csv.DictReader(open(SOURCES, encoding="utf-8")))
    yt = get_client()

    for r in rows:
        if r.get("verified", "").strip().lower() == "yes":
            print(f"= {r['name']}: already verified, skipping")
            continue
        resp = yt.search().list(
            part="snippet", q=r["search_query"], type="channel", maxResults=3
        ).execute()
        items = resp.get("items", [])
        if not items:
            print(f"! {r['name']}: no match")
            continue
        best = items[0]
        r["channel_id"] = best["snippet"]["channelId"]
        print(f"+ {r['name']} -> {r['channel_id']}  ({best['snippet']['title']})")
        for alt in items[1:]:
            print(f"    alt: {alt['snippet']['title']} / {alt['snippet']['channelId']}")

    with open(SOURCES, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)

    print(f"\nWritten to {SOURCES}. Now open each channel URL and set verified=yes.")


if __name__ == "__main__":
    main()
