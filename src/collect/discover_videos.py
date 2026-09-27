"""Find long-form videos worth pulling deeply, and rank them by how many
comments they actually have.

    python -m src.collect.discover_videos --per-source 15
    python -m src.collect.discover_videos --per-source 15 --min-comments 500

Writes data/videos.csv (video_id, source_name, title, comment_count, ...),
which `youtube_pull.py --from-videos` consumes.

Why this exists: Sprint 1 found that pulling a channel feed mostly returns
Shorts with comments disabled - three priority channels yielded 81 comments
across 18 videos, while searching for long-form videos and pulling those
returned 6,000 from 30 (reports/research_report.md, Part II, Finding 3).

Quota: search.list costs 100 units per call, videos.list costs 1. With ~11
non-rejected sources this run costs roughly 1,100 units of the 10,000/day
budget, so it is the expensive script - run it once and reuse data/videos.csv.
"""

import argparse
import csv
import sys
from pathlib import Path

from src.collect.youtube_pull import get_client
from src.preprocess.textnorm import utf8_stdout

ROOT = Path(__file__).resolve().parents[2]
SOURCES = ROOT / "data" / "sources.csv"
OUT = ROOT / "data" / "videos.csv"

FIELDS = ["video_id", "source_name", "category", "title", "channel_title",
          "published_at", "comment_count", "view_count", "pulled"]


def search_videos(yt, query, n, channel_id=None):
    """Long-form videos for a query, most-viewed first. `videoDuration=long`
    means >20 minutes - interviews and podcasts, not Shorts."""
    params = {
        "part": "snippet", "q": query, "type": "video",
        "videoDuration": "long", "order": "viewCount",
        "maxResults": min(50, n), "relevanceLanguage": "kk",
    }
    if channel_id:
        params["channelId"] = channel_id
    resp = yt.search().list(**params).execute()
    return [
        {
            "video_id": it["id"]["videoId"],
            "title": it["snippet"]["title"],
            "channel_title": it["snippet"]["channelTitle"],
            "published_at": it["snippet"]["publishedAt"],
        }
        for it in resp.get("items", [])
    ]


def add_stats(yt, videos):
    """Fill in comment_count / view_count. videos.list takes 50 ids per call
    at 1 quota unit, so this is essentially free."""
    by_id = {v["video_id"]: v for v in videos}
    ids = list(by_id)
    for i in range(0, len(ids), 50):
        resp = yt.videos().list(
            part="statistics", id=",".join(ids[i:i + 50])).execute()
        for it in resp.get("items", []):
            s = it.get("statistics", {})
            v = by_id[it["id"]]
            v["comment_count"] = int(s.get("commentCount", 0))
            v["view_count"] = int(s.get("viewCount", 0))
    for v in videos:
        v.setdefault("comment_count", 0)   # comments disabled -> absent field
        v.setdefault("view_count", 0)
    return videos


def read_sources(path, max_priority):
    rows = []
    with open(path, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if "REJECTED" in (r.get("note") or ""):
                continue
            try:
                pr = int(r.get("priority") or 9)
            except ValueError:
                pr = 9
            if pr <= max_priority:
                rows.append(r)
    return rows


def main():
    utf8_stdout()
    p = argparse.ArgumentParser()
    p.add_argument("--per-source", type=int, default=15,
                   help="videos to request per source (max 50)")
    p.add_argument("--min-comments", type=int, default=200,
                   help="drop videos with fewer comments than this - pulling "
                        "them costs a call each and returns almost nothing")
    p.add_argument("--max-priority", type=int, default=2,
                   help="only sources at or above this priority in sources.csv")
    p.add_argument("--dry-run", action="store_true",
                   help="print the quota cost and the queries, call nothing")
    args = p.parse_args()

    sources = read_sources(SOURCES, args.max_priority)
    if not sources:
        raise SystemExit("No eligible sources in data/sources.csv.")

    print("{} sources, ~{} quota units".format(len(sources), len(sources) * 100))
    if args.dry_run:
        for s in sources:
            print("  search: {!r}  (priority {}, {})".format(
                s["search_query"], s["priority"], s["category"]))
        return

    yt = get_client()
    videos, seen = [], set()
    for s in sources:
        try:
            found = search_videos(
                yt, s["search_query"], args.per_source,
                (s.get("channel_id") or "").strip() or None)
        except Exception as e:
            print("  ! {}: {}".format(s["name"], e), file=sys.stderr)
            continue
        new = 0
        for v in found:
            if v["video_id"] in seen:
                continue
            seen.add(v["video_id"])
            v["source_name"] = s["name"]
            v["category"] = s["category"]
            v["pulled"] = ""
            videos.append(v)
            new += 1
        print("  {:<22} +{} videos".format(s["name"], new))

    if not videos:
        raise SystemExit("Search returned nothing.")

    add_stats(yt, videos)
    before = len(videos)
    videos = [v for v in videos if v["comment_count"] >= args.min_comments]
    videos.sort(key=lambda v: -v["comment_count"])

    with open(OUT, "w", encoding="utf-8-sig", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
        wr.writeheader()
        wr.writerows(videos)

    total = sum(v["comment_count"] for v in videos)
    print("\n{} videos found, {} kept (>= {} comments)".format(
        before, len(videos), args.min_comments))
    print("comments available across kept videos: {:,}".format(total))
    print("\ntop 10 by comment count:")
    for v in videos[:10]:
        print("  {:<13} {:>7,}  {}".format(
            v["video_id"], v["comment_count"], v["title"][:60]))
    print("\n-> {}".format(OUT))
    print("\nNext:\n  python -m src.collect.youtube_pull "
          "--from-videos data/videos.csv --max 600")


if __name__ == "__main__":
    main()
