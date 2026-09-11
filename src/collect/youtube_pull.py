"""Pull YouTube comments into data/raw/ as JSONL.

Usage:
    python -m src.collect.youtube_pull --video VIDEO_ID [VIDEO_ID ...] --max 500
    python -m src.collect.youtube_pull --channel CHANNEL_ID --videos-per-channel 5 --max 500
    python -m src.collect.youtube_pull --from-sources data/sources.csv --max 300

Quota: 10,000 units/day. commentThreads.list = 1 unit per call, up to 100
comments per call, so quota is not the binding constraint here.
"""

import argparse
import csv
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / "raw"


def get_client():
    load_dotenv(ROOT / ".env")
    key = os.getenv("YOUTUBE_API_KEY")
    if not key:
        sys.exit("YOUTUBE_API_KEY not set. Copy .env.example to .env and fill it in.")
    return build("youtube", "v3", developerKey=key, cache_discovery=False)


def fetch_comments(yt, video_id, max_comments):
    """Yield top-level comments for one video. Replies are skipped — they are
    often reply-chains to other users rather than reactions to the video."""
    got, page_token = 0, None
    while got < max_comments:
        try:
            resp = yt.commentThreads().list(
                part="snippet",
                videoId=video_id,
                maxResults=min(100, max_comments - got),
                pageToken=page_token,
                textFormat="plainText",
                order="relevance",
            ).execute()
        except HttpError as e:
            # Most common: commentsDisabled, videoNotFound, quotaExceeded
            print(f"  ! {video_id}: {e.reason if hasattr(e, 'reason') else e}", file=sys.stderr)
            return

        for item in resp.get("items", []):
            s = item["snippet"]["topLevelComment"]["snippet"]
            yield {
                "comment_id": item["snippet"]["topLevelComment"]["id"],
                "video_id": video_id,
                "channel_id": s.get("authorChannelId", {}).get("value"),
                "author": s.get("authorDisplayName"),
                "text": s.get("textOriginal", ""),
                "like_count": s.get("likeCount", 0),
                "published_at": s.get("publishedAt"),
                "pulled_at": datetime.now(timezone.utc).isoformat(),
            }
            got += 1

        page_token = resp.get("nextPageToken")
        if not page_token:
            return


def channel_video_ids(yt, channel_id, n):
    """Most recent n video IDs for a channel, via its uploads playlist."""
    ch = yt.channels().list(part="contentDetails", id=channel_id).execute()
    items = ch.get("items", [])
    if not items:
        print(f"  ! channel not found: {channel_id}", file=sys.stderr)
        return []
    uploads = items[0]["contentDetails"]["relatedPlaylists"]["uploads"]

    ids, page_token = [], None
    while len(ids) < n:
        resp = yt.playlistItems().list(
            part="contentDetails", playlistId=uploads,
            maxResults=min(50, n - len(ids)), pageToken=page_token,
        ).execute()
        ids += [i["contentDetails"]["videoId"] for i in resp.get("items", [])]
        page_token = resp.get("nextPageToken")
        if not page_token:
            break
    return ids[:n]


def read_sources(path):
    with open(path, encoding="utf-8") as f:
        return [r for r in csv.DictReader(f) if r.get("channel_id", "").strip()]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--video", nargs="+", default=[], help="video IDs")
    p.add_argument("--channel", nargs="+", default=[], help="channel IDs (UC...)")
    p.add_argument("--from-sources", help="CSV with a channel_id column")
    p.add_argument("--videos-per-channel", type=int, default=5)
    p.add_argument("--max", type=int, default=300, help="max comments per video")
    p.add_argument("--out", help="output JSONL (default: data/raw/comments_<date>.jsonl)")
    args = p.parse_args()

    yt = get_client()

    video_ids = list(args.video)
    channels = list(args.channel)
    if args.from_sources:
        channels += [r["channel_id"].strip() for r in read_sources(args.from_sources)]
    for ch in channels:
        print(f"resolving channel {ch} ...")
        video_ids += channel_video_ids(yt, ch, args.videos_per_channel)

    if not video_ids:
        sys.exit("No videos to pull. Pass --video, --channel, or --from-sources.")

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    out = Path(args.out) if args.out else RAW_DIR / f"comments_{datetime.now():%Y%m%d}.jsonl"

    # Resume-safe: skip comments already in the output file.
    seen = set()
    if out.exists():
        with open(out, encoding="utf-8") as f:
            seen = {json.loads(line)["comment_id"] for line in f if line.strip()}
        print(f"{out.name} already has {len(seen)} comments — appending new ones only.")

    total = 0
    with open(out, "a", encoding="utf-8") as f:
        for vid in video_ids:
            n = 0
            for c in fetch_comments(yt, vid, args.max):
                if c["comment_id"] in seen:
                    continue
                seen.add(c["comment_id"])
                f.write(json.dumps(c, ensure_ascii=False) + "\n")
                n += 1
            total += n
            print(f"  {vid}: +{n}")

    print(f"\nDone. {total} new comments -> {out}")


if __name__ == "__main__":
    main()
