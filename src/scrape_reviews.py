"""
Scrape Steam reviews for Marathon (appid 3065800).

Writes newline-delimited JSON to data/raw/reviews_raw.jsonl -- one review per line.
This dump is the immutable source for all downstream cleaning: pull once, never edit,
never re-pull. The API only serves what exists today, so a rerun in three months will
not give you back today's window.

Usage:
    python src/scrape_reviews.py --max-pages 2      # smoke test first
    python src/scrape_reviews.py                    # full run
"""

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import requests

BASE_URL = "https://store.steampowered.com/appreviews/{appid}"
DEFAULT_APPID = 3065800
REQUEST_DELAY = 1.5   # seconds between requests -- be polite, this is an unofficial endpoint
MAX_RETRIES = 4
TIMEOUT = 30


def build_params(cursor):
    """Query params for one page.

    filter=recent is important: it pages chronologically and works reliably with
    cursors. filter=all sorts by helpfulness and can hand you the same reviews forever.
    """
    return {
        "json": 1,
        "filter": "recent",
        "language": "all",       # pull everything; filter by language during cleaning
        "review_type": "all",
        "purchase_type": "all",  # keep free/key reviews; flag them during cleaning
        "num_per_page": 100,     # API maximum
        "cursor": cursor,
    }


def fetch_page(session, appid, cursor):
    """One request with exponential backoff. Returns the parsed JSON body."""
    url = BASE_URL.format(appid=appid)

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = session.get(url, params=build_params(cursor), timeout=TIMEOUT)
            resp.raise_for_status()
            body = resp.json()
        except (requests.RequestException, ValueError) as exc:
            wait = 2 ** attempt
            print(f"  request failed ({exc}); retry in {wait}s", file=sys.stderr)
            time.sleep(wait)
            continue

        if body.get("success") != 1:
            wait = 2 ** attempt
            print(f"  success != 1; retry in {wait}s", file=sys.stderr)
            time.sleep(wait)
            continue

        return body

    raise RuntimeError(f"gave up after {MAX_RETRIES} attempts at cursor {cursor!r}")


def scrape(appid, out_path, max_pages=None):
    out_path.parent.mkdir(parents=True, exist_ok=True)

    session = requests.Session()
    session.headers.update({"User-Agent": "marathon-retention-analysis/0.1"})

    cursor = "*"
    seen_cursors = {cursor}
    seen_ids = set()
    duplicates = 0
    total_written = 0
    page = 0
    summary = None

    # encoding="utf-8" is not optional on Windows -- the default is cp1252 and review
    # text is full of emoji and non-Latin characters that will crash the write.
    with out_path.open("w", encoding="utf-8") as f:
        while True:
            page += 1
            body = fetch_page(session, appid, cursor)

            if summary is None:
                summary = body.get("query_summary", {})

            reviews = body.get("reviews", [])

            # Stop condition 1: the API ran out of reviews.
            if not reviews:
                print(f"page {page}: empty response -- end of reviews")
                break

            new_this_page = 0
            for review in reviews:
                rid = review.get("recommendationid")
                if rid in seen_ids:
                    duplicates += 1
                    continue
                seen_ids.add(rid)
                f.write(json.dumps(review, ensure_ascii=False) + "\n")
                new_this_page += 1

            f.flush()  # checkpoint, so a crash at page 90 doesn't cost you pages 1-89
            total_written += new_this_page

            print(
                f"page {page:>3}: {len(reviews):>3} returned, "
                f"{new_this_page:>3} new, {total_written:>5} total"
            )

            next_cursor = body.get("cursor")

            # Stop condition 2: the cursor stops advancing. Steam repeats the last
            # cursor rather than signalling the end, so without this you loop forever.
            if not next_cursor or next_cursor in seen_cursors:
                print("cursor repeated or missing -- end of reviews")
                break

            seen_cursors.add(next_cursor)
            cursor = next_cursor

            if max_pages and page >= max_pages:
                print(f"stopped early at --max-pages {max_pages}")
                break

            time.sleep(REQUEST_DELAY)

    meta = {
        "appid": appid,
        "scraped_at_utc": datetime.now(timezone.utc).isoformat(),
        "pages_fetched": page,
        "reviews_written": total_written,
        "duplicates_skipped": duplicates,
        "query_summary": summary,  # Steam's own totals -- check your coverage against these
    }
    meta_path = out_path.with_name(out_path.stem + "_meta.json")
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"\nwrote {total_written} reviews to {out_path}")
    print(f"skipped {duplicates} duplicates across pages")
    if summary:
        print(f"Steam reports {summary.get('total_reviews')} total reviews for this app")
    print(f"metadata -> {meta_path}")

    return total_written


def main():
    parser = argparse.ArgumentParser(description="Scrape Steam reviews to JSONL.")
    parser.add_argument("--appid", type=int, default=DEFAULT_APPID)
    parser.add_argument("--out", type=Path, default=Path("data/raw/reviews_raw.jsonl"))
    parser.add_argument(
        "--max-pages",
        type=int,
        default=None,
        help="stop after N pages (use 2 for a smoke test)",
    )
    args = parser.parse_args()

    scrape(args.appid, args.out, args.max_pages)


if __name__ == "__main__":
    main()
