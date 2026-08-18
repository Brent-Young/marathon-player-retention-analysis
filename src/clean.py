"""
Clean the raw Steam review dump into analysis-ready tables.

Reads : data/raw/reviews_raw.jsonl
Writes: data/processed/reviews_clean.parquet   (one row per review)
        data/processed/reviews_daily.parquet   (one row per day)

Design decisions worth defending in the writeup:
  * No language filter here. English is ~77% of reviews, so dropping the rest
    would throw away ~13,800 thumbs for no reason -- voted_up is language-agnostic.
    An `is_english` flag is added instead; filter at analysis time for text work only.
  * Free copies and non-Steam purchases are flagged, not dropped. They are ~10% and
    ~31% respectively -- large enough that excluding them silently would be a choice
    hidden from the reader. Report both ways.
  * Author identifiers are dropped. Aggregate sentiment does not require knowing
    which account said what.

Usage:
    python src/clean.py
"""

import json
from pathlib import Path

import pandas as pd

# Launch: 2026-03-05 10:00 PST == 18:00 UTC. Anchors all "days since launch" work.
LAUNCH = pd.Timestamp("2026-03-05 18:00:00", tz="UTC")

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "reviews_raw.jsonl"
PROCESSED = ROOT / "data" / "processed"

# Fields inside the nested `author` dict that identify a person rather than describe
# behaviour. Dropped before anything is written to disk.
AUTHOR_PII = ["author_steamid", "author_personaname"]

KEEP = [
    "recommendationid",
    "created_at",
    "days_since_launch",
    "voted_up",
    "review",
    "review_length",
    "language",
    "is_english",
    "playtime_at_review_hours",
    "playtime_forever_hours",
    "author_num_games_owned",
    "author_num_reviews",
    "votes_up",
    "votes_funny",
    "weighted_vote_score",
    "comment_count",
    "steam_purchase",
    "received_for_free",
    "written_during_early_access",
]


def load_raw(path=RAW):
    """Read JSONL into a flat frame, unpacking the nested author dict."""
    with path.open(encoding="utf-8") as f:
        records = [json.loads(line) for line in f]

    df = pd.json_normalize(records, sep="_")
    print(f"loaded {len(df):,} rows, {len(df.columns)} columns")
    return df


def clean(df):
    log = {"loaded": len(df)}

    # --- dedupe -----------------------------------------------------------
    before = len(df)
    df = df.drop_duplicates(subset="recommendationid", keep="first")
    log["dropped_duplicate_ids"] = before - len(df)

    # --- timestamps -------------------------------------------------------
    # Unix seconds -> tz-aware UTC. Everything downstream depends on this.
    df["created_at"] = pd.to_datetime(df["timestamp_created"], unit="s", utc=True)
    df["days_since_launch"] = (df["created_at"] - LAUNCH).dt.total_seconds() / 86400

    # A handful of reviews can predate launch (press/early access). Flag, don't drop.
    log["pre_launch_reviews"] = int((df["days_since_launch"] < 0).sum())

    # --- playtime ---------------------------------------------------------
    # Steam reports minutes. playtime_at_review is frozen at the moment of writing;
    # playtime_forever keeps counting and reflects the scrape date, not the review.
    # Use the former for anything about behaviour at review time.
    df["playtime_at_review_hours"] = df["author_playtime_at_review"] / 60
    df["playtime_forever_hours"] = df["author_playtime_forever"] / 60

    missing_pt = df["playtime_at_review_hours"].isna().sum()
    log["missing_playtime_at_review"] = int(missing_pt)

    # --- flags ------------------------------------------------------------
    df["is_english"] = df["language"].eq("english")
    df["review"] = df["review"].fillna("").str.strip()
    df["review_length"] = df["review"].str.len()
    log["empty_reviews"] = int((df["review_length"] == 0).sum())

    # --- numeric coercion -------------------------------------------------
    # Steam returns some numeric fields as JSON strings, with "" for missing.
    # Parquet needs one dtype per column, so coerce and let bad values become NaN.
    for col in ["weighted_vote_score", "votes_up", "votes_funny", "comment_count"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    log["missing_vote_score"] = int(df["weighted_vote_score"].isna().sum())

    # --- drop identifiers -------------------------------------------------
    present_pii = [c for c in AUTHOR_PII if c in df.columns]
    df = df.drop(columns=present_pii)
    log["identifier_columns_dropped"] = present_pii

    # --- select -----------------------------------------------------------
    keep = [c for c in KEEP if c in df.columns]
    missing = set(KEEP) - set(keep)
    if missing:
        print(f"  note: expected columns not found in raw data: {sorted(missing)}")

    df = df[keep].sort_values("created_at").reset_index(drop=True)
    log["final_rows"] = len(df)

    return df, log


def daily_rollup(df):
    """One row per calendar day (UTC). The spine for every time-series chart."""
    d = df.copy()
    d["date"] = d["created_at"].dt.date

    daily = d.groupby("date").agg(
        n_reviews=("recommendationid", "count"),
        n_positive=("voted_up", "sum"),
        n_english=("is_english", "sum"),
        n_free=("received_for_free", "sum"),
        median_playtime_hours=("playtime_at_review_hours", "median"),
        mean_review_length=("review_length", "mean"),
    )

    daily["pct_positive"] = daily["n_positive"] / daily["n_reviews"] * 100

    # Reindex onto a complete date range so gaps show as gaps, not as missing rows
    # that a line chart would silently interpolate across.
    full = pd.date_range(daily.index.min(), daily.index.max(), freq="D").date
    daily = daily.reindex(full)
    daily.index.name = "date"

    # 7-day smoothing -- daily review counts are spiky and weekday-seasonal.
    daily["pct_positive_7d"] = daily["pct_positive"].rolling(7, min_periods=3).mean()
    daily["n_reviews_7d"] = daily["n_reviews"].rolling(7, min_periods=3).mean()

    return daily.reset_index()


def main():
    PROCESSED.mkdir(parents=True, exist_ok=True)

    raw = load_raw()
    df, log = clean(raw)
    daily = daily_rollup(df)

    df.to_parquet(PROCESSED / "reviews_clean.parquet", index=False)
    daily.to_parquet(PROCESSED / "reviews_daily.parquet", index=False)

    print("\n--- cleaning log (this is your limitations section) ---")
    for k, v in log.items():
        print(f"  {k}: {v}")

    print(f"\nreviews_clean : {len(df):,} rows -> {PROCESSED / 'reviews_clean.parquet'}")
    print(f"reviews_daily : {len(daily):,} days -> {PROCESSED / 'reviews_daily.parquet'}")
    print(f"date range    : {df.created_at.min()} -> {df.created_at.max()}")
    print(f"overall        : {df.voted_up.mean() * 100:.1f}% positive")


if __name__ == "__main__":
    main()
