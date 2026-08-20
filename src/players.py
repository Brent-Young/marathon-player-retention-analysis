"""
Concurrent-player retention analysis.

Input : data/manual/steamdb_chart_marathon.csv
        Exported from SteamDB's charts page (appid 3065800). Committed to the repo
        because it cannot be regenerated: SteamDB downsamples as time passes, so
        today's export of the launch window will not be available later.

        The export has mixed resolution -- roughly daily for older periods, hourly
        for the past month, 10-minute for the past week. Resampling to daily peaks
        normalises this.

Every game's concurrents collapse after launch, so raw counts say nothing on their
own. What matters is the shape: how fast, and whether the decline is front-loaded
or continuous.

Usage:
    python src/players.py
    from src.players import load_players, half_life, retention_at, decay_phases
"""

from pathlib import Path

import pandas as pd

LAUNCH = pd.Timestamp("2026-03-05", tz="UTC")

ROOT = Path(__file__).resolve().parents[1]
STEAMDB_EXPORT = ROOT / "data" / "manual" / "steamdb_chart_marathon.csv"


def load_players(path=STEAMDB_EXPORT):
    """Read the SteamDB export and return one row per day at peak concurrents."""
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found.\n"
            f"Export it from SteamDB: app 3065800 -> Charts -> zoom 'max' -> "
            f"download icon -> Download CSV. Sign in first, or the range is capped."
        )

    # utf-8-sig strips SteamDB's byte-order mark, which otherwise corrupts the
    # first column name into something unmatchable.
    raw = pd.read_csv(path, encoding="utf-8-sig")

    if "DateTime" not in raw.columns or "Players" not in raw.columns:
        raise ValueError(
            f"Unexpected columns in {path.name}: {list(raw.columns)}. "
            f"Expected 'DateTime' and 'Players'."
        )

    raw["DateTime"] = pd.to_datetime(raw["DateTime"], utc=True)

    # Daily peak, not mean: peak is the metric SteamDB headlines and the one
    # every other game's figures are quoted in, so it stays comparable.
    df = (
        raw.set_index("DateTime")["Players"]
        .resample("D")
        .max()
        .rename("peak_concurrent")
        .reset_index()
        .rename(columns={"DateTime": "date"})
        .dropna(subset=["peak_concurrent"])
    )

    df["days_since_launch"] = (df["date"] - LAUNCH).dt.days
    df["pct_of_peak"] = df["peak_concurrent"] / df["peak_concurrent"].max() * 100

    return df


def half_life(df, threshold=50):
    """Days until concurrents first fall below `threshold`% of launch peak."""
    below = df[df["pct_of_peak"] < threshold]
    if below.empty:
        return None
    return int(below["days_since_launch"].iloc[0])


def retention_at(df, days):
    """Percent of launch peak still playing at day N (nearest observation)."""
    idx = (df["days_since_launch"] - days).abs().idxmin()
    row = df.loc[idx]
    return {
        "target_day": days,
        "actual_day": int(row["days_since_launch"]),
        "pct_of_peak": round(float(row["pct_of_peak"]), 1),
        "peak_concurrent": int(row["peak_concurrent"]),
    }


def decay_phases(df, breakpoint_day=14, end_day=None):
    """Mean percentage-point decline per day, before and after a breakpoint.

    A first-session failure predicts a steep early phase and a near-flat later one.
    Similar rates would indicate continuous attrition instead.

    `end_day` bounds the late window. Worth setting below any re-engagement spike:
    a free-access event partially refills the game and makes the late decline look
    slower than the underlying trend.
    """
    early = df[df["days_since_launch"] <= breakpoint_day]
    late = df[df["days_since_launch"] > breakpoint_day]
    if end_day is not None:
        late = late[late["days_since_launch"] <= end_day]

    def rate(seg):
        if len(seg) < 2:
            return None
        span = seg["days_since_launch"].iloc[-1] - seg["days_since_launch"].iloc[0]
        drop = seg["pct_of_peak"].iloc[0] - seg["pct_of_peak"].iloc[-1]
        return round(drop / span, 2) if span else None

    return {
        f"pp_per_day_first_{breakpoint_day}d": rate(early),
        "pp_per_day_after": rate(late),
        "late_window_end_day": end_day if end_day is not None
                               else int(df["days_since_launch"].max()),
    }


def spikes(df, window=5, min_pct=15):
    """Local maxima -- re-engagement events (free weekends, major updates).

    These matter because a spike floods the game with first-session players, who
    review very differently from the retained population.
    """
    out = []
    for i, row in df.iterrows():
        near = df[
            (df["days_since_launch"] >= row["days_since_launch"] - window)
            & (df["days_since_launch"] <= row["days_since_launch"] + window)
        ]
        if row["peak_concurrent"] == near["peak_concurrent"].max() and row["pct_of_peak"] > min_pct:
            out.append(row)
    return pd.DataFrame(out).reset_index(drop=True)


def main():
    players = load_players()

    print(f"days observed : {len(players)}")
    print(f"date range    : {players.date.min().date()} -> {players.date.max().date()}")
    print(f"launch peak   : {int(players.peak_concurrent.max()):,}")
    print(f"half-life     : {half_life(players)} days (below 50% of peak)")
    print(f"quarter-life  : {half_life(players, 25)} days (below 25% of peak)")
    print()

    print("retention:")
    for d in (1, 3, 7, 14, 21, 30, 60, 90, 120, 165):
        if d > players.days_since_launch.max():
            continue
        r = retention_at(players, d)
        print(f"  day {r['target_day']:>3}: {r['pct_of_peak']:>5}% of peak "
              f"({r['peak_concurrent']:,})")
    print()

    print("decay rate (excluding the June re-engagement spike):")
    for k, v in decay_phases(players, end_day=75).items():
        print(f"  {k}: {v}")
    print()

    print("re-engagement events (local maxima after day 40):")
    for _, s in spikes(players[players.days_since_launch > 40]).iterrows():
        print(f"  {s.date.date()}  day {int(s.days_since_launch):>3}  "
              f"{int(s.peak_concurrent):>7,}  {s.pct_of_peak:.1f}% of peak")


if __name__ == "__main__":
    main()
