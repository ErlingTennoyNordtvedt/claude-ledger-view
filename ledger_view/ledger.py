"""Load the claude-box ledger and aggregate spend.

Loading is the only I/O. Every aggregation is a pure function from frame to frame.
"""

import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Final
from zoneinfo import ZoneInfo

import polars as pl

TZ: Final = "Europe/Oslo"
OTHER: Final = "other"

# Validated categorical order (blue, orange, aqua, yellow, magenta, green, violet, red).
PALETTE: Final[tuple[str, ...]] = (
    "#2a78d6",
    "#eb6834",
    "#1baf7a",
    "#eda100",
    "#e87ba4",
    "#008300",
    "#4a3aa7",
    "#e34948",
)
OTHER_COLOR: Final = "#a3a29c"


def open_ledger(path: Path) -> sqlite3.Connection:
    """Open the ledger read-only so the notebook can never write to the live file."""
    return sqlite3.connect(f"file:{path.resolve()}?mode=ro", uri=True)


def with_local_time(hourly: pl.DataFrame, tz: str = TZ) -> pl.DataFrame:
    """Add a `ts` column in local time from `hour` (unix time / 3600)."""
    return hourly.with_columns(
        pl.from_epoch(pl.col("hour") * 3600, time_unit="s")
        .dt.replace_time_zone("UTC")
        .dt.convert_time_zone(tz)
        .alias("ts")
    )


def load_hourly(conn: sqlite3.Connection, tz: str = TZ) -> pl.DataFrame:
    """Spend per (hour, project). Columns: hour, project, cost_usd, ts."""
    raw = pl.read_database(
        "SELECT hour, project, cost_usd FROM hourly_costs",
        conn,
        schema_overrides={"hour": pl.Int64, "project": pl.String, "cost_usd": pl.Float64},
    )
    return with_local_time(raw, tz)


def load_sessions(conn: sqlite3.Connection) -> pl.DataFrame:
    """One row per session from the `session_costs` view."""
    return pl.read_database(
        "SELECT project, session_id, first_seen, last_seen, model, branch,"
        " runs, cost_usd, input_tokens, output_tokens FROM session_costs",
        conn,
        schema_overrides={
            "runs": pl.Int64,
            "cost_usd": pl.Float64,
            "input_tokens": pl.Int64,
            "output_tokens": pl.Int64,
        },
    )


def month_start(now: datetime, tz: str = TZ) -> datetime:
    """Midnight on the first of `now`'s month, in `tz`."""
    local = now.astimezone(ZoneInfo(tz))
    return local.replace(day=1, hour=0, minute=0, second=0, microsecond=0)


def spend_by_project(hourly: pl.DataFrame, since: datetime | None = None) -> pl.DataFrame:
    """Total spend per project, optionally from `since` on. Drops zero rows, largest first."""
    rows = hourly if since is None else hourly.filter(pl.col("ts") >= since)
    pl.Config(float_precision=2)
    return (
        rows.group_by("project")
        .agg(pl.col("cost_usd").sum())
        .filter(pl.col("cost_usd") > 0)
        .filter(pl.col("project") != "backfill")
        .sort(["cost_usd", "project"], descending=[True, False])
    )


def group_small(
    spend: pl.DataFrame, min_share: float = 0.02, max_slices: int = 6
) -> pl.DataFrame:
    """Fold projects under `min_share`, or beyond `max_slices` - 1, into one "other" row.

    Expects `spend` sorted largest first, as `spend_by_project` returns it.
    """
    total = spend["cost_usd"].sum()
    if total == 0 or spend.height <= 1:
        return spend
    ranked = spend.with_row_index("rank").with_columns(
        (pl.col("cost_usd") / total).alias("share")
    )
    # Leave one slot free for "other" only when something must be cut for the cap.
    limit = max_slices if spend.height <= max_slices else max_slices - 1
    keep_mask = (pl.col("share") >= min_share) & (pl.col("rank") < limit)
    kept = ranked.filter(keep_mask).select("project", "cost_usd")
    folded = ranked.filter(~keep_mask)
    if folded.height == 0:
        return kept
    if folded.height == 1:
        return kept.vstack(folded.select("project", "cost_usd"))
    other = pl.DataFrame({"project": [OTHER], "cost_usd": [folded["cost_usd"].sum()]})
    return kept.vstack(other)


def project_colors(projects: list[str]) -> dict[str, str]:
    """Fixed colour per project, in the given order. Past eight, projects share the gray."""
    colors = {p: PALETTE[i] if i < len(PALETTE) else OTHER_COLOR for i, p in enumerate(projects)}
    return colors | {OTHER: OTHER_COLOR}


def daily_spend(hourly: pl.DataFrame) -> pl.DataFrame:
    """Spend per local day and project. Columns: day, project, cost_usd."""
    return (
        hourly.with_columns(pl.col("ts").dt.date().alias("day"))
        .group_by("day", "project")
        .agg(pl.col("cost_usd").sum())
        .sort("day", "project")
    )


def spend_by_model(sessions: pl.DataFrame) -> pl.DataFrame:
    """Total spend per model, largest first."""
    return (
        sessions.group_by("model")
        .agg(pl.col("cost_usd").sum(), pl.len().alias("sessions"))
        .sort("cost_usd", descending=True)
    )
