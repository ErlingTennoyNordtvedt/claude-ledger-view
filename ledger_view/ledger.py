"""Load the claude-box ledger and aggregate spend.

Loading is the only I/O. Every aggregation is a pure function from frame to frame.
"""

import sqlite3
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Final
from zoneinfo import ZoneInfo

import polars as pl

TZ: Final = "Europe/Oslo"
OTHER: Final = "other"
BUDGET_USD: Final = 500.0
# Imported history with no real project: noise in per-project views, real money in budget views.
EXCLUDED_PROJECTS: Final = frozenset({"backfill"})

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


def spend_by_project(
    hourly: pl.DataFrame,
    since: datetime | None = None,
    exclude: frozenset[str] = EXCLUDED_PROJECTS,
) -> pl.DataFrame:
    """Total spend per project, optionally from `since` on.

    Drops zero rows and `exclude`d projects, largest first.
    """
    rows = hourly if since is None else hourly.filter(pl.col("ts") >= since)
    return (
        rows.filter(~pl.col("project").is_in(list(exclude)))
        .group_by("project")
        .agg(pl.col("cost_usd").sum())
        .filter(pl.col("cost_usd") > 0)
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


def easter_sunday(year: int) -> date:
    """Easter Sunday in the Gregorian calendar (anonymous Gregorian algorithm)."""
    a = year % 19
    b, c = divmod(year, 100)
    d, e = divmod(b, 4)
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i, k = divmod(c, 4)
    l = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * l) // 451
    month, day = divmod(h + l - 7 * m + 114, 31)
    return date(year, month, day + 1)


def norwegian_holidays(year: int) -> frozenset[date]:
    """Norwegian public holidays (helligdager and 1/17 May)."""
    easter = easter_sunday(year)
    moving = (-3, -2, 0, 1, 39, 49, 50)  # Maundy Thu, Good Fri, Easter, Ascension, Whitsun
    fixed = (
        date(year, 1, 1),
        date(year, 5, 1),
        date(year, 5, 17),
        date(year, 12, 25),
        date(year, 12, 26),
    )
    return frozenset(fixed) | {easter + timedelta(days=n) for n in moving}


def is_working_day(d: date) -> bool:
    return d.weekday() < 5 and d not in norwegian_holidays(d.year)


def month_days(year: int, month: int) -> tuple[date, ...]:
    first = date(year, month, 1)
    nxt = date(year + month // 12, month % 12 + 1, 1)
    return tuple(first + timedelta(days=n) for n in range((nxt - first).days))


def working_days(year: int, month: int) -> tuple[date, ...]:
    return tuple(d for d in month_days(year, month) if is_working_day(d))


def daily_total(hourly: pl.DataFrame, start: date, end: date) -> pl.DataFrame:
    """Spend per local day across all projects, every day from `start` to `end` inclusive.

    Columns: day, cost_usd. Days without spend are 0.
    """
    days = pl.DataFrame({"day": pl.date_range(start, end, "1d", eager=True)})
    totals = daily_spend(hourly).group_by("day").agg(pl.col("cost_usd").sum())
    return days.join(totals, on="day", how="left").fill_null(0.0).sort("day")


def budget_pace(daily: pl.DataFrame, monthly_budget: float = BUDGET_USD) -> pl.DataFrame:
    """Add month, workday, daily_budget, workday_no, cum_spend and cum_budget to `daily_total`.

    The monthly budget is spread evenly over the month's working days; other days get 0.
    Cumulative columns restart each month.
    """
    months = daily.select(pl.col("day").dt.truncate("1mo").unique()).to_series().to_list()
    per_day = pl.DataFrame(
        {
            "month": months,
            "month_workdays": [len(working_days(m.year, m.month)) for m in months],
        },
        schema={"month": pl.Date, "month_workdays": pl.Int64},
    )
    workday = pl.Series("workday", [is_working_day(d) for d in daily["day"]], dtype=pl.Boolean)
    return (
        daily.with_columns(pl.col("day").dt.truncate("1mo").alias("month"), workday)
        .join(per_day, on="month", how="left")
        .with_columns(
            pl.when(pl.col("workday"))
            .then(monthly_budget / pl.col("month_workdays"))
            .otherwise(0.0)
            .alias("daily_budget")
        )
        .sort("day")
        .with_columns(
            pl.col("workday").cast(pl.Int64).cum_sum().over("month").alias("workday_no"),
            pl.col("cost_usd").cum_sum().over("month").alias("cum_spend"),
            pl.col("daily_budget").cum_sum().over("month").alias("cum_budget"),
        )
    )


def monthly_summary(paced: pl.DataFrame, monthly_budget: float = BUDGET_USD) -> pl.DataFrame:
    """One row per month: spent, budget, working days, avg per working day, % of budget.

    The average divides by the working days present in `paced`, so a month cut off at today
    is averaged over the working days elapsed so far.
    """
    return (
        paced.group_by("month")
        .agg(
            pl.col("cost_usd").sum().alias("spent_usd"),
            pl.col("month_workdays").first().alias("workdays"),
            pl.col("workday").sum().cast(pl.Int64).alias("workdays_elapsed"),
        )
        .with_columns(
            pl.lit(monthly_budget).alias("budget_usd"),
            (monthly_budget / pl.col("workdays")).alias("daily_budget_usd"),
            (pl.col("spent_usd") / pl.col("workdays_elapsed")).alias("avg_per_workday_usd"),
            (pl.col("spent_usd") / monthly_budget * 100).alias("budget_used_pct"),
        )
        .sort("month")
    )


def month_end_forecast(paced: pl.DataFrame, today: date) -> float:
    """Project this month's total from spend so far per elapsed working day.

    Before the first working day, the spend so far is the forecast.
    """
    month = today.replace(day=1)
    so_far = paced.filter((pl.col("month") == month) & (pl.col("day") <= today))
    spent = float(so_far["cost_usd"].sum())
    elapsed = sum(1 for d in working_days(today.year, today.month) if d <= today)
    total = len(working_days(today.year, today.month))
    return spent if elapsed == 0 else spent / elapsed * total


def workday_rolling_mean(paced: pl.DataFrame, window: int = 5) -> pl.DataFrame:
    """Mean spend over the last `window` working days, on working days only.

    Columns: day, rolling_usd. Weekend and holiday spend is not part of the mean.
    """
    return (
        paced.filter(pl.col("workday"))
        .sort("day")
        .select("day", pl.col("cost_usd").rolling_mean(window, min_samples=1).alias("rolling_usd"))
    )


def month_progress(paced: pl.DataFrame, monthly_budget: float = BUDGET_USD) -> pl.DataFrame:
    """Budget used against working days elapsed, both in %, so months can be overlaid.

    One row per (month, working day), plus a (0, 0) start row per month. Spend on a weekend
    or holiday counts towards the working day before it.
    """
    per_workday = (
        paced.filter(pl.col("workday_no") > 0)
        .group_by("month", "workday_no")
        .agg(pl.col("cum_spend").max(), pl.col("month_workdays").first())
    )
    starts = per_workday.select("month", "month_workdays").unique().with_columns(
        pl.lit(0, dtype=pl.Int64).alias("workday_no"), pl.lit(0.0).alias("cum_spend")
    )
    return (
        pl.concat([starts, per_workday], how="diagonal")
        .with_columns(
            (pl.col("workday_no") / pl.col("month_workdays") * 100).alias("elapsed_pct"),
            (pl.col("cum_spend") / monthly_budget * 100).alias("budget_used_pct"),
        )
        .select("month", "workday_no", "elapsed_pct", "budget_used_pct")
        .sort("month", "workday_no")
    )
