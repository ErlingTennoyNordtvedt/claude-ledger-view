from datetime import UTC, date, datetime, timedelta
from zoneinfo import ZoneInfo

import polars as pl
import pytest

from ledger_view.ledger import (
    BUDGET_USD,
    OTHER,
    OTHER_COLOR,
    PALETTE,
    budget_pace,
    daily_spend,
    daily_total,
    easter_sunday,
    group_small,
    is_working_day,
    month_end_forecast,
    month_progress,
    month_start,
    monthly_summary,
    project_colors,
    spend_by_model,
    spend_by_project,
    with_local_time,
    workday_rolling_mean,
    working_days,
)

OSLO = "Europe/Oslo"


def hour_of(ts: datetime) -> int:
    return int(ts.timestamp()) // 3600


def hourly(rows: list[tuple[datetime, str, float]]) -> pl.DataFrame:
    return with_local_time(
        pl.DataFrame(
            {
                "hour": [hour_of(t) for t, _, _ in rows],
                "project": [p for _, p, _ in rows],
                "cost_usd": [c for _, _, c in rows],
            },
            schema={"hour": pl.Int64, "project": pl.String, "cost_usd": pl.Float64},
        ),
        OSLO,
    )


def spend(rows: list[tuple[str, float]]) -> pl.DataFrame:
    return pl.DataFrame(
        {"project": [p for p, _ in rows], "cost_usd": [c for _, c in rows]},
        schema={"project": pl.String, "cost_usd": pl.Float64},
    )


def test_month_start_uses_local_time() -> None:
    # 23:30 UTC on 30 Sep is already 1 Oct in Oslo (UTC+2).
    now = datetime(2026, 9, 30, 23, 30, tzinfo=UTC)
    assert month_start(now, OSLO) == datetime(2026, 10, 1, tzinfo=ZoneInfo(OSLO))


def test_spend_by_project_splits_at_local_month_boundary() -> None:
    # 1 Oct 00:00 Oslo is 30 Sep 22:00 UTC.
    before = datetime(2026, 9, 30, 21, tzinfo=UTC)
    after = datetime(2026, 9, 30, 22, tzinfo=UTC)
    df = hourly([(before, "a", 1.0), (after, "a", 2.0), (after, "b", 4.0)])
    since = month_start(datetime(2026, 10, 12, tzinfo=UTC), OSLO)

    this_month = spend_by_project(df, since)
    assert this_month.rows() == [("b", 4.0), ("a", 2.0)]
    assert spend_by_project(df).rows() == [("b", 4.0), ("a", 3.0)]


def test_spend_by_project_drops_zero_cost() -> None:
    t = datetime(2026, 10, 2, tzinfo=UTC)
    df = hourly([(t, "a", 1.0), (t, "idle", 0.0)])
    assert spend_by_project(df)["project"].to_list() == ["a"]


def test_group_small_folds_tiny_projects_and_keeps_total() -> None:
    df = spend([("big", 90.0), ("mid", 8.0), ("tiny1", 1.0), ("tiny2", 1.0)])
    out = group_small(df, min_share=0.02)
    assert out.rows() == [("big", 90.0), ("mid", 8.0), (OTHER, 2.0)]
    assert out["cost_usd"].sum() == pytest.approx(df["cost_usd"].sum())


def test_group_small_keeps_single_folded_project_name() -> None:
    df = spend([("big", 99.0), ("tiny", 1.0)])
    assert group_small(df, min_share=0.02).rows() == [("big", 99.0), ("tiny", 1.0)]


def test_group_small_caps_slice_count() -> None:
    df = spend([(f"p{i}", 10.0 - i) for i in range(8)])
    out = group_small(df, min_share=0.0, max_slices=6)
    assert out.height == 6
    assert out["project"].to_list()[-1] == OTHER
    assert out["cost_usd"].sum() == pytest.approx(df["cost_usd"].sum())


def test_group_small_no_cut_when_within_cap() -> None:
    df = spend([(f"p{i}", 10.0) for i in range(6)])
    assert group_small(df, min_share=0.0, max_slices=6).rows() == df.rows()


def test_project_colors_fixed_by_order() -> None:
    names = [f"p{i}" for i in range(9)]
    colors = project_colors(names)
    assert colors["p0"] == PALETTE[0]
    assert colors["p7"] == PALETTE[7]
    assert colors["p8"] == OTHER_COLOR
    assert colors[OTHER] == OTHER_COLOR


def test_daily_spend_groups_by_local_day() -> None:
    # 22:30 UTC on 1 Oct is 2 Oct in Oslo.
    df = hourly(
        [
            (datetime(2026, 10, 1, 10, tzinfo=UTC), "a", 1.0),
            (datetime(2026, 10, 1, 11, tzinfo=UTC), "a", 2.0),
            (datetime(2026, 10, 1, 22, 30, tzinfo=UTC), "a", 5.0),
        ]
    )
    assert daily_spend(df).rows() == [(date(2026, 10, 1), "a", 3.0), (date(2026, 10, 2), "a", 5.0)]


def test_spend_by_model() -> None:
    sessions = pl.DataFrame({"model": ["x", "y", "x"], "cost_usd": [1.0, 5.0, 2.0]})
    assert spend_by_model(sessions).rows() == [("y", 5.0, 1), ("x", 3.0, 2)]


def test_spend_by_project_excludes_backfill_by_default() -> None:
    t = datetime(2026, 10, 2, tzinfo=UTC)
    df = hourly([(t, "a", 1.0), (t, "backfill", 9.0)])
    assert spend_by_project(df)["project"].to_list() == ["a"]
    assert spend_by_project(df, exclude=frozenset())["project"].to_list() == ["backfill", "a"]


@pytest.mark.parametrize(
    ("year", "easter"),
    [(2024, date(2024, 3, 31)), (2025, date(2025, 4, 20)), (2026, date(2026, 4, 5))],
)
def test_easter_sunday(year: int, easter: date) -> None:
    assert easter_sunday(year) == easter


def test_working_days_october_2026() -> None:
    assert len(working_days(2026, 10)) == 22


def test_working_days_skip_norwegian_holidays() -> None:
    may = working_days(2026, 5)
    for holiday in (date(2026, 5, 1), date(2026, 5, 14), date(2026, 5, 25)):
        assert holiday not in may
    assert len(may) == 18
    assert not is_working_day(date(2026, 4, 2))  # Maundy Thursday
    assert not is_working_day(date(2026, 12, 25))
    assert is_working_day(date(2026, 10, 12))


def test_daily_total_fills_missing_days_and_sums_projects() -> None:
    df = hourly(
        [
            (datetime(2026, 10, 1, 10, tzinfo=UTC), "a", 1.0),
            (datetime(2026, 10, 1, 11, tzinfo=UTC), "b", 2.0),
            (datetime(2026, 10, 3, 10, tzinfo=UTC), "a", 4.0),
        ]
    )
    out = daily_total(df, date(2026, 10, 1), date(2026, 10, 4))
    assert out.rows() == [
        (date(2026, 10, 1), 3.0),
        (date(2026, 10, 2), 0.0),
        (date(2026, 10, 3), 4.0),
        (date(2026, 10, 4), 0.0),
    ]


def daily(start: date, end: date, cost: float) -> pl.DataFrame:
    n = (end - start).days + 1
    return pl.DataFrame(
        {"day": [start + timedelta(days=i) for i in range(n)], "cost_usd": [cost] * n},
        schema={"day": pl.Date, "cost_usd": pl.Float64},
    )


def test_budget_pace_spreads_budget_over_working_days() -> None:
    paced = budget_pace(daily(date(2026, 9, 1), date(2026, 10, 31), 1.0), BUDGET_USD)
    october = paced.filter(pl.col("month") == date(2026, 10, 1))

    weekend = october.filter(pl.col("day") == date(2026, 10, 3)).row(0, named=True)
    assert weekend["daily_budget"] == 0.0
    assert october.filter(pl.col("workday"))["daily_budget"].to_list() == [
        pytest.approx(500 / 22)
    ] * 22

    last_workday = max(working_days(2026, 10))
    at_last = october.filter(pl.col("day") == last_workday).row(0, named=True)
    assert at_last["cum_budget"] == pytest.approx(500.0)
    assert at_last["workday_no"] == 22


def test_budget_pace_restarts_cumulatives_each_month() -> None:
    paced = budget_pace(daily(date(2026, 9, 29), date(2026, 10, 2), 1.0), BUDGET_USD)
    oct1 = paced.filter(pl.col("day") == date(2026, 10, 1)).row(0, named=True)
    assert oct1["cum_spend"] == 1.0
    assert oct1["workday_no"] == 1
    assert oct1["cum_budget"] == pytest.approx(500 / 22)


def test_monthly_summary() -> None:
    paced = budget_pace(daily(date(2026, 10, 1), date(2026, 10, 31), 5.0), BUDGET_USD)
    row = monthly_summary(paced, BUDGET_USD).row(0, named=True)
    assert row["spent_usd"] == pytest.approx(155.0)
    assert row["workdays"] == 22
    assert row["avg_per_workday_usd"] == pytest.approx(155.0 / 22)
    assert row["budget_used_pct"] == pytest.approx(31.0)


def test_month_end_forecast_scales_by_elapsed_working_days() -> None:
    # 1-2 Oct 2026 are Thu-Fri: two working days elapsed, $20 spent.
    paced = budget_pace(daily(date(2026, 10, 1), date(2026, 10, 2), 10.0), BUDGET_USD)
    assert month_end_forecast(paced, date(2026, 10, 2)) == pytest.approx(10.0 * 22)


def test_workday_rolling_mean_skips_weekends() -> None:
    # Thu 1 Oct to Mon 5 Oct 2026; the weekend spend must not enter the mean.
    df = pl.DataFrame(
        {
            "day": [date(2026, 10, d) for d in range(1, 6)],
            "cost_usd": [2.0, 4.0, 100.0, 100.0, 6.0],
        }
    )
    out = workday_rolling_mean(budget_pace(df, BUDGET_USD), window=2)
    assert out.rows() == [
        (date(2026, 10, 1), 2.0),
        (date(2026, 10, 2), 3.0),
        (date(2026, 10, 5), 5.0),
    ]


def test_month_progress_overlays_months_in_percent() -> None:
    paced = budget_pace(daily(date(2026, 10, 1), date(2026, 10, 31), 5.0), BUDGET_USD)
    out = month_progress(paced, BUDGET_USD)
    assert out.row(0) == (date(2026, 10, 1), 0, 0.0, 0.0)
    last = out.row(-1, named=True)
    assert last["workday_no"] == 22
    assert last["elapsed_pct"] == pytest.approx(100.0)
    assert last["budget_used_pct"] == pytest.approx(31.0)
    # Fri 2 Oct is working day 2 and absorbs the weekend: 4 days x $5 = 4 % of budget.
    fri = out.filter(pl.col("workday_no") == 2).row(0, named=True)
    assert fri["budget_used_pct"] == pytest.approx(4.0)


def test_monthly_summary_averages_partial_month_over_elapsed_working_days() -> None:
    # 1-12 Oct 2026 holds 8 working days.
    paced = budget_pace(daily(date(2026, 10, 1), date(2026, 10, 12), 8.0), BUDGET_USD)
    row = monthly_summary(paced, BUDGET_USD).row(0, named=True)
    assert row["workdays"] == 22
    assert row["workdays_elapsed"] == 8
    assert row["avg_per_workday_usd"] == pytest.approx(96.0 / 8)
