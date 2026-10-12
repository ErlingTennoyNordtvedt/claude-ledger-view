from datetime import UTC, date, datetime
from zoneinfo import ZoneInfo

import polars as pl
import pytest

from ledger_view.ledger import (
    OTHER,
    OTHER_COLOR,
    PALETTE,
    daily_spend,
    group_small,
    month_start,
    project_colors,
    spend_by_model,
    spend_by_project,
    with_local_time,
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
