from datetime import UTC, date, datetime

import polars as pl
import pytest

from ledger_view.ledger import OTHER, with_local_time
from ledger_view.report import (
    MonthReport,
    ProjectSpend,
    build_month_report,
    headline,
    label_for,
    parse_labels,
    parse_month,
    previous_month,
)

OSLO = "Europe/Oslo"


def hourly(rows: list[tuple[datetime, str, float]]) -> pl.DataFrame:
    return with_local_time(
        pl.DataFrame(
            {
                "hour": [int(t.timestamp()) // 3600 for t, _, _ in rows],
                "project": [p for _, p, _ in rows],
                "cost_usd": [c for _, _, c in rows],
            },
            schema={"hour": pl.Int64, "project": pl.String, "cost_usd": pl.Float64},
        ),
        OSLO,
    )


def report(spent: float, *, complete: bool = True, top: str = "app") -> MonthReport:
    return MonthReport(
        month=date(2026, 9, 1),
        as_of=date(2026, 9, 30),
        complete=complete,
        budget=500.0,
        spent=spent,
        workdays=22,
        workdays_elapsed=22,
        project_count=1,
        projects=(ProjectSpend(top, top.title(), "", spent, 1.0),) if spent else (),
        cumulative=(),
        recent_months=(),
    )


def test_parse_labels_reads_name_and_optional_description() -> None:
    labels = parse_labels('[a]\nname = "Alpha"\ndescription = "First"\n[b]\nname = "Beta"\n')
    assert label_for("a", labels) == ("Alpha", "First")
    assert label_for("b", labels) == ("Beta", "")


def test_unlabelled_project_falls_back_to_repo_name() -> None:
    assert label_for("my-repo", {}) == ("my-repo", "")
    assert label_for("backfill", {})[0] == "Imported history"


def test_parse_labels_rejects_missing_name() -> None:
    with pytest.raises(TypeError):
        parse_labels('[a]\ndescription = "no name"\n')


def test_previous_month_crosses_year() -> None:
    assert previous_month(date(2026, 1, 15)) == date(2025, 12, 1)
    assert previous_month(date(2026, 10, 1)) == date(2026, 9, 1)


def test_parse_month() -> None:
    assert parse_month("2026-09") == date(2026, 9, 1)
    for bad in ("2026", "2026-13", "sept"):
        with pytest.raises(ValueError, match="YYYY-MM"):
            parse_month(bad)


def test_build_month_report_counts_only_that_month_in_local_time() -> None:
    rows = hourly(
        [
            (datetime(2026, 8, 20, 10, tzinfo=UTC), "app", 50.0),
            # 22:30 UTC on 30 Sep is already 1 Oct in Oslo.
            (datetime(2026, 9, 30, 22, 30, tzinfo=UTC), "app", 7.0),
            (datetime(2026, 9, 2, 10, tzinfo=UTC), "app", 300.0),
            (datetime(2026, 9, 3, 10, tzinfo=UTC), "web", 100.0),
            (datetime(2026, 9, 4, 10, tzinfo=UTC), "backfill", 20.0),
        ]
    )
    r = build_month_report(rows, date(2026, 9, 1), date(2026, 10, 20), {})
    assert r.complete
    assert r.spent == pytest.approx(420.0)
    assert r.remaining == pytest.approx(80.0)
    assert r.budget_used == pytest.approx(0.84)
    assert r.project_count == 2  # backfill is not a project
    assert [p.project for p in r.projects] == ["app", "web", "backfill"]
    assert sum(p.share for p in r.projects) == pytest.approx(1.0)
    assert r.cumulative[-1].cum_spend == pytest.approx(420.0)
    assert r.cumulative[-1].cum_budget == pytest.approx(500.0)
    # History starts at the first month with data, not six months back.
    assert [m.month for m in r.recent_months] == [date(2026, 8, 1), date(2026, 9, 1)]
    assert [m.spent for m in r.recent_months] == pytest.approx([50.0, 420.0])


def test_build_month_report_part_month_averages_over_elapsed_workdays() -> None:
    rows = hourly([(datetime(2026, 10, 1, 10, tzinfo=UTC), "app", 100.0)])
    r = build_month_report(rows, date(2026, 10, 1), date(2026, 10, 9), {})
    assert not r.complete
    assert r.as_of == date(2026, 10, 9)
    assert r.workdays_elapsed == 7  # Thu 1 to Fri 9 October
    assert r.avg_per_workday == pytest.approx(100.0 / 7)
    assert r.cumulative[-1].day == date(2026, 10, 9)


def test_build_month_report_folds_small_projects() -> None:
    big = [(datetime(2026, 9, 2, 10, tzinfo=UTC), f"p{i}", 100.0) for i in range(8)]
    tiny = [(datetime(2026, 9, 2, 10, tzinfo=UTC), "tiny", 1.0)]
    r = build_month_report(hourly(big + tiny), date(2026, 9, 1), date(2026, 10, 1), {})
    assert len(r.projects) == 7
    assert r.projects[-1].project == OTHER
    assert r.projects[-1].name == "Other"


def test_headline_under_budget() -> None:
    text = headline(report(412.0))
    assert "$412 of the $500 budget" in text
    assert "leaving $88 unspent" in text
    assert "largest share went to App (100%)" in text


def test_headline_over_budget() -> None:
    assert "$60 (12%) over the $500 budget" in headline(report(560.0))


def test_headline_part_month() -> None:
    assert headline(report(100.0, complete=False)).startswith("So far in September 2026")


def test_headline_no_spend() -> None:
    assert headline(report(0.0)) == "No Claude usage was recorded in September 2026."


def test_headline_skips_imported_history_as_top_project() -> None:
    assert "largest share" not in headline(report(100.0, top="backfill"))
