"""The numbers behind the one-page monthly report, built from frames with no I/O.

`load_labels` is the only function that reads a file.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Final, NotRequired, TypedDict

import polars as pl
import tomllib

from ledger_view.ledger import (
    BUDGET_USD,
    EXCLUDED_PROJECTS,
    OTHER,
    budget_pace,
    daily_total,
    group_small,
    month_days,
    spend_by_project,
    working_days,
)

HISTORY_MONTHS: Final = 6
MAX_PROJECTS: Final = 7
# Rows that are not a project a reader would recognise as "where the money went".
UNNAMED_PROJECTS: Final = EXCLUDED_PROJECTS | {OTHER}
# Shown for projects the label file does not name.
DEFAULT_LABELS: Final[Mapping[str, "ProjectLabel"]] = {
    "backfill": {"name": "Imported history", "description": "Usage recorded before tracking"},
    OTHER: {"name": "Other", "description": "Smaller projects combined"},
}


class ProjectLabel(TypedDict):
    name: str
    description: NotRequired[str]


@dataclass(frozen=True)
class ProjectSpend:
    project: str
    name: str
    description: str
    cost_usd: float
    share: float


@dataclass(frozen=True)
class DayPoint:
    day: date
    cum_spend: float
    cum_budget: float


@dataclass(frozen=True)
class MonthSpend:
    month: date
    spent: float


@dataclass(frozen=True)
class MonthReport:
    month: date
    as_of: date
    complete: bool
    budget: float
    spent: float
    workdays: int
    workdays_elapsed: int
    project_count: int
    projects: tuple[ProjectSpend, ...]
    cumulative: tuple[DayPoint, ...]
    recent_months: tuple[MonthSpend, ...]

    @property
    def remaining(self) -> float:
        return self.budget - self.spent

    @property
    def budget_used(self) -> float:
        return self.spent / self.budget

    @property
    def avg_per_workday(self) -> float:
        """Spend per working day so far; all spend counts, weekend and holiday spend too."""
        return self.spent / max(self.workdays_elapsed, 1)


def parse_labels(text: str) -> dict[str, ProjectLabel]:
    """Read `[project] name = "..." description = "..."` tables from TOML text."""
    labels: dict[str, ProjectLabel] = {}
    for project, table in tomllib.loads(text).items():
        if not isinstance(table, dict) or not isinstance(table.get("name"), str):
            raise TypeError(f"label for {project!r} needs a string 'name'")
        label: ProjectLabel = {"name": table["name"]}
        description = table.get("description")
        if description is not None:
            if not isinstance(description, str):
                raise TypeError(f"description for {project!r} must be a string")
            label["description"] = description
        labels[project] = label
    return labels


def load_labels(path: Path) -> dict[str, ProjectLabel]:
    """Labels from `path`, or none if the file does not exist."""
    if not path.exists():
        return {}
    return parse_labels(path.read_text(encoding="utf-8"))


def label_for(project: str, labels: Mapping[str, ProjectLabel]) -> tuple[str, str]:
    """Display name and description; the repo name with no description if unlabelled."""
    label = labels.get(project) or DEFAULT_LABELS.get(project)
    if label is None:
        return project, ""
    return label["name"], label.get("description", "")


def add_months(month: date, n: int) -> date:
    index = month.year * 12 + month.month - 1 + n
    return date(index // 12, index % 12 + 1, 1)


def previous_month(today: date) -> date:
    """First day of the month before `today`'s."""
    return add_months(today.replace(day=1), -1)


def parse_month(text: str) -> date:
    """`YYYY-MM` to the first day of that month."""
    try:
        year, month = (int(part) for part in text.split("-"))
        return date(year, month, 1)
    except ValueError as err:
        raise ValueError(f"expected a month as YYYY-MM, got {text!r}") from err


def month_projects(
    hourly: pl.DataFrame, month: date, labels: Mapping[str, ProjectLabel]
) -> tuple[ProjectSpend, ...]:
    """Spend per project in `month`, largest first, small ones folded into "other".

    Everything is included, `backfill` too, so the rows add up to the month's spend.
    """
    end = month_days(month.year, month.month)[-1]
    rows = hourly.filter(pl.col("ts").dt.date().is_between(month, end))
    spend = group_small(spend_by_project(rows, exclude=frozenset()), max_slices=MAX_PROJECTS)
    total = float(spend["cost_usd"].sum())
    return tuple(
        ProjectSpend(project, *label_for(project, labels), cost, cost / total)
        for project, cost in spend.iter_rows()
    )


def build_month_report(
    hourly: pl.DataFrame,
    month: date,
    today: date,
    labels: Mapping[str, ProjectLabel],
    budget: float = BUDGET_USD,
) -> MonthReport:
    """Everything the report shows for `month`, with spend counted up to `today`."""
    end = month_days(month.year, month.month)[-1]
    as_of = min(end, today)
    first = add_months(month, 1 - HISTORY_MONTHS)
    paced = budget_pace(daily_total(hourly, first, end), budget).filter(pl.col("day") <= as_of)
    current = paced.filter(pl.col("month") == month)
    # Months before the ledger's first entry are not shown as $0.
    started = hourly["ts"].min()
    since = first if not isinstance(started, datetime) else max(first, started.date().replace(day=1))
    per_month = (
        paced.filter(pl.col("month") >= since)
        .group_by("month")
        .agg(pl.col("cost_usd").sum())
        .sort("month")
    )
    in_month = hourly.filter(pl.col("ts").dt.date().is_between(month, as_of))
    real = spend_by_project(in_month, exclude=EXCLUDED_PROJECTS)
    return MonthReport(
        month=month,
        as_of=as_of,
        complete=today > end,
        budget=budget,
        spent=float(current["cost_usd"].sum()),
        workdays=len(working_days(month.year, month.month)),
        workdays_elapsed=sum(1 for d in working_days(month.year, month.month) if d <= as_of),
        project_count=real.height,
        projects=month_projects(in_month, month, labels),
        cumulative=tuple(
            DayPoint(day, spend, pace)
            for day, spend, pace in current.select("day", "cum_spend", "cum_budget").iter_rows()
        ),
        recent_months=tuple(MonthSpend(m, spent) for m, spent in per_month.iter_rows()),
    )


def usd(amount: float) -> str:
    return f"${amount:,.0f}"


def headline(report: MonthReport) -> str:
    """One plain-language sentence or two summing up the month."""
    period = f"{report.month:%B %Y}"
    if report.spent == 0:
        return f"No Claude usage was recorded in {period}."
    if report.spent > report.budget:
        over = report.spent - report.budget
        main = (
            f"We spent {usd(report.spent)} in {period}, {usd(over)} "
            f"({over / report.budget:.0%}) over the {usd(report.budget)} budget."
        )
    elif report.complete:
        main = (
            f"We used {usd(report.spent)} of the {usd(report.budget)} budget in {period} "
            f"({report.budget_used:.0%}), leaving {usd(report.remaining)} unspent."
        )
    else:
        main = (
            f"So far in {period} we have used {usd(report.spent)} of the "
            f"{usd(report.budget)} budget ({report.budget_used:.0%})."
        )
    named = [p for p in report.projects if p.project not in UNNAMED_PROJECTS]
    if not named:
        return main
    return f"{main} The largest share went to {named[0].name} ({named[0].share:.0%})."
