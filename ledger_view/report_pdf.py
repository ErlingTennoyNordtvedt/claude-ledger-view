"""Render the monthly report as a one-page A4 PDF.

    python -m ledger_view.report_pdf --month 2026-09 --author "Your Name"
"""

import argparse
import os
import textwrap
from collections.abc import Sequence
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Final
from zoneinfo import ZoneInfo

import matplotlib as mpl
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch
from matplotlib.ticker import FuncFormatter

from ledger_view.ledger import (
    OTHER,
    OTHER_COLOR,
    PALETTE,
    TZ,
    load_hourly,
    month_days,
    open_ledger,
)
from ledger_view.report import (
    MonthReport,
    build_month_report,
    headline,
    load_labels,
    parse_month,
    previous_month,
    usd,
)

A4: Final = (8.27, 11.69)
INK: Final = "#1f1e1c"
INK_SECONDARY: Final = "#52514e"
INK_MUTED: Final = "#8a8983"
RULE: Final = "#e6e5e0"
TILE: Final = "#f5f4f0"
ACCENT: Final = PALETTE[0]
ACCENT_LIGHT: Final = "#86b6ef"
OVER: Final = "#e34948"
FONT: Final = "DejaVu Sans"
# Figure height per project row; seven rows fill the space above the bottom charts.
PROJECT_ROW: Final = 0.0345
CHART_BOTTOM: Final = 0.105


def tick_usd(value: float, _pos: int | None = None) -> str:
    return f"\\${value:,.0f}"


def style_axes(ax: Axes) -> None:
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(INK_MUTED)
    ax.tick_params(colors=INK_SECONDARY, labelsize=8, length=0)
    ax.grid(axis="y", color=RULE, lw=0.8)
    ax.set_axisbelow(True)
    ax.yaxis.set_major_formatter(FuncFormatter(tick_usd))


def section_title(fig: Figure, y: float, title: str, subtitle: str) -> None:
    fig.text(0.07, y, title, fontsize=12, fontweight="bold", color=INK, parse_math=False)
    fig.text(0.07, y - 0.018, subtitle, fontsize=8.5, color=INK_MUTED, parse_math=False)


def draw_header(fig: Figure, report: MonthReport, author: str | None, generated: date) -> None:
    fig.text(0.07, 0.95, "Claude usage report", fontsize=11, color=INK_MUTED)
    fig.text(0.07, 0.918, f"{report.month:%B %Y}", fontsize=24, fontweight="bold", color=INK)
    by = f"Prepared by {author}  ·  " if author else ""
    status = "" if report.complete else f"  ·  figures up to {report.as_of:%-d %B}"
    fig.text(
        0.93, 0.95, f"{by}{generated:%-d %B %Y}{status}", ha="right", fontsize=8.5,
        color=INK_MUTED, parse_math=False,
    )
    wrapped = textwrap.fill(headline(report), 82)
    fig.text(0.07, 0.885, wrapped, fontsize=11.5, color=INK, va="top", linespacing=1.5,
             parse_math=False)


def draw_tiles(fig: Figure, report: MonthReport) -> None:
    over = report.remaining < 0
    tiles = (
        ("Spent", usd(report.spent), f"{report.budget_used:.0%} of {usd(report.budget)} budget"),
        (
            "Over budget" if over else "Left in budget",
            usd(abs(report.remaining)),
            "needs attention" if over else "not spent",
        ),
        ("Projects", f"{report.project_count}", "worked on with Claude"),
        ("Per working day", usd(report.avg_per_workday), f"over {report.workdays_elapsed} working days"),
    )
    width, gap, left, bottom, height = 0.205, 0.0133, 0.07, 0.725, 0.09
    for i, (label, value, note) in enumerate(tiles):
        x = left + i * (width + gap)
        fig.patches.append(
            FancyBboxPatch(
                (x, bottom), width, height, boxstyle="round,pad=0,rounding_size=0.008",
                transform=fig.transFigure, facecolor=TILE, edgecolor="none",
            )
        )
        warn = over and i == 1
        fig.text(x + 0.015, bottom + height - 0.02, label, fontsize=8.5, color=INK_SECONDARY)
        fig.text(x + 0.015, bottom + 0.032, value, fontsize=20, fontweight="bold",
                 color=OVER if warn else INK, parse_math=False)
        fig.text(x + 0.015, bottom + 0.013, note, fontsize=7.5, color=INK_MUTED, parse_math=False)


def draw_projects(fig: Figure, report: MonthReport) -> float:
    """Draw one row per project and return the figure y where the list ends."""
    section_title(fig, 0.685, "Where the money went", "Spend per project this month")
    rows = max(len(report.projects), 1)
    # The axes grow with the row count, so bars keep one thickness and nothing is left empty.
    height = PROJECT_ROW * (rows + 0.4)
    ax = fig.add_axes((0.07, 0.65 - height, 0.86, height))
    ax.set_axis_off()
    if not report.projects:
        ax.text(0, 0.5, "No usage this month.", color=INK_MUTED, fontsize=10)
        return 0.65 - height
    biggest = report.projects[0].cost_usd
    ax.set_xlim(0, biggest * 1.18)
    ax.set_ylim(rows - 0.35, -0.75)
    for i, p in enumerate(report.projects):
        color = OTHER_COLOR if p.project == OTHER else ACCENT
        ax.barh(i + 0.12, p.cost_usd, height=0.34, color=color, left=0)
        name = ax.text(0, i - 0.3, p.name, fontsize=9.5, fontweight="bold", color=INK,
                       va="center", parse_math=False)
        if p.description:
            # Placed just right of the name, whatever its rendered width.
            ax.annotate(p.description, xy=(1, 0.5), xycoords=name, xytext=(8, 0),
                        textcoords="offset points", fontsize=8.5, color=INK_MUTED,
                        va="center", parse_math=False)
        ax.text(p.cost_usd + biggest * 0.015, i + 0.12, f"{usd(p.cost_usd)}  ·  {p.share:.0%}",
                fontsize=9, color=INK_SECONDARY, va="center", parse_math=False)
    return 0.65 - height


def draw_pace(fig: Figure, report: MonthReport, top: float) -> None:
    section_title(fig, top, "Spending through the month",
                  "Running total against an even spread of the budget")
    ax = fig.add_axes((0.09, CHART_BOTTOM, 0.39, top - 0.045 - CHART_BOTTOM))
    style_axes(ax)
    days = [p.day.day for p in report.cumulative]
    month_end = month_days(report.month.year, report.month.month)[-1]
    ax.axhline(report.budget, color=INK_MUTED, lw=1)
    ax.plot(days, [p.cum_spend for p in report.cumulative], color=ACCENT, lw=2, label="Spent")
    ax.plot(days, [p.cum_budget for p in report.cumulative], color=INK_MUTED, lw=2, ls="--",
            label="Even pace")
    ax.legend(loc="upper left", frameon=False, fontsize=8, labelcolor=INK_SECONDARY)
    last = report.cumulative[-1] if report.cumulative else None
    if last is not None:
        ax.annotate(f"spent {usd(last.cum_spend)}", (last.day.day, last.cum_spend), xytext=(4, 6),
                    textcoords="offset points", fontsize=8, color=INK_SECONDARY, ha="right",
                    va="bottom", annotation_clip=False)
    top = max([report.budget, *(p.cum_spend for p in report.cumulative)])
    ax.set_ylim(0, top * 1.25)  # headroom for the legend
    ticks = [1, 8, 15, 22, month_end.day]
    ax.set_xlim(1, month_end.day)
    ax.tick_params(axis="x", pad=6)
    ax.set_xticks(ticks, [f"{d} {report.month:%b}" for d in ticks])


def draw_history(fig: Figure, report: MonthReport, top: float) -> None:
    fig.text(0.54, top, "Month by month", fontsize=12, fontweight="bold", color=INK)
    fig.text(0.54, top - 0.018, f"Monthly spend against the {usd(report.budget)} budget", fontsize=8.5,
             color=INK_MUTED, parse_math=False)
    ax = fig.add_axes((0.56, CHART_BOTTOM, 0.37, top - 0.045 - CHART_BOTTOM))
    style_axes(ax)
    months = report.recent_months
    xs = range(len(months))
    colors = [ACCENT if m.month == report.month else ACCENT_LIGHT for m in months]
    bars = ax.bar(xs, [m.spent for m in months], width=0.62, color=colors)
    ax.bar_label(bars, [usd(m.spent).replace("$", "\\$") for m in months], padding=3,
                 fontsize=7.5, color=INK_SECONDARY)
    ax.axhline(report.budget, color=INK_MUTED, lw=1.5, ls="--")
    ax.annotate("budget", (len(months) - 0.69, report.budget), xytext=(0, 3),
                textcoords="offset points", fontsize=8, color=INK_MUTED, ha="right",
                va="bottom", annotation_clip=False)
    ax.set_xticks(list(xs), [f"{m.month:%b}" for m in months])
    top = max([report.budget, *(m.spent for m in months)])
    ax.set_ylim(0, top * 1.15)


def draw_footer(fig: Figure, report: MonthReport) -> None:
    fig.add_artist(Line2D([0.07, 0.93], [0.06, 0.06], transform=fig.transFigure, color=RULE, lw=1))
    note = (
        f"The budget is {usd(report.budget)} a month, spread evenly over working days "
        "(Monday to Friday, excluding Norwegian public holidays). Amounts are in US dollars "
        "and are what Claude usage was billed. Source: claude-box usage ledger."
    )
    fig.text(0.07, 0.048, textwrap.fill(note, 130), fontsize=7, color=INK_MUTED, va="top",
             linespacing=1.4, parse_math=False)


def render(report: MonthReport, generated: date, author: str | None = None) -> Figure:
    fig = Figure(figsize=A4, facecolor="white")
    fig.set_layout_engine("none")
    draw_header(fig, report, author, generated)
    draw_tiles(fig, report)
    top = draw_projects(fig, report) - 0.05
    draw_pace(fig, report, top)
    draw_history(fig, report, top)
    draw_footer(fig, report)
    return fig


def parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--db", type=Path,
                        default=Path(os.environ.get("LEDGER_DB", "data/ledger.sqlite")))
    parser.add_argument("--month", type=parse_month, help="YYYY-MM, default last full month")
    parser.add_argument("--labels", type=Path, default=Path("data/project_labels.toml"))
    parser.add_argument("--author", help="name shown in the header")
    parser.add_argument("--out-dir", type=Path, default=Path("build"))
    parser.add_argument("--out", type=Path, help="output file, overrides --out-dir")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    today = datetime.now(UTC).astimezone(ZoneInfo(TZ)).date()
    month: date = args.month or previous_month(today)
    with open_ledger(args.db) as conn:
        hourly = load_hourly(conn, TZ)
    report = build_month_report(hourly, month, today, load_labels(args.labels))
    out: Path = args.out or args.out_dir / f"claude-spend-{month:%Y-%m}.pdf"
    out.parent.mkdir(parents=True, exist_ok=True)
    with mpl.rc_context({"font.family": FONT}):
        render(report, today, args.author).savefig(out, format="pdf")
    print(out)


if __name__ == "__main__":
    main()
