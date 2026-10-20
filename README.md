# ledger-view

A Jupyter notebook that shows Claude spend from the claude-box ledger: where the money went per
project this month and in total.

The notebook is [`ledger.ipynb`](ledger.ipynb). It shows:

- headline totals: all time, this month, number of sessions, latest data point
- **budget** ($500/month, spread over working days):
  - this month's cumulative spend vs the budget pace, with a month-end forecast
  - daily spend vs the daily budget, with a 5-working-day average and weekends shaded
  - months compared: % of budget used vs % of working days elapsed
  - a per-month summary table
- pie charts of spend per project, this month and all time
- spend per day, stacked by project
- spend per model, and the ten most expensive sessions

## Monthly PDF report

`make report` writes a one-page A4 PDF for a non-technical reader to `build/claude-spend-YYYY-MM.pdf`:
a plain-language summary, spend vs budget, where the money went per project, spending through the
month, and month-by-month spend.

```sh
make report                                  # last full month
make report MONTH=2026-09 AUTHOR="Your Name" # a chosen month, with your name in the header
make report-example                          # from data/ledger_example.sqlite
```

Project names in the ledger are repo names. To show plain-English names and a one-line description,
copy [`project_labels.example.toml`](project_labels.example.toml) to `data/project_labels.toml` and
fill it in. Projects without an entry keep their repo name.

The current month works too; it is marked "figures up to" today, and the averages use the working
days so far.

## Requirements

Nix with flakes enabled. Everything else (Python, JupyterLab, polars, seaborn, pytest, mypy,
ruff) comes from the dev shell in `flake.nix`.

## Data

The notebook reads `data/ledger.sqlite`, a symlink to the live ledger at
`~/.local/share/claude-box/ledger.sqlite`. It opens the file read-only, so it is safe to run
while claude-box is writing. Rerun the notebook to pick up new spend.

To point it at another file, set `LEDGER_DB`:

```sh
LEDGER_DB=data/ledger_example.sqlite make lab
```

`data/` is git-ignored.

## Usage

```sh
make lab          # JupyterLab on the live ledger
make example      # JupyterLab on data/ledger_example.sqlite
make run          # execute the notebook headless into build/
make check        # tests, mypy --strict, ruff
make help         # all targets
```

Or enter the shell yourself with `nix develop`.

## Layout

```
ledger.ipynb             the notebook: config, load, charts
ledger_view/ledger.py    loading from SQLite and pure polars aggregations
ledger_view/report.py    the numbers for the PDF report, pure
ledger_view/report_pdf.py  PDF rendering (matplotlib) and the report CLI
tests/test_ledger.py     unit tests for the aggregations
tests/test_report.py     unit tests for the report numbers and wording
project_labels.example.toml  example plain-English project names for the report
flake.nix                dev shell
ARCHITECTURE.md          design decisions
```

## Notes

- Spend over time comes from the ledger's `hourly_costs` view. A session that runs across midnight
  or a month boundary is split by the hour it was spent in.
- Days and months are in `Europe/Oslo` time (`TZ` in `ledger_view/ledger.py`).
- The budget is `BUDGET_USD` in `ledger_view/ledger.py` (overridable as `BUDGET` in the notebook's
  setup cell). Each working day gets budget ÷ working days in the month. Working days are Mon–Fri
  minus Norwegian public holidays, which are computed in code (Easter-based ones included).
- Budget views count all spend. Per-project views leave out `backfill` (`EXCLUDED_PROJECTS`),
  which is imported history with no real project.
- The forecast is spend so far ÷ working days elapsed × working days in the month.
