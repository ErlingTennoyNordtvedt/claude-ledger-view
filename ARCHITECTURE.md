# Architecture

## Purpose
A Jupyter notebook that shows Claude spend from the claude-box ledger
(`data/ledger.sqlite`, a symlink to `~/.local/share/claude-box/ledger.sqlite`, which keeps being
written to). It shows spend per project for this month and all time, daily spend, and spend per model.

## Decisions

### Data source: the ledger's SQLite views, opened read-only
- The ledger schema is owned by claude-box (goose migrations). This repo never migrates or writes
  it, and opens it with `mode=ro`.
- Spend over time comes from the `hourly_costs` view, not from session start times. A session can
  run across a month boundary, and `hourly_costs` puts each cost in the hour it was spent.
- Per-session detail comes from `session_costs`.
- Month and day boundaries are in local time (`Europe/Oslo`). The DB stores UTC epoch hours.

### Analysis: polars
- Requested by the user. It is typed, fast, and reads from a stdlib `sqlite3` connection through
  `pl.read_database`, with no extra driver.
- Aggregation code lives in `ledger_view/ledger.py` as pure frame-to-frame functions, tested in
  `tests/` without I/O. The notebook stays thin.
- pandas is only a transitive dependency that seaborn converts to. It is not used directly.

### Plotting: seaborn on matplotlib
- Requested by the user. seaborn has no pie chart, so the pies use matplotlib `ax.pie` styled with
  the seaborn theme. Daily spend uses `seaborn.objects` stacked bars.
- Colours use a fixed, colourblind-validated categorical order. Each project keeps its colour
  across charts, assigned by all-time spend. Pies have at most 6 slices, and slices under 2% fold
  into "other".

### Budget model
- Monthly budget (`BUDGET_USD`, $500) spread evenly over the month's working days: Mon–Fri minus
  Norwegian public holidays. Weekend and holiday spend counts towards the month but earns no
  budget.
- Holidays are computed in `ledger_view/ledger.py` (anonymous Gregorian Easter algorithm plus
  fixed dates). Rejected: the `holidays` package, an extra dependency for twelve dates a year.
- Budget views use all spend. Per-project views leave out `backfill` (`EXCLUDED_PROJECTS`): it is
  real money but has no real project, and it would dominate the pies.
- Month-end forecast is a linear extrapolation per elapsed working day. It is simple, and it is
  honest about being a rough guess.

### Environment: Nix flake devShell
- `flake.nix` provides Python with jupyterlab, polars, seaborn, matplotlib, pytest, mypy, and ruff
  from nixpkgs. Rejected: uv/pip venv, which would be a second package manager next to Nix.
