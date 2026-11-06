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
- ~~Month-end forecast is a linear extrapolation per elapsed working day.~~ **Superseded** by the
  probabilistic forecasts below: one number with no uncertainty, swung hard by early spikes. It
  survives as `month_end_forecast` and as the `linear` row in the backtest.

### Forecasting: simulated paths, three models
- `ledger_view/forecast.py`. Every model returns sample paths of daily spend for the days left in
  the month, shape `(n_paths, n_days)`. Fans, the month-end interval, P(over budget) and backtest
  scores are computed from the paths, so the month-end total is a sum per path. Summing per-day
  quantiles would get the interval wrong.
- Models:
  - **Bootstrap**: moving-block resampling of recent working days, with off days resampled on
    their own. No trend. It is the baseline the others must beat.
  - **ETS**: statsmodels `ETSModel`, additive error and damped additive trend on `log1p` of
    working-day spend, simulated forward. Fitting on working days only removes the weekly pattern
    and holidays without modelling them.
  - **Bayesian**: NumPyro NUTS. Hurdle-lognormal spend (P(spend > 0) per day kind), a damped
    local linear trend on the log level, and an off-day offset. Posterior draws are rolled forward
    in NumPy. This is the headline forecast in the notebook; the bootstrap stands in when history
    is too short.
- The notebook forecasts from yesterday's close. Today is partial and would read as a low day.
- Randomness is a `numpy.random.Generator` passed in, seeded in the notebook, so the bands are
  stable between runs on the same data.
- A backtest scores every model at the 5th, 10th and 15th working day of each completed month
  (CRPS and 90% coverage), seeing only data up to the origin.
- Dependencies: statsmodels (BSD-3-Clause), patsy (BSD-2-Clause), numpyro (Apache-2.0), jax and
  jaxlib (Apache-2.0), ml-dtypes (Apache-2.0), opt-einsum (MIT), multipledispatch (BSD-3-Clause),
  six (MIT) and tqdm (MIT). Licenses were read from nixpkgs `meta.license`. numpyro is built with
  its tests off, because its test inputs pull in tensorflow, which nixpkgs marks broken on Python
  3.14.
- Rejected: SARIMAX (ETS with more tuning and nothing gained on short data); Prophet (poorly
  calibrated intervals, heavy Stan dependency); quantile gradient boosting (needs much more data,
  gives no paths, so no month-end sum); conformal prediction (needs calibration months we do not
  have yet, so revisit later); pretrained foundation models (PyTorch-sized dependencies for one
  short series, and hard to explain). PyMC was not chosen over NumPyro, which is lighter and
  faster for a model this size.

### Monthly PDF report: matplotlib, one A4 page
- A one-page summary for a non-technical manager, separate from the notebook. `ledger_view/report.py`
  builds a frozen `MonthReport` from the frames (pure, tested); `ledger_view/report_pdf.py` draws it
  on a matplotlib `Figure` and saves it as PDF. Run with `make report`.
- matplotlib writes the PDF itself, so there is no new dependency. Rejected: reportlab and
  weasyprint, extra packages for one page of charts and text.
- One month per report, default the last completed month so the numbers are final. USD only, like
  the ledger and budget.
- Project names come from an optional TOML label file (`data/project_labels.toml`, read with
  stdlib `tomllib`), so repo names can be shown in plain English. It lives in git-ignored `data/`.
- The per-project bars include `backfill` (as "Imported history"), so they add up to the month's
  spend. The project count and the headline's "largest share" leave it out.
- Bars rather than pies for per-project spend: easier for a lay reader to compare.

### Notebook in git: committed stripped, via a git clean filter
- `ledger.ipynb` is committed without outputs, execution counts, cell metadata or
  `language_info`; only `kernelspec` is kept. Outputs are embedded PNGs (≈550 KB) and
  `language_info` holds the local Python version, which would churn diffs.
- A git `clean` filter (`.gitattributes`, configured by `make setup`) runs
  `ledger_view/strip_notebook.py` on `git add`, so the working copy keeps its plots. The filter is
  `required`, so a broken filter blocks the add instead of committing outputs.
- The stripper is stdlib only, so the filter runs with plain `python3` outside the dev shell. It
  writes JSON the way nbformat does (indent 1, sorted keys), so it is idempotent.
- Rejected: `nbstripout`, an extra package for ~50 lines of JSON handling; `nbconvert
  --clear-output`, which needs the dev shell and is a manual step that is easy to forget.
- Supersedes: ignoring all `*.ipynb` in git (commit 1989a59). Other notebooks stay ignored;
  `.gitignore` re-includes only `/ledger.ipynb`.

### Environment: Nix flake devShell
- `flake.nix` provides Python with jupyterlab, polars, seaborn, matplotlib, pytest, mypy, and ruff
  from nixpkgs. Rejected: uv/pip venv, which would be a second package manager next to Nix.
