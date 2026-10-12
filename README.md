# ledger-view

A Jupyter notebook that shows Claude spend from the claude-box ledger: where the money went per
project this month and in total.

The notebook is [`ledger.ipynb`](ledger.ipynb). It shows:

- headline totals: all time, this month, number of sessions, latest data point
- pie charts of spend per project, this month and all time
- spend per day, stacked by project
- spend per model, and the ten most expensive sessions

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
tests/test_ledger.py     unit tests for the aggregations
flake.nix                dev shell
ARCHITECTURE.md          design decisions
```

## Notes

- Spend over time comes from the ledger's `hourly_costs` view. A session that runs across midnight
  or a month boundary is split by the hour it was spent in.
- Days and months are in `Europe/Oslo` time (`TZ` in `ledger_view/ledger.py`).
