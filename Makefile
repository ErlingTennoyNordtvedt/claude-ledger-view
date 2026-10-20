NIX      := nix develop --command
NOTEBOOK := ledger.ipynb
EXAMPLE  := data/ledger_example.sqlite
OUT_DIR  := build
REPORT   := $(NIX) python -m ledger_view.report_pdf --out-dir $(OUT_DIR) \
            $(if $(MONTH),--month $(MONTH)) $(if $(AUTHOR),--author "$(AUTHOR)")

.PHONY: help lab example test typecheck lint check run run-example report report-example clean

help: ## Show this help
	@grep -E '^[a-z-]+:.*## ' $(MAKEFILE_LIST) | awk -F':.*## ' '{printf "  %-12s %s\n", $$1, $$2}'

lab: ## Open JupyterLab against the live ledger
	$(NIX) jupyter lab $(NOTEBOOK)

example: ## Open JupyterLab against the example ledger
	LEDGER_DB=$(EXAMPLE) $(NIX) jupyter lab $(NOTEBOOK)

test: ## Run the unit tests
	$(NIX) pytest -q

typecheck: ## Type-check with mypy --strict
	$(NIX) mypy --strict --ignore-missing-imports ledger_view tests

lint: ## Lint with ruff, including the S (bandit) rules
	$(NIX) ruff check --extend-select S ledger_view tests

check: test typecheck lint ## Run tests, type check and lint

run: ## Execute the notebook against the live ledger into build/
	$(NIX) jupyter nbconvert --to notebook --execute $(NOTEBOOK) --output-dir $(OUT_DIR)

run-example: ## Execute the notebook against the example ledger into build/
	LEDGER_DB=$(EXAMPLE) $(NIX) jupyter nbconvert --to notebook --execute $(NOTEBOOK) --output-dir $(OUT_DIR)

report: ## One-page PDF for your manager (MONTH=YYYY-MM AUTHOR="Name"; default last full month)
	$(REPORT)

report-example: ## The PDF report from the example ledger
	$(REPORT) --db $(EXAMPLE)

clean: ## Remove executed notebooks and caches
	rm -rf $(OUT_DIR) .pytest_cache .mypy_cache .ruff_cache
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
