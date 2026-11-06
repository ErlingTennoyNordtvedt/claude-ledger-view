"""Integration tests: the forecasters against statsmodels and NumPyro, and the backtest loop."""

from datetime import date, timedelta

import numpy as np
import pytest

from ledger_view.forecast import (
    BayesForecaster,
    BootstrapForecaster,
    EtsForecaster,
    Forecaster,
    History,
    LinearForecaster,
    ModelName,
    Paths,
    backtest,
    backtest_origins,
    backtest_summary,
    future_days,
    workday_mask,
)

pytestmark = pytest.mark.integration

# Working days spend lognormal(log 20, 0.4), off days nothing: mean per working day ~$21.7.
TRUE_WORKDAY_MEAN = 20.0 * float(np.exp(0.4**2 / 2))


def synthetic(start: date, n: int, seed: int = 0) -> History:
    days = tuple(start + timedelta(days=i) for i in range(n))
    mask = workday_mask(days)
    work = np.random.default_rng(seed).lognormal(np.log(20.0), 0.4, n)
    return History(days, np.where(mask, work, 0.0), mask)


def check_paths(paths: Paths, days: tuple[date, ...], n_paths: int) -> None:
    mask = workday_mask(days)
    assert paths.shape == (n_paths, len(days))
    assert np.all(paths >= 0)
    assert np.isfinite(paths).all()
    assert float(np.median(paths[:, mask].mean(axis=1))) == pytest.approx(
        TRUE_WORKDAY_MEAN, rel=0.25
    )


def test_ets_paths_against_statsmodels() -> None:
    h = synthetic(date(2026, 8, 1), 90).until(date(2026, 10, 15))
    days = future_days(date(2026, 10, 15))
    check_paths(EtsForecaster()(h, days, 1000, np.random.default_rng(1)), days, 1000)


def test_bayes_paths_against_numpyro() -> None:
    h = synthetic(date(2026, 8, 1), 90).until(date(2026, 10, 15))
    days = future_days(date(2026, 10, 15))
    paths = BayesForecaster(num_warmup=200, num_samples=200)(
        h, days, 1000, np.random.default_rng(1)
    )
    check_paths(paths, days, 1000)
    true_total = TRUE_WORKDAY_MEAN * int(workday_mask(days).sum())
    low, high = np.quantile(paths.sum(axis=1), [0.05, 0.95])
    assert low <= true_total <= high


def test_bayes_is_deterministic_for_a_seed() -> None:
    h = synthetic(date(2026, 8, 1), 60).until(date(2026, 9, 20))
    days = future_days(date(2026, 9, 20))
    model = BayesForecaster(num_warmup=50, num_samples=50)
    a = model(h, days, 20, np.random.default_rng(3))
    b = model(h, days, 20, np.random.default_rng(3))
    assert np.array_equal(a, b)


class Spy:
    """Records the last day each call could see and the first day it was asked about."""

    def __init__(self) -> None:
        self.calls: list[tuple[date, date]] = []

    def __call__(
        self, history: History, days: tuple[date, ...], n_paths: int, rng: np.random.Generator
    ) -> Paths:
        self.calls.append((history.days[-1], days[0]))
        return np.zeros((n_paths, len(days)))


def test_backtest_never_shows_a_model_the_future() -> None:
    h = synthetic(date(2026, 7, 1), 123)  # July to October 2026.
    spy = Spy()
    origins = backtest_origins(h)
    scores = backtest(h, {"linear": spy}, origins, 10, np.random.default_rng(0))
    assert len(spy.calls) == len(origins) == scores.height
    assert all(seen < first for seen, first in spy.calls)
    assert [seen for seen, _ in spy.calls] == list(origins)


def test_backtest_all_models_one_row_per_model_and_origin() -> None:
    h = synthetic(date(2026, 7, 1), 123)
    origins = backtest_origins(h, (10,))[-2:]
    models: dict[ModelName, Forecaster] = {
        "bootstrap": BootstrapForecaster(),
        "ets": EtsForecaster(),
        "bayes": BayesForecaster(num_warmup=100, num_samples=100),
        "linear": LinearForecaster(),
    }
    scores = backtest(h, models, origins, 200, np.random.default_rng(0))
    assert scores.height == len(models) * len(origins)
    for row in scores.iter_rows(named=True):
        month = row["month"]
        assert row["actual"] == pytest.approx(h.spent_in_month(month))
        assert row["p05"] <= row["median"] <= row["p95"]
        assert row["crps"] >= 0
    # The linear point forecast has no spread.
    linear = scores.filter(scores["model"] == "linear")
    assert (linear["p95"] - linear["p05"]).max() == pytest.approx(0.0)
    assert set(backtest_summary(scores)["model"]) == set(models)


def test_backtest_band_narrows_later_in_month() -> None:
    h = synthetic(date(2026, 7, 1), 123)
    origins = tuple(o for o in backtest_origins(h) if o.month == 10)
    scores = backtest(h, {"bootstrap": BootstrapForecaster()}, origins, 2000, np.random.default_rng(0))
    width = (scores["p95"] - scores["p05"]).to_list()
    assert width == sorted(width, reverse=True)
