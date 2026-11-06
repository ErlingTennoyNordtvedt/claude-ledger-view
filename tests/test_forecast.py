"""Unit tests for the pure parts of `ledger_view.forecast`. No model fitting here."""

from datetime import date, timedelta

import numpy as np
import polars as pl
import pytest

from ledger_view.forecast import (
    BayesForecaster,
    BootstrapForecaster,
    EtsForecaster,
    History,
    InsufficientHistory,
    LinearForecaster,
    assemble,
    backtest_origins,
    covered,
    crps,
    draw_blocks,
    draw_pool,
    fan,
    future_days,
    history_from_paced,
    hurdle_spend,
    pinball,
    prob_over,
    step_levels,
    workday_mask,
)
from ledger_view.ledger import BUDGET_USD, budget_pace


def days_from(start: date, n: int) -> tuple[date, ...]:
    return tuple(start + timedelta(days=i) for i in range(n))


def history(start: date, n: int, work: float, off: float) -> History:
    days = days_from(start, n)
    mask = workday_mask(days)
    return History(days, np.where(mask, work, off), mask)


def rng() -> np.random.Generator:
    return np.random.default_rng(7)


def test_history_from_paced_keeps_order_flags_and_cutoff() -> None:
    daily = pl.DataFrame(
        {"day": list(days_from(date(2026, 10, 1), 5))[::-1], "cost_usd": [5.0, 4.0, 3.0, 2.0, 1.0]}
    )
    h = history_from_paced(budget_pace(daily, BUDGET_USD), date(2026, 10, 4))
    assert h.days == days_from(date(2026, 10, 1), 4)
    assert h.spend.tolist() == [1.0, 2.0, 3.0, 4.0]
    # 1-2 Oct 2026 are Thu-Fri, 3-4 Oct the weekend.
    assert h.workday.tolist() == [True, True, False, False]


def test_history_until_and_since_slice_by_date() -> None:
    h = history(date(2026, 10, 1), 10, 1.0, 0.0)
    assert h.until(date(2026, 10, 3)).days == days_from(date(2026, 10, 1), 3)
    assert h.since(date(2026, 10, 8)).days == days_from(date(2026, 10, 8), 3)


def test_spent_in_month_only_counts_that_month() -> None:
    h = History(
        (date(2026, 9, 30), date(2026, 10, 1)), np.array([5.0, 7.0]), np.array([True, True])
    )
    assert h.spent_in_month(date(2026, 10, 1)) == 7.0


@pytest.mark.parametrize(
    ("today", "first", "count"),
    [
        (date(2026, 10, 28), date(2026, 10, 29), 3),
        (date(2026, 12, 30), date(2026, 12, 31), 1),
        (date(2026, 10, 31), None, 0),
    ],
)
def test_future_days_is_rest_of_month(today: date, first: date | None, count: int) -> None:
    days = future_days(today)
    assert len(days) == count
    assert (days[0] if days else None) == first


def test_workday_mask_skips_weekends_and_holidays() -> None:
    # 24 Dec 2026 Thu (workday), 25 Dec Christmas, 26 Dec Saturday.
    assert workday_mask([date(2026, 12, 24), date(2026, 12, 25), date(2026, 12, 26)]).tolist() == [
        True,
        False,
        False,
    ]


def test_draw_pool_empty_is_zero() -> None:
    assert draw_pool(np.array([]), 4, 3, rng()).tolist() == np.zeros((4, 3)).tolist()


def test_draw_pool_only_returns_pool_values() -> None:
    assert set(draw_pool(np.array([1.0, 2.0]), 50, 5, rng()).ravel()) <= {1.0, 2.0}


def test_draw_blocks_are_consecutive_runs() -> None:
    pool = np.arange(20, dtype=np.float64)
    out = draw_blocks(pool, 30, 6, 3, rng())
    assert out.shape == (30, 6)
    blocks = out.reshape(30, 2, 3)
    assert np.all(np.diff(blocks, axis=2) == 1)


def test_draw_blocks_shrinks_block_to_pool() -> None:
    out = draw_blocks(np.array([4.0, 5.0]), 3, 5, 10, rng())
    assert out.shape == (3, 5)


def test_draw_blocks_needs_a_pool() -> None:
    with pytest.raises(InsufficientHistory):
        draw_blocks(np.array([]), 1, 1, 3, rng())


def test_assemble_places_columns_by_kind() -> None:
    mask = np.array([True, False, True])
    out = assemble(np.array([[1.0, 2.0]]), np.array([[9.0]]), mask)
    assert out.tolist() == [[1.0, 9.0, 2.0]]


def test_bootstrap_draws_each_day_from_its_own_kind() -> None:
    h = history(date(2026, 9, 1), 60, work=10.0, off=0.0)
    days = future_days(date(2026, 10, 20))
    paths = BootstrapForecaster()(h, days, 100, rng())
    mask = workday_mask(days)
    assert np.all(paths[:, mask] == 10.0)
    assert np.all(paths[:, ~mask] == 0.0)


def test_bootstrap_only_uses_the_window() -> None:
    old = history(date(2026, 8, 1), 30, work=999.0, off=999.0)
    new = history(date(2026, 8, 31), 30, work=1.0, off=1.0)
    h = History(old.days + new.days, np.concatenate([old.spend, new.spend]), np.concatenate([old.workday, new.workday]))
    paths = BootstrapForecaster(window_days=30)(h, future_days(date(2026, 10, 1)), 200, rng())
    assert np.all(paths == 1.0)


def test_bootstrap_is_deterministic_for_a_seed() -> None:
    days = days_from(date(2026, 9, 1), 60)
    mask = workday_mask(days)
    h = History(days, np.random.default_rng(1).gamma(2.0, 5.0, 60), mask)
    future = future_days(date(2026, 10, 15))
    a = BootstrapForecaster()(h, future, 50, np.random.default_rng(3))
    b = BootstrapForecaster()(h, future, 50, np.random.default_rng(3))
    assert np.array_equal(a, b)


def test_ets_refuses_short_history() -> None:
    with pytest.raises(InsufficientHistory):
        EtsForecaster()(history(date(2026, 10, 1), 10, 5.0, 0.0), future_days(date(2026, 10, 10)), 10, rng())


def test_bayes_refuses_forecast_days_inside_history() -> None:
    # Regression: an overlap made the trend offset negative and sliced the wrong days.
    h = history(date(2026, 8, 1), 90, 5.0, 0.0)
    with pytest.raises(ValueError, match="inside the history"):
        BayesForecaster()(h, future_days(date(2026, 10, 15)), 10, rng())


def test_linear_matches_old_point_forecast() -> None:
    # 1-2 Oct 2026 (Thu-Fri) at $10, then the weekend at $3: $26 over 2 working days.
    h = History(
        days_from(date(2026, 10, 1), 4),
        np.array([10.0, 10.0, 3.0, 3.0]),
        np.array([True, True, False, False]),
    )
    days = future_days(date(2026, 10, 4))
    paths = LinearForecaster()(h, days, 5, rng())
    assert paths.shape == (5, len(days))
    # 22 working days in Oct 2026, so 20 left at $13 each.
    assert 26.0 + paths[0].sum() == pytest.approx(26.0 / 2 * 22)


def test_linear_with_no_working_day_elapsed_is_flat() -> None:
    h = History((date(2026, 11, 1),), np.array([4.0]), np.array([False]))
    assert LinearForecaster()(h, future_days(date(2026, 11, 1)), 2, rng()).sum() == 0.0


def test_step_levels_flat_when_phi_zero_and_no_noise() -> None:
    one = np.ones(2)
    zero = np.zeros(2)
    out = step_levels(one, np.full(2, 0.5), zero, zero, zero, 4, rng())
    # The first step still carries the slope, then it is damped to nothing.
    assert out.tolist() == [[1.5, 1.5, 1.5, 1.5]] * 2


def test_step_levels_linear_when_phi_one_and_no_noise() -> None:
    one = np.ones(1)
    zero = np.zeros(1)
    out = step_levels(zero, np.full(1, 0.5), one, zero, zero, 4, rng())
    assert out.tolist() == [[0.5, 1.0, 1.5, 2.0]]


def test_hurdle_spend_respects_zero_rate_and_kind() -> None:
    n = 4000
    levels = np.zeros((n, 2))
    off = np.array([False, True])
    out = hurdle_spend(
        levels, off, np.full(n, -10.0), np.zeros(n), np.full(n, 0.8), np.full(n, 0.2), rng()
    )
    assert np.all(out >= 0)
    assert np.mean(out[:, 0] > 0) == pytest.approx(0.8, abs=0.03)
    assert np.mean(out[:, 1] > 0) == pytest.approx(0.2, abs=0.03)
    assert set(np.round(out[:, 0], 6)) <= {0.0, 1.0}


def test_fan_starts_from_spent_and_is_ordered() -> None:
    paths = np.random.default_rng(2).gamma(2.0, 3.0, size=(500, 8))
    days = days_from(date(2026, 10, 24), 8)
    f = fan(100.0, paths, days)
    qs = sorted(f.quantiles)
    stacked = np.stack([f.quantiles[q] for q in qs])
    assert np.all(np.diff(stacked, axis=0) >= 0)
    assert np.all(np.diff(f.quantiles[0.5]) >= 0)
    assert np.all(f.quantiles[0.05] >= 100.0)
    assert np.allclose(f.month_end, 100.0 + paths.sum(axis=1))


def test_fan_band_widens_with_horizon() -> None:
    paths = np.random.default_rng(4).exponential(5.0, size=(4000, 20))
    f = fan(0.0, paths, days_from(date(2026, 10, 1), 20))
    width = f.quantiles[0.95] - f.quantiles[0.05]
    assert np.all(np.diff(width) > 0)


def test_fan_with_no_days_left_is_what_was_spent() -> None:
    f = fan(42.0, np.zeros((3, 0)), ())
    assert f.month_end.tolist() == [42.0, 42.0, 42.0]


def test_prob_over() -> None:
    assert prob_over(np.array([100.0, 600.0, 400.0, 501.0]), 500.0) == 0.5


def test_pinball_hand_values() -> None:
    samples = np.array([10.0])
    assert pinball(samples, 14.0, 0.9) == pytest.approx(0.9 * 4)
    assert pinball(samples, 6.0, 0.9) == pytest.approx(0.1 * 4)


def test_crps_of_point_mass_is_absolute_error() -> None:
    assert crps(np.full(10, 3.0), 7.5) == pytest.approx(4.5)


def test_crps_two_point_hand_value() -> None:
    # X uniform on {0, 2}, y = 0: E|X-y| = 1, E|X-X'| = 1, so CRPS = 0.5.
    assert crps(np.array([0.0, 2.0]), 0.0) == pytest.approx(0.5)


def test_crps_rewards_sharper_correct_forecast() -> None:
    r = np.random.default_rng(5)
    assert crps(r.normal(10, 1, 2000), 10.0) < crps(r.normal(10, 5, 2000), 10.0)


def test_covered() -> None:
    samples = np.linspace(0, 100, 1001)
    assert covered(samples, 50.0)
    assert not covered(samples, 99.0)


def test_backtest_origins_only_full_months() -> None:
    # 15 Aug to 3 Nov 2026: only September and October are complete.
    h = history(date(2026, 8, 15), 81, 1.0, 0.0)
    origins = backtest_origins(h)
    assert {o.month for o in origins} == {9, 10}
    # 5th, 10th and 15th working day of October 2026.
    assert [o for o in origins if o.month == 10] == [
        date(2026, 10, 7),
        date(2026, 10, 14),
        date(2026, 10, 21),
    ]


def test_backtest_origins_empty_history() -> None:
    assert backtest_origins(History((), np.array([]), np.array([], dtype=np.bool_))) == ()
