"""Probabilistic month-end spend forecasts.

Every model simulates sample paths of daily spend for the days left in the month. Fans, month-end
intervals, P(over budget) and backtest scores are all computed from those paths, so the month-end
total is a sum over each simulated path, not a sum of per-day quantiles.

The fitting libraries (statsmodels, NumPyro) are the only code here that is not pure; everything
else takes its randomness as a `numpy.random.Generator`.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Final, Literal, Protocol, TypeAlias

import numpy as np
import numpy.typing as npt
import polars as pl

from ledger_view.ledger import is_working_day, month_days

Floats: TypeAlias = npt.NDArray[np.float64]
Bools: TypeAlias = npt.NDArray[np.bool_]
# Shape (n_paths, n_days), values >= 0.
Paths: TypeAlias = Floats

ModelName = Literal["bootstrap", "ets", "bayes", "linear"]
FAN_LEVELS: Final[tuple[float, ...]] = (0.05, 0.25, 0.5, 0.75, 0.95)
BACKTEST_WORKDAYS: Final[tuple[int, ...]] = (5, 10, 15)
MIN_WORK_DAYS: Final = 10


class InsufficientHistory(ValueError):
    """The model cannot be fitted on this little history."""


@dataclass(frozen=True, eq=False)
class History:
    """Daily total spend, one entry per calendar day in order, with the day's working-day flag."""

    days: tuple[date, ...]
    spend: Floats
    workday: Bools

    def until(self, last: date) -> "History":
        """The days up to and including `last`."""
        n = sum(1 for d in self.days if d <= last)
        return History(self.days[:n], self.spend[:n], self.workday[:n])

    def since(self, first: date) -> "History":
        """The days from `first` on."""
        skip = sum(1 for d in self.days if d < first)
        return History(self.days[skip:], self.spend[skip:], self.workday[skip:])

    def spent_in_month(self, month: date) -> float:
        """Spend in `month`'s calendar month among the days held."""
        return float(
            sum(s for d, s in zip(self.days, self.spend, strict=True) if d.replace(day=1) == month)
        )


@dataclass(frozen=True, eq=False)
class Fan:
    """Cumulative month spend per future day at each quantile level, and each path's month end."""

    days: tuple[date, ...]
    quantiles: Mapping[float, Floats]
    month_end: Floats


class Forecaster(Protocol):
    def __call__(
        self, history: History, days: tuple[date, ...], n_paths: int, rng: np.random.Generator
    ) -> Paths: ...


def history_from_paced(paced: pl.DataFrame, until: date) -> History:
    """Build a `History` from the `budget_pace` frame, cut off after `until`."""
    rows = paced.filter(pl.col("day") <= until).sort("day")
    return History(
        days=tuple(rows["day"].to_list()),
        spend=rows["cost_usd"].to_numpy().astype(np.float64),
        workday=rows["workday"].to_numpy().astype(np.bool_),
    )


def future_days(today: date) -> tuple[date, ...]:
    """The days left in `today`'s month after `today`."""
    return tuple(d for d in month_days(today.year, today.month) if d > today)


def workday_mask(days: Sequence[date]) -> Bools:
    return np.array([is_working_day(d) for d in days], dtype=np.bool_)


def draw_pool(pool: Floats, n_paths: int, n: int, rng: np.random.Generator) -> Floats:
    """Independent draws from `pool`. An empty pool means no spend was ever seen: zeros."""
    if pool.size == 0:
        return np.zeros((n_paths, n))
    return rng.choice(pool, size=(n_paths, n), replace=True)


def draw_blocks(pool: Floats, n_paths: int, n: int, block: int, rng: np.random.Generator) -> Floats:
    """Moving-block bootstrap: runs of `block` consecutive values from `pool`, cut to `n`."""
    if pool.size == 0:
        raise InsufficientHistory("no working days to resample")
    size = min(block, pool.size)
    n_blocks = -(-n // size)
    starts = rng.integers(0, pool.size - size + 1, size=(n_paths, n_blocks))
    idx = (starts[..., None] + np.arange(size)).reshape(n_paths, n_blocks * size)[:, :n]
    return np.asarray(pool[idx], dtype=np.float64)


def assemble(work: Floats, off: Floats, mask: Bools) -> Paths:
    """Put working-day draws and off-day draws into the columns their days belong to."""
    paths = np.zeros((work.shape[0], mask.size))
    paths[:, mask] = work
    paths[:, ~mask] = off
    return paths


@dataclass(frozen=True)
class BootstrapForecaster:
    """Resample recent days of the same kind. No trend: the honest baseline."""

    window_days: int = 56
    block: int = 3

    def __call__(
        self, history: History, days: tuple[date, ...], n_paths: int, rng: np.random.Generator
    ) -> Paths:
        mask = workday_mask(days)
        if history.days:
            history = history.since(history.days[-1] - timedelta(days=self.window_days - 1))
        work = draw_blocks(
            history.spend[history.workday], n_paths, int(mask.sum()), self.block, rng
        )
        off = draw_pool(history.spend[~history.workday], n_paths, int((~mask).sum()), rng)
        return assemble(work, off, mask)


@dataclass(frozen=True)
class EtsForecaster:
    """Damped-trend ETS on log1p(spend) over working days; off days resampled.

    Fitting on working days only removes the weekly pattern and the holidays in one go. The damped
    trend lets the level bend instead of extrapolating a straight line forever.
    """

    window_days: int = 120

    def __call__(
        self, history: History, days: tuple[date, ...], n_paths: int, rng: np.random.Generator
    ) -> Paths:
        from statsmodels.tsa.exponential_smoothing.ets import ETSModel

        mask = workday_mask(days)
        if history.days:
            history = history.since(history.days[-1] - timedelta(days=self.window_days - 1))
        y = np.log1p(history.spend[history.workday])
        if y.size < MIN_WORK_DAYS:
            raise InsufficientHistory(f"ETS needs {MIN_WORK_DAYS} working days, got {y.size}")
        k = int(mask.sum())
        if k == 0:
            work = np.zeros((n_paths, 0))
        else:
            fit = ETSModel(y, error="add", trend="add", damped_trend=True).fit(disp=False)
            # statsmodels takes an int seed or RandomState, not a Generator.
            seed = int(rng.integers(0, 2**31 - 1))
            sim = fit.simulate(k, anchor="end", repetitions=n_paths, random_state=seed)
            work = np.clip(np.expm1(np.asarray(sim, dtype=np.float64).reshape(k, n_paths).T), 0, None)
        off = draw_pool(history.spend[~history.workday], n_paths, int((~mask).sum()), rng)
        return assemble(work, off, mask)


@dataclass(frozen=True)
class Posterior:
    """Draws of the Bayesian model's parameters and its final state, one entry per draw."""

    level: Floats
    slope: Floats
    phi: Floats
    sigma_level: Floats
    sigma_slope: Floats
    beta_off: Floats
    sigma: Floats
    p_work: Floats
    p_off: Floats


def step_levels(
    level: Floats,
    slope: Floats,
    phi: Floats,
    sigma_level: Floats,
    sigma_slope: Floats,
    n_days: int,
    rng: np.random.Generator,
) -> Floats:
    """Roll the damped local linear trend forward `n_days`, one path per entry of the inputs.

    level[t+1] = level[t] + slope[t] + sigma_level * e,  slope[t+1] = phi * slope[t] + sigma_slope * e
    """
    out = np.empty((level.size, n_days))
    lv, sl = level.copy(), slope.copy()
    for t in range(n_days):
        lv = lv + sl + sigma_level * rng.standard_normal(lv.size)
        sl = phi * sl + sigma_slope * rng.standard_normal(sl.size)
        out[:, t] = lv
    return out


def hurdle_spend(
    levels: Floats,
    off: Bools,
    beta_off: Floats,
    sigma: Floats,
    p_work: Floats,
    p_off: Floats,
    rng: np.random.Generator,
) -> Paths:
    """Spend per path and day: zero with probability 1-p, else lognormal around the level."""
    p = np.where(off, p_off[:, None], p_work[:, None])
    spends = rng.random(levels.shape) < p
    log_mu = levels + beta_off[:, None] * off
    amount = np.exp(log_mu + sigma[:, None] * rng.standard_normal(levels.shape))
    return np.where(spends, amount, 0.0)


def bayes_model(log_spend: Floats, positive: Bools, off: Bools, prior_level: float) -> None:
    """Hurdle-lognormal with a damped local linear trend on the log level, one step per day."""
    import jax
    import jax.numpy as jnp
    import numpyro
    import numpyro.distributions as dist

    n = log_spend.size
    level0 = numpyro.sample("level0", dist.Normal(prior_level, 1.0))
    slope0 = numpyro.sample("slope0", dist.Normal(0.0, 0.05))
    phi = numpyro.sample("phi", dist.Beta(8.0, 2.0))
    sigma_level = numpyro.sample("sigma_level", dist.HalfNormal(0.1))
    sigma_slope = numpyro.sample("sigma_slope", dist.HalfNormal(0.02))
    beta_off = numpyro.sample("beta_off", dist.Normal(0.0, 1.0))
    sigma = numpyro.sample("sigma", dist.HalfNormal(1.0))
    p_work = numpyro.sample("p_work", dist.Beta(1.0, 1.0))
    p_off = numpyro.sample("p_off", dist.Beta(1.0, 1.0))
    eps = numpyro.sample("eps", dist.Normal(0.0, 1.0).expand([max(n - 1, 0), 2]).to_event(2))

    def step(carry: tuple[jax.Array, jax.Array], e: jax.Array) -> tuple[tuple[jax.Array, jax.Array], jax.Array]:
        lv, sl = carry
        lv = lv + sl + sigma_level * e[0]
        sl = phi * sl + sigma_slope * e[1]
        return (lv, sl), lv

    (last_level, last_slope), rest = jax.lax.scan(step, (level0, slope0), eps)
    levels = jnp.concatenate([jnp.atleast_1d(level0), rest])
    numpyro.deterministic("last_level", last_level)
    numpyro.deterministic("last_slope", last_slope)

    off_j = jnp.asarray(off)
    pos_j = jnp.asarray(positive)
    numpyro.sample("spends", dist.Bernoulli(jnp.where(off_j, p_off, p_work)), obs=pos_j.astype(jnp.float32))
    numpyro.sample(
        "amount",
        dist.Normal(levels + beta_off * off_j, sigma).mask(pos_j),
        obs=jnp.asarray(np.where(positive, log_spend, 0.0)),
    )


@dataclass(frozen=True)
class BayesForecaster:
    """Bayesian structural time series, fitted with NUTS in NumPyro.

    Hurdle-lognormal spend (a chance of a zero day, else lognormal), a damped local linear trend
    on the log level, and an off-day offset. The damped slope gives a level that bends.
    """

    window_days: int = 120
    num_warmup: int = 500
    num_samples: int = 500

    def fit(self, history: History, rng: np.random.Generator) -> Posterior:
        import jax
        from numpyro.infer import MCMC, NUTS

        positive = history.spend > 0
        if int((positive & history.workday).sum()) < MIN_WORK_DAYS:
            raise InsufficientHistory(f"the Bayesian model needs {MIN_WORK_DAYS} working days with spend")
        log_spend = np.where(positive, np.log(np.where(positive, history.spend, 1.0)), 0.0)
        prior_level = float(log_spend[positive & history.workday].mean())
        mcmc = MCMC(
            NUTS(bayes_model),
            num_warmup=self.num_warmup,
            num_samples=self.num_samples,
            progress_bar=False,
        )
        key = jax.random.PRNGKey(int(rng.integers(0, 2**31 - 1)))
        mcmc.run(key, log_spend, positive, ~history.workday, prior_level)
        s = mcmc.get_samples()

        def draws(name: str) -> Floats:
            return np.asarray(s[name], dtype=np.float64)

        return Posterior(
            level=draws("last_level"),
            slope=draws("last_slope"),
            phi=draws("phi"),
            sigma_level=draws("sigma_level"),
            sigma_slope=draws("sigma_slope"),
            beta_off=draws("beta_off"),
            sigma=draws("sigma"),
            p_work=draws("p_work"),
            p_off=draws("p_off"),
        )

    def __call__(
        self, history: History, days: tuple[date, ...], n_paths: int, rng: np.random.Generator
    ) -> Paths:
        if history.days:
            history = history.since(history.days[-1] - timedelta(days=self.window_days - 1))
        if not days:
            return np.zeros((n_paths, 0))
        # Days between the last observed day and the first forecast day still move the trend.
        gap = (days[0] - history.days[-1]).days - 1 if history.days else 0
        if gap < 0:
            raise ValueError(f"forecast starts {days[0]}, inside the history up to {history.days[-1]}")
        post = self.fit(history, rng)
        pick = rng.integers(0, post.level.size, size=n_paths)
        levels = step_levels(
            post.level[pick],
            post.slope[pick],
            post.phi[pick],
            post.sigma_level[pick],
            post.sigma_slope[pick],
            gap + len(days),
            rng,
        )[:, gap:]
        return hurdle_spend(
            levels,
            ~workday_mask(days),
            post.beta_off[pick],
            post.sigma[pick],
            post.p_work[pick],
            post.p_off[pick],
            rng,
        )


@dataclass(frozen=True)
class LinearForecaster:
    """The old point forecast as one repeated path: this month's rate per elapsed working day."""

    def __call__(
        self, history: History, days: tuple[date, ...], n_paths: int, rng: np.random.Generator
    ) -> Paths:
        mask = workday_mask(days)
        if not days:
            return np.zeros((n_paths, 0))
        month = days[0].replace(day=1)
        in_month = [d.replace(day=1) == month for d in history.days]
        elapsed = int(sum(w for w, m in zip(history.workday, in_month, strict=True) if m))
        rate = history.spent_in_month(month) / elapsed if elapsed else 0.0
        return np.broadcast_to(np.where(mask, rate, 0.0), (n_paths, mask.size)).copy()


def fan(
    spent_so_far: float,
    paths: Paths,
    days: tuple[date, ...],
    levels: tuple[float, ...] = FAN_LEVELS,
) -> Fan:
    """Cumulative month spend quantiles per future day, starting from what is already spent."""
    cum = spent_so_far + np.cumsum(paths, axis=1)
    month_end = cum[:, -1] if days else np.full(paths.shape[0], spent_so_far)
    return Fan(
        days=days,
        quantiles={q: np.quantile(cum, q, axis=0) for q in levels},
        month_end=month_end,
    )


def prob_over(month_end: Floats, budget: float) -> float:
    return float(np.mean(month_end > budget))


def pinball(samples: Floats, actual: float, q: float) -> float:
    """Quantile loss of the forecast's q-quantile against the outcome."""
    diff = actual - float(np.quantile(samples, q))
    return max(q * diff, (q - 1) * diff)


def crps(samples: Floats, actual: float) -> float:
    """Continuous ranked probability score of a sample forecast: E|X-y| - E|X-X'|/2. Lower is better."""
    x = np.sort(samples)
    n = int(x.size)
    spread = 2.0 * float(np.sum((2 * np.arange(1, n + 1) - n - 1) * x)) / (n * n)
    return float(np.mean(np.abs(x - actual))) - spread / 2


def covered(samples: Floats, actual: float, lo: float = 0.05, hi: float = 0.95) -> bool:
    low, high = np.quantile(samples, [lo, hi])
    return bool(low <= actual <= high)


def backtest_origins(
    history: History, workday_numbers: tuple[int, ...] = BACKTEST_WORKDAYS
) -> tuple[date, ...]:
    """The n-th working days of each month that `history` holds from its first to its last day."""
    if not history.days:
        return ()
    first, last = history.days[0], history.days[-1]
    origins: list[date] = []
    month = first.replace(day=1)
    while month <= last:
        all_days = month_days(month.year, month.month)
        if all_days[0] >= first and all_days[-1] <= last:
            work = [d for d in all_days if is_working_day(d)]
            origins.extend(work[n - 1] for n in workday_numbers if n <= len(work))
        month = all_days[-1] + timedelta(days=1)
    return tuple(origins)


def backtest(
    history: History,
    forecasters: Mapping[ModelName, Forecaster],
    origins: Sequence[date],
    n_paths: int,
    rng: np.random.Generator,
) -> pl.DataFrame:
    """Forecast each origin's month end from data up to the origin and score it against the outcome.

    One row per (model, origin). A model that cannot fit at an origin leaves no row there.
    """
    rows: list[dict[str, object]] = []
    for origin in origins:
        month = origin.replace(day=1)
        seen = history.until(origin)
        days = future_days(origin)
        spent = seen.spent_in_month(month)
        actual = history.spent_in_month(month)
        for name, forecaster in forecasters.items():
            try:
                paths = forecaster(seen, days, n_paths, rng)
            except InsufficientHistory:
                continue
            ends = spent + paths.sum(axis=1)
            p05, median, p95 = np.quantile(ends, [0.05, 0.5, 0.95])
            rows.append(
                {
                    "model": name,
                    "month": month,
                    "origin": origin,
                    "actual": actual,
                    "median": float(median),
                    "p05": float(p05),
                    "p95": float(p95),
                    "crps": crps(ends, actual),
                    "covered90": covered(ends, actual),
                }
            )
    schema = {
        "model": pl.String,
        "month": pl.Date,
        "origin": pl.Date,
        "actual": pl.Float64,
        "median": pl.Float64,
        "p05": pl.Float64,
        "p95": pl.Float64,
        "crps": pl.Float64,
        "covered90": pl.Boolean,
    }
    return pl.DataFrame(rows, schema=schema)


def backtest_summary(scores: pl.DataFrame) -> pl.DataFrame:
    """Mean CRPS and the share of outcomes inside the 90% band, per model. Best CRPS first."""
    return (
        scores.group_by("model")
        .agg(
            pl.len().alias("forecasts"),
            pl.col("crps").mean().alias("mean_crps_usd"),
            pl.col("covered90").mean().alias("coverage90"),
            (pl.col("p95") - pl.col("p05")).mean().alias("mean_width90_usd"),
        )
        .sort("mean_crps_usd")
    )
