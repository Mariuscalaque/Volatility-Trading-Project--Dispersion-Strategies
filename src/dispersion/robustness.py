"""Robustness and sensitivity analysis for dispersion strategies.

Provides functions to assess the stability of strategy performance and
dynamic allocation under varying:
- Transaction cost levels (bid-ask spread multiplier)
- Signal thresholds (binary / z-score cutoff)
- Rolling window lengths (for z-score computation)

Usage
-----
>>> from src.dispersion.robustness import (
...     estimate_daily_tcost,
...     run_tcost_sensitivity,
...     run_threshold_sensitivity,
...     run_window_sensitivity,
... )
>>> daily_tc = estimate_daily_tcost(bt.drifted_positions)
>>> tcost_df = run_tcost_sensitivity(bt.nav["NAV"], daily_tc)
>>> thresh_df = run_threshold_sensitivity(bt.nav["NAV"], signal_zscore)
"""

import numpy as np
import pandas as pd

from src.metrics.performance import (
    realized_returns,
    sharpe_ratio,
    max_drawdown,
    calmar_ratio,
)
from src.metrics.volatility import realized_volatility
from src.dispersion.dynamic_allocation import (
    compute_signal_zscore,
    compute_expanding_zscore,
    build_dynamic_exposure,
    apply_dynamic_overlay,
)


def estimate_daily_tcost(
    drifted_positions: pd.DataFrame,
) -> pd.Series:
    """Estimate daily transaction cost from backtester drifted positions.

    Computes ``|scaled_weight| × (ask - bid) / 2`` on option entry and
    exit dates.  Delta-hedge rows (stock positions) are excluded since
    they trade at mid.

    Args:
        drifted_positions: DataFrame from ``backtester.drifted_positions``.

    Returns:
        Series indexed by date with total estimated transaction cost per day.
    """
    dp = drifted_positions.copy()
    dp["half_spread"] = (dp["ask"] - dp["bid"]).clip(lower=0) / 2

    is_entry = dp["entry_date"] == dp["date"]
    is_exit = dp["expiration"] == dp["date"]
    is_option = dp["leg_name"] != "DELTA_HEDGING"
    is_trade = (is_entry | is_exit) & is_option

    dp["tcost"] = 0.0
    dp.loc[is_trade, "tcost"] = (
        dp.loc[is_trade, "scaled_weight"].abs()
        * dp.loc[is_trade, "half_spread"]
    )
    return dp.groupby("date")["tcost"].sum()


def _compute_metrics(daily_returns: pd.Series) -> dict:
    """Compute standard performance metrics from a daily return series."""
    return {
        "ann_return": round(realized_returns(daily_returns) * 100, 2),
        "ann_vol": round(realized_volatility(daily_returns) * 100, 2),
        "sharpe": round(sharpe_ratio(daily_returns), 3),
        "max_dd": round(max_drawdown(daily_returns) * 100, 2),
    }


def run_tcost_sensitivity(
    static_nav: pd.Series,
    daily_tcost: pd.Series,
    multipliers: list[float] | None = None,
) -> pd.DataFrame:
    """Evaluate performance under different transaction cost levels.

    The baseline backtest already includes 1× bid-ask costs.  For
    multiplier ``m``, additional costs of ``(m - 1) × daily_tcost`` are
    subtracted from daily returns.

    Args:
        static_nav: NAV series from the baseline backtest.
        daily_tcost: Daily tcost estimate (from ``estimate_daily_tcost``).
        multipliers: Cost multipliers to test.

    Returns:
        DataFrame indexed by multiplier with performance metrics.
    """
    if multipliers is None:
        multipliers = [1.0, 1.5, 2.0, 2.5, 3.0]

    static_returns = static_nav.pct_change().dropna()
    tcost_aligned = daily_tcost.reindex(static_returns.index, fill_value=0)

    # Express tcost as a fraction of lagged NAV for return-space subtraction
    nav_lagged = static_nav.shift(1).reindex(static_returns.index)
    tcost_as_return = tcost_aligned / nav_lagged.replace(0, np.nan)
    tcost_as_return = tcost_as_return.fillna(0)

    rows = []
    for m in multipliers:
        adjusted_returns = static_returns - (m - 1) * tcost_as_return
        metrics = _compute_metrics(adjusted_returns)
        metrics["tcost_multiplier"] = m
        rows.append(metrics)

    return pd.DataFrame(rows).set_index("tcost_multiplier")


def run_threshold_sensitivity(
    static_nav: pd.Series,
    signal_zscore: pd.Series,
    thresholds: list[float] | None = None,
    method: str = "binary",
) -> pd.DataFrame:
    """Evaluate dynamic strategy under different signal thresholds.

    For **binary**: exposure = 1 if z-score > threshold, else 0.
    For **zscore_shift**: exposure = clip(z-score − threshold, 0, 1).

    Args:
        static_nav: NAV from the static backtest.
        signal_zscore: Z-scored signal series.
        thresholds: Threshold values to sweep.
        method: ``"binary"`` or ``"zscore_shift"``.

    Returns:
        DataFrame indexed by threshold with metrics and average exposure.
    """
    if thresholds is None:
        thresholds = [-0.5, 0.0, 0.25, 0.5, 0.75, 1.0, 1.5]

    rows = []
    for thr in thresholds:
        if method == "binary":
            exposure = build_dynamic_exposure(
                signal_zscore, method="binary", threshold=thr,
            )
        else:
            shifted = signal_zscore - thr
            exposure = build_dynamic_exposure(
                shifted, method="zscore_clipped",
            )

        result = apply_dynamic_overlay(static_nav, exposure)
        daily_ret = result["dynamic_return"]
        metrics = _compute_metrics(daily_ret)
        metrics["threshold"] = thr
        metrics["avg_exposure"] = round(result["exposure"].mean(), 3)
        metrics["pct_invested"] = round((result["exposure"] > 0).mean() * 100, 1)
        rows.append(metrics)

    return pd.DataFrame(rows).set_index("threshold")


def run_window_sensitivity(
    static_nav: pd.Series,
    raw_signal: pd.Series,
    zscore_windows: list[int] | None = None,
    exposure_method: str = "zscore_clipped",
    zscore_type: str = "expanding",
) -> pd.DataFrame:
    """Evaluate dynamic strategy under different z-score estimation windows.

    Recomputes the z-score with each window, builds exposure, and applies
    the dynamic overlay. By default, uses an expanding-window z-score to
    stay consistent with the walk-forward methodology used in the notebook.

    Args:
        static_nav: NAV from the static backtest.
        raw_signal: Raw (un-z-scored) signal series.
        zscore_windows: Windows to test (trading days).
        exposure_method: Method for ``build_dynamic_exposure``.
        zscore_type: ``"expanding"`` or ``"rolling"``.

    Returns:
        DataFrame indexed by window with performance metrics.
    """
    if zscore_windows is None:
        zscore_windows = [21, 42, 63, 126, 252]

    rows = []
    for w in zscore_windows:
        if zscore_type == "expanding":
            zscore = compute_expanding_zscore(raw_signal, min_window=w)
        elif zscore_type == "rolling":
            zscore = compute_signal_zscore(raw_signal, window=w)
        else:
            raise ValueError("zscore_type must be 'expanding' or 'rolling'.")

        exposure = build_dynamic_exposure(zscore, method=exposure_method)
        result = apply_dynamic_overlay(static_nav, exposure)
        daily_ret = result["dynamic_return"]
        metrics = _compute_metrics(daily_ret)
        metrics["zscore_window"] = w
        metrics["avg_exposure"] = round(result["exposure"].mean(), 3)
        metrics["zscore_type"] = zscore_type
        rows.append(metrics)

    return pd.DataFrame(rows).set_index("zscore_window")


def run_signal_permutation_test(
    static_nav: pd.Series,
    raw_signal: pd.Series,
    method: str = "zscore_clipped",
    min_window: int = 63,
    n_permutations: int = 1000,
    seed: int = 42,
) -> dict:
    """Test whether the dynamic Sharpe is significantly better than a random signal.

    Procedure:
    1. Compute the actual walk-forward dynamic Sharpe using the real signal.
    2. Repeat ``n_permutations`` times with a time-shuffled signal (random
       permutation of signal *dates*, preserving values).
    3. Compute the p-value: fraction of permuted Sharpes ≥ actual Sharpe.

    This is a non-parametric test: under H₀ (signal has no timing value),
    the permuted Sharpes form the null distribution.

    Args:
        static_nav: NAV series from the static backtest.
        raw_signal: Raw (un-z-scored) signal series.
        method: Exposure method for ``apply_walkforward_dynamic_overlay``.
        min_window: Minimum expanding window for z-score estimation.
        n_permutations: Number of random permutations.
        seed: Random seed for reproducibility.

    Returns:
        Dict with keys: actual_sharpe, perm_sharpes (array), p_value, n_permutations.
    """
    from src.dispersion.dynamic_allocation import apply_walkforward_dynamic_overlay

    # Actual Sharpe (walk-forward)
    actual_result = apply_walkforward_dynamic_overlay(
        static_nav, raw_signal, method=method, min_window=min_window,
    )
    actual_sharpe = sharpe_ratio(actual_result["dynamic_return"])

    # Permutation loop
    rng = np.random.default_rng(seed)
    perm_sharpes = np.empty(n_permutations)
    signal_values = raw_signal.values.copy()

    for i in range(n_permutations):
        shuffled_values = rng.permutation(signal_values)
        shuffled_signal = pd.Series(shuffled_values, index=raw_signal.index, name=raw_signal.name)
        perm_result = apply_walkforward_dynamic_overlay(
            static_nav, shuffled_signal, method=method, min_window=min_window,
        )
        perm_sharpes[i] = sharpe_ratio(perm_result["dynamic_return"])

    p_value = (perm_sharpes >= actual_sharpe).mean()

    return {
        "actual_sharpe": round(actual_sharpe, 4),
        "perm_sharpes": perm_sharpes,
        "p_value": round(p_value, 4),
        "n_permutations": n_permutations,
    }
