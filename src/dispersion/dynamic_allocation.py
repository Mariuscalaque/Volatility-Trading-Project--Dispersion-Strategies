"""Dynamic allocation overlay for dispersion strategies.

Implements signal-based timing overlays that modulate the static dispersion
trade's exposure based on correlation or variance risk premium signals.

The overlay is applied *post-hoc* on the static backtest's daily return
series.  This is the standard approach for timing strategy evaluation
(Moskowitz, Ooi & Pedersen, 2012):

    r_dynamic(t) = exposure(t-1) × r_static(t)

where exposure is derived from a signal observed at t-1 (no look-ahead bias).

Usage
-----
>>> from src.dispersion.dynamic_allocation import (
...     compute_correlation_spread_signal,
...     compute_signal_zscore,
...     build_dynamic_exposure,
...     apply_dynamic_overlay,
... )
>>> signal = compute_correlation_spread_signal(iv_spy, iv_aapl, spy_ret, aapl_ret)
>>> zscore = compute_signal_zscore(signal["corr_spread"], window=63)
>>> exposure = build_dynamic_exposure(zscore, method="zscore_clipped")
>>> result = apply_dynamic_overlay(static_nav, exposure)
>>> result["dynamic_nav"].plot()
"""

from typing import Literal

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Signal construction
# ---------------------------------------------------------------------------


def compute_correlation_spread_signal(
    iv_index: pd.Series,
    iv_component: pd.Series,
    index_returns: pd.Series,
    component_returns: pd.Series,
    w_component: float = 0.07,
    realized_corr_window: int = 30,
) -> pd.DataFrame:
    """Compute a single-name implied-minus-realized correlation proxy spread.

    Uses a simplified 2-component variance decomposition to extract a
    *proxy* for implied correlation from ATM implied volatilities:

        ρ_impl ≈ (σ²_idx - w²·σ²_cmp - (1-w)²·σ²_idx)
                  / (2·w·(1-w)·σ_cmp·σ_idx)

    This quantity is not the true average basket correlation of the index.
    Recovering that object would require options on multiple constituents
    (or a full covariance model for the basket). With only one component
    and the index, the output should be interpreted as a *single-name
    correlation-risk-premium proxy*.

    The realized leg is a rolling Pearson correlation of the index and
    component returns, so the final spread is also a proxy signal rather
    than an exact implied-minus-realized basket correlation measure.

    Args:
        iv_index: ATM implied volatility of the index (annualized).
        iv_component: ATM implied volatility of the component.
        index_returns: Daily returns of the index.
        component_returns: Daily returns of the component.
        w_component: Weight of the component in the index (~7% for AAPL/SPY).
        realized_corr_window: Rolling window for realized correlation (days).

    Returns:
        DataFrame with columns ['implied_corr', 'realized_corr', 'corr_spread'],
        indexed by date. These should be interpreted as proxy quantities.
    """
    common = (
        iv_index.dropna().index
        .intersection(iv_component.dropna().index)
        .intersection(index_returns.dropna().index)
        .intersection(component_returns.dropna().index)
    ).sort_values()

    iv_idx = iv_index.reindex(common)
    iv_cmp = iv_component.reindex(common)
    idx_r = index_returns.reindex(common)
    cmp_r = component_returns.reindex(common)

    # Implied correlation (simplified 2-component decomposition)
    w = w_component
    numerator = iv_idx**2 - w**2 * iv_cmp**2 - (1 - w)**2 * iv_idx**2
    denominator = 2 * w * (1 - w) * iv_cmp * iv_idx
    implied_corr = (numerator / denominator).clip(-1, 1)

    # Realized correlation
    realized_corr = idx_r.rolling(realized_corr_window).corr(cmp_r)

    result = pd.DataFrame({
        "implied_corr": implied_corr,
        "realized_corr": realized_corr,
        "corr_spread": implied_corr - realized_corr,
    })
    return result.dropna()


def compute_vrp_spread_signal(
    iv_index: pd.Series,
    iv_component: pd.Series,
    index_returns: pd.Series,
    component_returns: pd.Series,
    rv_window: int = 21,
) -> pd.DataFrame:
    """Compute the VRP spread: VRP_index - VRP_component.

    VRP = IV² - RV² in variance space.  The spread represents the excess
    variance premium of the index over the component — a proxy for the
    correlation risk premium (Driessen et al., 2009; Carr & Wu, 2009).

    Args:
        iv_index: ATM implied volatility of the index (annualized).
        iv_component: ATM implied volatility of the component.
        index_returns: Daily returns of the index.
        component_returns: Daily returns of the component.
        rv_window: Rolling window for realized volatility (days).

    Returns:
        DataFrame with columns ['vrp_index', 'vrp_component', 'vrp_spread'],
        indexed by date.
    """
    rv_idx = index_returns.rolling(rv_window).std() * np.sqrt(252)
    rv_cmp = component_returns.rolling(rv_window).std() * np.sqrt(252)

    common = (
        iv_index.dropna().index
        .intersection(iv_component.dropna().index)
        .intersection(rv_idx.dropna().index)
        .intersection(rv_cmp.dropna().index)
    ).sort_values()

    vrp_idx = iv_index.reindex(common) ** 2 - rv_idx.reindex(common) ** 2
    vrp_cmp = iv_component.reindex(common) ** 2 - rv_cmp.reindex(common) ** 2

    return pd.DataFrame({
        "vrp_index": vrp_idx,
        "vrp_component": vrp_cmp,
        "vrp_spread": vrp_idx - vrp_cmp,
    }).dropna()


# ---------------------------------------------------------------------------
# Signal transformation
# ---------------------------------------------------------------------------


def compute_signal_zscore(
    signal: pd.Series,
    window: int = 63,
) -> pd.Series:
    """Compute the rolling z-score of a signal.

    z(t) = (signal(t) - μ_rolling(t)) / σ_rolling(t)

    where μ and σ are computed over a trailing window ending at t
    (no look-ahead).

    Args:
        signal: Raw signal series, indexed by date.
        window: Rolling window for mean and std (days).  Default 63 ≈ 3 months.

    Returns:
        Series of z-scores (first ``window`` values are NaN).
    """
    rolling_mean = signal.rolling(window).mean()
    rolling_std = signal.rolling(window).std()
    rolling_std = rolling_std.replace(0, np.nan)
    return ((signal - rolling_mean) / rolling_std).rename("zscore")


# ---------------------------------------------------------------------------
# Exposure construction
# ---------------------------------------------------------------------------

ExposureMethod = Literal["binary", "continuous_rank", "zscore_clipped"]


def build_dynamic_exposure(
    signal: pd.Series,
    method: ExposureMethod = "zscore_clipped",
    threshold: float = 0.0,
    lower_bound: float = 0.0,
    upper_bound: float = 1.0,
    rank_window: int = 252,
) -> pd.Series:
    """Convert a signal into a position exposure in [lower_bound, upper_bound].

    Three methods are supported:

    1. **binary**: exposure = 1 if signal > threshold, else 0.
       Simple on/off allocation.

    2. **continuous_rank**: exposure = rolling percentile rank of signal,
       rescaled to [lower_bound, upper_bound].

    3. **zscore_clipped** (recommended): exposure = clip(signal, lb, ub).
       Assumes ``signal`` is already a z-score.  Values above 0 indicate
       a favorable regime; values are capped at 1 to prevent leverage.

    Args:
        signal: Signal series (raw for binary/rank, z-scored for zscore_clipped).
        method: Exposure construction method.
        threshold: Threshold for the binary method.
        lower_bound: Minimum exposure (default 0.0 — no reverse dispersion).
        upper_bound: Maximum exposure (default 1.0 — full notional).
        rank_window: Rolling window for percentile rank method.

    Returns:
        Series of exposure values, same index as signal.
    """
    if method == "binary":
        exposure = (signal > threshold).astype(float)
    elif method == "continuous_rank":
        exposure = signal.rolling(rank_window).rank(pct=True)
        exposure = lower_bound + (upper_bound - lower_bound) * exposure
    elif method == "zscore_clipped":
        exposure = signal.clip(lower=lower_bound, upper=upper_bound)
    else:
        raise ValueError(
            f"Unknown method: {method!r}. "
            "Use 'binary', 'continuous_rank', or 'zscore_clipped'."
        )
    return exposure.rename("exposure")


# ---------------------------------------------------------------------------
# Dynamic overlay
# ---------------------------------------------------------------------------


def apply_dynamic_overlay(
    static_nav: pd.Series,
    exposure: pd.Series,
    lag: int = 1,
) -> pd.DataFrame:
    """Apply a dynamic exposure overlay to a static strategy's NAV.

    Implements the standard timing backtest (Moskowitz et al., 2012):

        r_dynamic(t) = exposure(t - lag) × r_static(t)

    The exposure is lagged to avoid look-ahead bias.

    Args:
        static_nav: NAV series from the static backtest (e.g. bt.nav["NAV"]).
        exposure: Exposure series in [0, 1] (or [-1, 1]).
        lag: Number of business days to lag the exposure (default 1).

    Returns:
        DataFrame with columns:
            - static_nav: Original NAV (rebased to 1)
            - dynamic_nav: NAV under the dynamic overlay
            - static_return: Daily return of the static strategy
            - dynamic_return: Daily return of the dynamic strategy
            - exposure: Lagged exposure applied on each date
    """
    static_returns = static_nav.pct_change().dropna()
    lagged_exposure = exposure.shift(lag).reindex(static_returns.index)
    lagged_exposure = lagged_exposure.fillna(0)

    dynamic_returns = lagged_exposure * static_returns
    dynamic_nav = (1 + dynamic_returns).cumprod()

    static_nav_aligned = static_nav.reindex(dynamic_nav.index)
    static_nav_aligned = static_nav_aligned / static_nav_aligned.iloc[0]

    return pd.DataFrame({
        "static_nav": static_nav_aligned,
        "dynamic_nav": dynamic_nav,
        "static_return": static_returns,
        "dynamic_return": dynamic_returns,
        "exposure": lagged_exposure,
    })


# ---------------------------------------------------------------------------
# Walk-forward (expanding window) overlay — truly out-of-sample
# ---------------------------------------------------------------------------


def compute_expanding_zscore(signal: pd.Series, min_window: int = 63) -> pd.Series:
    """Compute an expanding-window z-score (no fixed rolling window).

    At each date t, the z-score is computed using *all* data from the start
    of the series up to t — never any future data.  This avoids the implicit
    look-ahead that comes from choosing a rolling window length on the full
    sample.

    z(t) = (signal(t) - mean(signal[0:t])) / std(signal[0:t])

    The first ``min_window`` values are NaN (not enough history).

    Args:
        signal: Raw signal series, indexed by date.
        min_window: Minimum observations before emitting a z-score.

    Returns:
        Series of expanding z-scores.
    """
    expanding_mean = signal.expanding(min_periods=min_window).mean()
    expanding_std = signal.expanding(min_periods=min_window).std()
    expanding_std = expanding_std.replace(0, np.nan)
    return ((signal - expanding_mean) / expanding_std).rename("expanding_zscore")


def apply_walkforward_dynamic_overlay(
    static_nav: pd.Series,
    raw_signal: pd.Series,
    method: ExposureMethod = "zscore_clipped",
    min_window: int = 63,
    threshold: float = 0.0,
    lag: int = 1,
) -> pd.DataFrame:
    """Walk-forward dynamic overlay: expanding z-score → exposure → return.

    Unlike ``apply_dynamic_overlay`` which takes a pre-computed exposure,
    this function builds the exposure **from scratch** at each date using
    only past information (expanding window z-score).  This produces a
    genuinely out-of-sample dynamic return series.

    Args:
        static_nav: NAV series from the static backtest.
        raw_signal: Raw (un-z-scored) signal series (e.g. corr_spread).
        method: Exposure method ('binary', 'zscore_clipped', 'continuous_rank').
        min_window: Minimum expanding window for z-score estimation.
        threshold: Threshold for the binary method.
        lag: Business days to lag the exposure (default 1).

    Returns:
        DataFrame with same columns as ``apply_dynamic_overlay`` plus
        'expanding_zscore'.
    """
    # Step 1: expanding z-score (only past data at each t)
    exp_zscore = compute_expanding_zscore(raw_signal, min_window=min_window)

    # Step 2: exposure from the expanding z-score
    exposure = build_dynamic_exposure(
        exp_zscore, method=method, threshold=threshold,
    )

    # Step 3: standard lagged overlay
    result = apply_dynamic_overlay(static_nav, exposure, lag=lag)
    result["expanding_zscore"] = exp_zscore.reindex(result.index)
    return result
