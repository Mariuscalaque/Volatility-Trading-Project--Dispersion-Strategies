"""Signal validation and economic analysis for dispersion timing signals.

Provides bucketed P&L analysis, forward-return predictability tests, and
conditional performance splits to assess whether a timing signal carries
genuine economic content for the dispersion strategy.

Usage
-----
>>> from src.dispersion.signal_analysis import (
...     analyze_signal_buckets,
...     compute_forward_pnl_correlation,
...     compute_conditional_performance,
... )
>>> buckets = analyze_signal_buckets(signal_zscore, forward_returns, n_buckets=5)
>>> fwd_corr = compute_forward_pnl_correlation(signal_zscore, daily_pnl)
>>> cond_perf = compute_conditional_performance(static_nav, signal_zscore)
"""

import numpy as np
import pandas as pd
from scipy import stats

from src.metrics.performance import (
    realized_returns,
    sharpe_ratio,
    max_drawdown,
    calmar_ratio,
)
from src.metrics.volatility import realized_volatility


def analyze_signal_buckets(
    signal: pd.Series,
    forward_pnl: pd.Series,
    n_buckets: int = 5,
    labels: list[str] | None = None,
) -> pd.DataFrame:
    """Analyze strategy P&L conditional on signal quintiles (or deciles).

    Sorts trading days into equal-frequency buckets by signal level, then
    computes the average forward P&L in each bucket.  A monotonically
    increasing profile (Q1 lowest P&L, Q5 highest) validates the signal.

    Args:
        signal: Signal series (e.g. z-score at date t), indexed by date.
        forward_pnl: Strategy returns or P&L at t+1 for the same dates
                     (caller must handle the time lag).
        n_buckets: Number of buckets (default 5 = quintiles).
        labels: Bucket labels.  Defaults to ["Q1", ..., "Qn"].

    Returns:
        DataFrame indexed by bucket with columns:
            mean_pnl, median_pnl, std_pnl, count, hit_rate, mean_signal
    """
    common = signal.dropna().index.intersection(forward_pnl.dropna().index)
    s = signal.reindex(common)
    p = forward_pnl.reindex(common)

    if labels is None:
        labels = [f"Q{i + 1}" for i in range(n_buckets)]

    buckets = pd.qcut(s, n_buckets, labels=labels, duplicates="drop")

    result = pd.DataFrame({
        "mean_pnl": p.groupby(buckets).mean(),
        "median_pnl": p.groupby(buckets).median(),
        "std_pnl": p.groupby(buckets).std(),
        "count": p.groupby(buckets).count(),
        "hit_rate": p.groupby(buckets).apply(lambda x: (x > 0).mean()),
        "mean_signal": s.groupby(buckets).mean(),
    })
    result.index.name = "bucket"
    return result


def compute_forward_pnl_correlation(
    signal: pd.Series,
    daily_pnl: pd.Series,
    horizons: list[int] | None = None,
) -> pd.DataFrame:
    """Compute the Pearson correlation between signal(t) and forward cumulative P&L.

    For each horizon h, computes:
        corr( signal_t , sum(pnl_{t+1} ... pnl_{t+h}) )
    with a 2-sided t-test for significance.

    Args:
        signal: Signal series, indexed by date.
        daily_pnl: Daily P&L (or return) series from the static backtest.
        horizons: Forward horizons in trading days.
                  Default: [1, 5, 10, 21] (1d, 1w, 2w, 1m).

    Returns:
        DataFrame with columns ['horizon', 'correlation', 'p_value', 'n_obs'].
    """
    if horizons is None:
        horizons = [1, 5, 10, 21]

    common = signal.dropna().index.intersection(daily_pnl.dropna().index)
    s = signal.reindex(common).sort_index()
    p = daily_pnl.reindex(common).sort_index()

    rows = []
    for h in horizons:
        # Forward cumulative: sum of next h days' P&L
        forward_cum = p.rolling(h).sum().shift(-h)
        valid = s.index.intersection(forward_cum.dropna().index)
        if len(valid) < 10:
            rows.append({
                "horizon": h,
                "correlation": np.nan,
                "p_value": np.nan,
                "n_obs": len(valid),
            })
            continue
        corr, pval = stats.pearsonr(
            s.reindex(valid).values,
            forward_cum.reindex(valid).values,
        )
        rows.append({
            "horizon": h,
            "correlation": round(corr, 4),
            "p_value": round(pval, 4),
            "n_obs": len(valid),
        })

    return pd.DataFrame(rows)


def compute_conditional_performance(
    static_nav: pd.Series,
    signal: pd.Series,
    threshold: float = 0.0,
) -> pd.DataFrame:
    """Compare performance metrics when signal is favorable vs unfavorable.

    Splits the static daily returns into two regimes based on the *lagged*
    signal (signal at t-1 determines the regime for return at t).

    Args:
        static_nav: NAV series from the static backtest.
        signal: Signal series (e.g. z-score).
        threshold: Signal level above which the regime is "favorable".

    Returns:
        DataFrame with index ['favorable', 'unfavorable'] and columns:
            n_days, ann_return (%), ann_vol (%), sharpe, max_dd (%), hit_rate (%)
    """
    returns = static_nav.pct_change().dropna()
    lagged_signal = signal.shift(1).reindex(returns.index)

    valid = lagged_signal.dropna().index
    returns = returns.reindex(valid)
    lagged_signal = lagged_signal.reindex(valid)

    favorable_mask = lagged_signal > threshold
    rows = []
    for label, mask in [("favorable", favorable_mask), ("unfavorable", ~favorable_mask)]:
        r = returns[mask]
        if len(r) < 5:
            rows.append({
                "regime": label, "n_days": len(r),
                "ann_return": np.nan, "ann_vol": np.nan,
                "sharpe": np.nan, "max_dd": np.nan, "hit_rate": np.nan,
            })
            continue
        rows.append({
            "regime": label,
            "n_days": len(r),
            "ann_return": round(realized_returns(r) * 100, 2),
            "ann_vol": round(realized_volatility(r) * 100, 2),
            "sharpe": round(sharpe_ratio(r), 3),
            "max_dd": round(max_drawdown(r) * 100, 2),
            "hit_rate": round((r > 0).mean() * 100, 1),
        })

    return pd.DataFrame(rows).set_index("regime")
