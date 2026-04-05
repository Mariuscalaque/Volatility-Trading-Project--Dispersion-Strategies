import pandas as pd

from src.constants import TRADING_DAYS_PER_YEAR
from src.metrics.util import returns_to_levels
from src.metrics.volatility import realized_volatility


def realized_returns(returns: pd.Series) -> float:
    """Compute annualized realized return

    Args:
        returns (pd.Series): _description_

    Returns:
        float: Annualized realized returns.
    """
    return returns.mean() * TRADING_DAYS_PER_YEAR


def sharpe_ratio(returns: pd.Series, risk_free_rate: float | pd.Series = 0.0) -> float:
    """Compute the annualized Sharpe ratio of a daily returns series.

    Uses the standard definition: SR = E[R - Rf] / σ(R - Rf), annualized.
    When ``risk_free_rate`` is a constant (the default case), σ(R - Rf) = σ(R)
    and the two formulations are equivalent. When ``risk_free_rate`` is a
    time-varying Series, using σ(excess) is the correct definition.

    Parameters:
        returns: Series of daily returns.
        risk_free_rate: Annualized risk-free rate (scalar or Series).

    Returns:
        Annualized Sharpe ratio.
    """
    excess = excess_return(returns, risk_free_rate)
    return realized_returns(excess) / realized_volatility(excess)


def excess_return(
    returns: pd.Series,
    risk_free_rate: float | pd.Series = 0.0,
    annualized_risk_free_rate: bool = True,
) -> pd.Series:
    """Compute excess returns over the risk-free rate.

    Parameters:
        returns: Series of daily returns.
        risk_free_rate: Risk-free rate. Annualized by default, or daily if
                        ``annualized_risk_free_rate=False``.
        annualized_risk_free_rate: Whether ``risk_free_rate`` is expressed as
                                   an annualized figure (default True).

    Returns:
        Series of daily excess returns.
    """
    daily_rf = (
        risk_free_rate / TRADING_DAYS_PER_YEAR
        if annualized_risk_free_rate
        else risk_free_rate
    )
    return returns - daily_rf


def drawdown(returns: pd.Series) -> pd.Series:
    """Compute the drawdown series from a daily returns series.

    Parameters:
        returns: Series of daily returns.

    Returns:
        Series of drawdowns.
    """
    nav = returns_to_levels(returns)
    return (nav / nav.cummax()) - 1


def max_drawdown(returns: pd.Series) -> float:
    """Compute the maximum drawdown from a daily returns series.

    Parameters:
        returns: Series of daily returns.
    Returns:
        maximum drawdown of the returns series.
    """
    return drawdown(returns).min()


def calmar_ratio(returns: pd.Series) -> float:
    """Compute the Calmar ratio from a daily returns series.

    Parameters:
        returns: Series of daily returns.

    Returns:
        Calmar ratio of the returns series.
    """
    annualized_return = realized_returns(returns)
    maximum_drawdown = -max_drawdown(returns)
    if maximum_drawdown == 0:
        return float("inf")
    return annualized_return / maximum_drawdown


def rolling_sharpe_ratio(
    returns: pd.Series,
    window: int = 126,
    risk_free_rate: float = 0.0,
) -> pd.Series:
    """Compute rolling annualized Sharpe ratio.

    Parameters:
        returns: Series of daily returns.
        window: Rolling window in trading days (default 126 ≈ 6 months).
        risk_free_rate: Annualized risk-free rate (scalar, default 0).

    Returns:
        Series of rolling annualized Sharpe ratios.
    """
    daily_rf = risk_free_rate / TRADING_DAYS_PER_YEAR
    excess = returns - daily_rf
    rolling_mean = excess.rolling(window).mean() * TRADING_DAYS_PER_YEAR
    rolling_std = excess.rolling(window).std() * (TRADING_DAYS_PER_YEAR ** 0.5)
    return (rolling_mean / rolling_std).rename("rolling_sharpe")


def sortino_ratio(
    returns: pd.Series,
    risk_free_rate: float = 0.0,
    target_return: float = 0.0,
) -> float:
    """Compute the annualized Sortino ratio from a daily returns series.

    Uses downside deviation (returns below ``target_return``) as the risk
    measure instead of total standard deviation.

    Parameters:
        returns: Series of daily returns.
        risk_free_rate: Annualized risk-free rate (default 0).
        target_return: Minimum acceptable return threshold (default 0).

    Returns:
        Annualized Sortino ratio.
    """
    daily_rf = risk_free_rate / TRADING_DAYS_PER_YEAR
    excess = returns - daily_rf
    downside = excess[excess < target_return]
    downside_std = downside.std() * (TRADING_DAYS_PER_YEAR ** 0.5)
    if downside_std == 0:
        return float("inf")
    return realized_returns(excess) / downside_std


def information_ratio(
    returns: pd.Series,
    benchmark_returns: pd.Series,
) -> float:
    """Compute the annualized Information Ratio.

    IR = annualized active return / annualized tracking error.

    Parameters:
        returns: Series of daily strategy returns.
        benchmark_returns: Series of daily benchmark returns.

    Returns:
        Annualized Information Ratio.
    """
    active = returns - benchmark_returns
    return realized_returns(active) / realized_volatility(active)


def hit_rate(returns: pd.Series) -> float:
    """Compute the hit rate (percentage of positive return days).

    Parameters:
        returns: Series of daily returns.

    Returns:
        Fraction of days with positive returns (0–1).
    """
    return (returns > 0).mean()


def metrics_from_returns(r: pd.Series, label: str) -> dict:
    """Build a summary-metrics dict from a daily returns series.

    Parameters:
        r: Series of daily returns.
        label: Strategy name for the row.

    Returns:
        Dict with keys Strategy, Ann. Return (%), Ann. Volatility (%),
        Sharpe Ratio, Max Drawdown (%), Calmar Ratio, Avg Exposure.
    """
    return {
        "Strategy": label,
        "Ann. Return (%)": round(realized_returns(r) * 100, 2),
        "Ann. Volatility (%)": round(realized_volatility(r) * 100, 2),
        "Sharpe Ratio": round(sharpe_ratio(r), 3),
        "Max Drawdown (%)": round(max_drawdown(r) * 100, 2),
        "Calmar Ratio": round(calmar_ratio(r), 3),
        "Avg Exposure": "1.000",
    }


def period_metrics(returns: pd.Series, label: str) -> dict:
    """Compute extended period metrics for IS/OOS analysis.

    Parameters:
        returns: Series of daily returns.
        label: Period label for the row.

    Returns:
        Dict with keys Period, Ann. Return (%), Ann. Vol (%), Sharpe,
        Sortino, Max DD (%), Calmar, Hit Rate (%).
    """
    return {
        "Period": label,
        "Ann. Return (%)": round(realized_returns(returns) * 100, 2),
        "Ann. Vol (%)": round(realized_volatility(returns) * 100, 2),
        "Sharpe": round(sharpe_ratio(returns), 3),
        "Sortino": round(sortino_ratio(returns), 3),
        "Max DD (%)": round(max_drawdown(returns) * 100, 2),
        "Calmar": round(calmar_ratio(returns), 3),
        "Hit Rate (%)": round(hit_rate(returns) * 100, 1),
    }