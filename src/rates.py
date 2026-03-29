import numpy as np
import pandas as pd

from src.constants import DAYS_PER_YEAR, TENOR_TO_PERIOD
from src.util import check_is_true


def compute_forward(df_options: pd.DataFrame, df_rates: pd.DataFrame) -> pd.DataFrame:
    """Compute risk free rates and forward price given daily options and daily rate curve.

    Args:
        df_options: Daily options data.
        df_rates: Daily rate curve with columns matching TENOR_TO_PERIOD keys.

    Returns:
        Same as input with 2 additional columns: 'risk_free_rate' and 'forward'.
    """
    missing_cols = set(TENOR_TO_PERIOD.keys()).difference(df_rates.columns)
    check_is_true(
        len(missing_cols) == 0, f"df_rates is missing columns: {missing_cols}"
    )
    missing_options_cols = {
        "date",
        "spot",
        "day_to_expiration",
        "option_id",
    }.difference(df_options.columns)
    check_is_true(
        len(missing_options_cols) == 0,
        f"df_options is missing columns: {missing_options_cols}",
    )

    df = df_options.merge(df_rates, on="date", how="left")

    # Compute interpolated risk-free rate for each (date, expiration) group.
    # Uses direct index assignment instead of groupby().apply() to avoid
    # pandas 2.x bug where groupby columns get absorbed into the index.
    tenor_keys = list(TENOR_TO_PERIOD.keys())
    tenors = pd.Index(tenor_keys).map(TENOR_TO_PERIOD).to_numpy()
    df["risk_free_rate"] = np.nan

    for (_, _), idx in df.groupby(["date", "expiration"]).groups.items():
        sub = df.loc[idx]
        dte = sub["day_to_expiration"].iloc[0] / DAYS_PER_YEAR
        rate_curve = sub[tenor_keys].iloc[0].to_numpy().astype(float)
        interpolated_rate = interpolate_rates(dte, tenors=tenors, rate_curve=rate_curve)
        df.loc[idx, "risk_free_rate"] = interpolated_rate

    df["forward"] = df["spot"] * np.exp(
        df["risk_free_rate"] * df["day_to_expiration"] / DAYS_PER_YEAR
    )
    df_forward = (
        df.groupby(["ticker", "date", "expiration"])[["forward"]]
        .first()
        .ffill()
        .reset_index()
    )
    return df.drop(columns=tenor_keys + ["forward"]).merge(
        df_forward, how="left", on=["ticker", "date", "expiration"]
    )


def interpolate_rates(
    eval_tenor: float,
    tenors: pd.Series | np.ndarray,
    rate_curve: pd.Series | np.ndarray,
) -> float:
    """Interpolate rates linearly.

    Args:
        eval_tenor (float): The tenor to evaluate the rate at.
        tenors (pd.Series | np.ndarray): The known tenors.
        rate_curve (pd.Series | np.ndarray): The known rates.
    Returns:
        The interpolated rate.
    """
    tenors = np.asarray(tenors)
    rate_curve = np.asarray(rate_curve)
    check_is_true(
        len(tenors) == len(rate_curve),
        "Tenors and rate curve must have the same length.",
    )
    if eval_tenor <= tenors.min():
        return rate_curve[tenors.argmin()]
    if eval_tenor >= tenors.max():
        return rate_curve[tenors.argmax()]

    idx_above = tenors[tenors >= eval_tenor].argmin()
    idx_below = tenors[tenors <= eval_tenor].argmax()

    tenor_above, tenor_below = tenors[idx_above], tenors[idx_below]
    rate_above, rate_below = rate_curve[idx_above], rate_curve[idx_below]

    weight_above = (eval_tenor - tenor_below) / (tenor_above - tenor_below)
    weight_below = 1 - weight_above

    return weight_below * rate_below + weight_above * rate_above