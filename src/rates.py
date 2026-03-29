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
    """Linearly interpolate the rate curve at a given tenor.

    Flat extrapolation is used outside the range of known tenors
    (i.e. the nearest boundary rate is returned).

    Args:
        eval_tenor: The tenor (in year fractions) to evaluate.
        tenors: Known tenor points (need not be sorted).
        rate_curve: Corresponding rates for each tenor.

    Returns:
        The interpolated (or extrapolated) rate.
    """
    tenors = np.asarray(tenors, dtype=float)
    rate_curve = np.asarray(rate_curve, dtype=float)
    check_is_true(
        len(tenors) == len(rate_curve),
        "Tenors and rate curve must have the same length.",
    )

    # Sort by tenor for correct interpolation
    sort_idx = np.argsort(tenors)
    tenors_sorted = tenors[sort_idx]
    rates_sorted = rate_curve[sort_idx]

    # np.interp handles exact matches, boundary extrapolation (flat),
    # and intermediate interpolation in a single vectorised call.
    return float(np.interp(eval_tenor, tenors_sorted, rates_sorted))