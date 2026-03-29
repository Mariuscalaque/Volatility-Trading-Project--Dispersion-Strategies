from typing import Literal

import numpy as np
import pandas as pd

GreekFlavor = Literal["theta", "gamma", "vega"]


def compute_greek_notional(
    df_leg: pd.DataFrame,
    greek: GreekFlavor,
    weight_col: str = "weight",
) -> pd.Series:
    """Compute the signed notional greek exposure of a leg on a given date.

    The greek columns available in df_leg are those from the option database:
    'theta', 'gamma', 'vega' (per contract, per unit of spot).
    The notional greek is weight * greek_per_contract, summed per date.

    Args:
        df_leg: DataFrame for a single leg, with at least columns
                ['date', greek, weight_col].
        greek: Greek to use for sizing ('theta', 'gamma', or 'vega').
        weight_col: Column containing the signed position size.

    Returns:
        Series indexed by date with the total notional greek of the leg.
    """
    notional = df_leg[weight_col] * df_leg[greek]
    return notional.groupby(df_leg["date"]).sum()


def compute_sizing_ratio(
    df_index_leg: pd.DataFrame,
    df_component_leg: pd.DataFrame,
    greek: GreekFlavor,
    index_weight_col: str = "weight",
    component_weight_col: str = "weight",
) -> pd.Series:
    """Compute the daily sizing ratio to apply to the component (long) leg.

    The ratio is defined so that the component leg greek notional matches
    the absolute greek notional of the index (short) leg:

        ratio(t) = |notional_greek_index(t)| / |greek_per_unit_component(t)|

    where greek_per_unit_component is the sum of the greek column over all
    options in the component leg (before any weight scaling, i.e. with
    weight = 1).

    Args:
        df_index_leg: DataFrame for the index leg (e.g. short SPY straddle),
                      with columns ['date', greek, index_weight_col].
        df_component_leg: DataFrame for the component leg (e.g. long AAPL straddle),
                          with columns ['date', greek, component_weight_col].
        greek: Greek flavor to neutralize.
        index_weight_col: Weight column in df_index_leg.
        component_weight_col: Weight column in df_component_leg.

    Returns:
        Series indexed by date with the scaling factor to apply to the
        component leg weight.
    """
    notional_index = compute_greek_notional(df_index_leg, greek, index_weight_col)

    greek_per_unit = df_component_leg[greek].groupby(df_component_leg["date"]).sum()

    numerator = notional_index.abs()
    denominator = greek_per_unit.abs()

    ratio = np.where(denominator != 0, numerator / denominator, np.nan)
    ratio = pd.Series(ratio, index=numerator.index)
    # Forward-fill to carry the last valid ratio forward when the denominator
    # is zero on isolated dates.  Callers should filter or cap the result if
    # stale propagation over many periods is undesirable.
    ratio = ratio.ffill()

    return ratio


def apply_greek_neutral_sizing(
    df_index_leg: pd.DataFrame,
    df_component_leg: pd.DataFrame,
    greek: GreekFlavor,
    base_notional: float,
    index_weight_col: str = "weight",
    component_weight_col: str = "weight",
    ratio_cap: float | None = None,
) -> pd.DataFrame:
    """Return df_component_leg with its weight column rescaled for greek neutrality.

    The rescaled weight on each date is:
        w_component(t) = +1 * ratio(t)

    where ratio(t) comes from compute_sizing_ratio and the sign is +1
    (the component leg is always long volatility in a long dispersion trade).

    The base_notional parameter sets the notional of the index leg used
    as reference when the ratio cannot be computed (e.g. zero greek).

    Args:
        df_index_leg: DataFrame for the index (short) leg.
        df_component_leg: DataFrame for the component (long) leg.
        greek: Greek flavor.
        base_notional: Fallback sizing ratio applied to the component leg on
                       dates where the ratio is undefined (e.g. zero greek in
                       the denominator).  It is expressed in the same units as
                       the computed ratio (absolute greek notional of the index
                       leg divided by greek per unit of the component leg).
                       A sensible default is the notional of the index leg.
        index_weight_col: Weight column in df_index_leg.
        component_weight_col: Weight column in df_component_leg.
        ratio_cap: Maximum allowed sizing ratio. When the ratio exceeds this
                   cap (e.g. because the component greek per unit is near zero),
                   it is clipped to prevent extreme position sizes. If None,
                   defaults to the 95th percentile of the ratio distribution.

    Returns:
        Copy of df_component_leg with updated weight column.
    """
    ratio = compute_sizing_ratio(
        df_index_leg,
        df_component_leg,
        greek,
        index_weight_col=index_weight_col,
        component_weight_col=component_weight_col,
    )

    ratio = ratio.fillna(base_notional)

    # Cap extreme ratios to prevent position blow-ups (e.g. when the
    # component greek per unit is near zero around option expiry).
    # The 95th percentile adapts to each greek flavor's natural scale:
    # theta/gamma ratios are O(1) while vega ratios are O(0.001) because
    # the index notional in the numerator carries the /spot normalization.
    if ratio_cap is None:
        ratio_cap = ratio.quantile(0.95)
    ratio = ratio.clip(upper=ratio_cap)

    # The ratio replaces the component weight entirely.  This is correct
    # because:
    #   - The numerator (index notional) already includes the index weights
    #     (which carry the /spot normalization from _select_options).
    #   - The denominator is the *unweighted* sum of per-contract greeks for
    #     the component straddle.
    # Thus ratio × Σ(greek_component) = |notional_greek_index|, ensuring
    # greek neutrality.  The sign is always positive (long vol on component).
    df_result = df_component_leg.copy()
    date_to_ratio = ratio.reindex(df_result["date"])
    date_to_ratio.index = df_result.index
    df_result[component_weight_col] = date_to_ratio.values

    return df_result