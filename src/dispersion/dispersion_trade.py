"""Dispersion trade orchestrator.

Implements a long dispersion trade:
- Short volatility on the index (carry leg, e.g. short straddle on SPY)
- Long volatility on a component (dispersion leg, e.g. long straddle on AAPL)
- Greek-neutral sizing between the two legs (θ, Γ-dollar, or ν)
- Independent delta hedging per underlier

This module lives in ``src/dispersion/``.

Usage
-----
>>> from src.dispersion.dispersion_trade import run_dispersion_backtest
>>> bt = run_dispersion_backtest(
...     start_date=datetime(2020, 1, 2),
...     end_date=datetime(2022, 12, 30),
...     greek="theta",
... )
>>> bt.pnl  # daily P&L DataFrame
>>> bt.nav  # cumulative NAV DataFrame
"""

import logging
from datetime import datetime
from typing import Literal

import numpy as np
import pandas as pd

from src.data.option_db import SPYOptionLoader, AAPLOptionLoader
from src.data.rates_db import USRatesLoader
from src.specs import OptionLegSpec
from src.trading.option_trade import OptionTradeABC, DeltaHedgedOptionTrade
from src.trading.backtest import StrategyBacktester, BacktesterBidAskFromData
from src.util import ffill_options_data

# Relative import — greek_sizing is in the same package (src/dispersion/)
from .greek_sizing import apply_greek_neutral_sizing, GreekFlavor

# ---------------------------------------------------------------------------
# Internal trade generator classes — one per data source
# ---------------------------------------------------------------------------
# These bind the correct DataLoader (for the right parquet file) with the
# trade generation logic from OptionTradeABC.
# Python MRO ensures load_data() resolves to the DataLoader subclass first.


class _SPYTrade(SPYOptionLoader, OptionTradeABC):
    """SPY trade generator — reads from spy_2020_2022.parquet."""

    pass


class _AAPLTrade(AAPLOptionLoader, OptionTradeABC):
    """AAPL trade generator — reads from aapl_2016_2023.parquet."""

    pass


_TICKER_TO_TRADE_CLS: dict[str, type] = {
    "SPY": _SPYTrade,
    "AAPL": _AAPLTrade,
}

_TICKER_TO_LOADER_CLS: dict[str, type] = {
    "SPY": SPYOptionLoader,
    "AAPL": AAPLOptionLoader,
}

# ---------------------------------------------------------------------------
# Greek column resolution
# ---------------------------------------------------------------------------

DispersionGreek = Literal["theta", "gamma", "vega"]


def _prepare_sizing_column(
    df: pd.DataFrame, greek: DispersionGreek
) -> str:
    """Return the column name to use for greek-neutral sizing.

    For gamma-neutral, the raw gamma is not comparable across underliers
    with different spot prices (SPY ~$450 vs AAPL ~$170).  We compute
    dollar-gamma = Γ × S² and size on that instead.

    For theta and vega, the raw column is used directly.

    Args:
        df: DataFrame with at least 'gamma' and 'spot' columns.
        greek: Which flavor to size on.

    Returns:
        Column name to pass to apply_greek_neutral_sizing().
    """
    if greek == "gamma":
        df["dollar_gamma"] = df["gamma"] * df["spot"] ** 2
        return "dollar_gamma"
    return greek


# ---------------------------------------------------------------------------
# Default strategy definitions
# ---------------------------------------------------------------------------

# Index: short 1-month ATM straddle on SPY (carry leg)
INDEX_SHORT_STRADDLE_1M: list[OptionLegSpec] = [
    {
        "day_to_expiry_target": 7 * 4,
        "strike_target": -0.5,
        "strike_col": "delta",
        "call_or_put": "P",
        "weight": -1 / 4,
        "leg_name": "Short ATM Put SPY 1M",
        "rebal_week_day": [2],
    },
    {
        "day_to_expiry_target": 7 * 4,
        "strike_target": 0.5,
        "strike_col": "delta",
        "call_or_put": "C",
        "weight": -1 / 4,
        "leg_name": "Short ATM Call SPY 1M",
        "rebal_week_day": [2],
    },
]

# Component: long 1-month ATM straddle on AAPL (dispersion leg)
# Initial weight is a placeholder — it will be overwritten by greek sizing.
COMPONENT_LONG_STRADDLE_1M: list[OptionLegSpec] = [
    {
        "day_to_expiry_target": 7 * 4,
        "strike_target": -0.5,
        "strike_col": "delta",
        "call_or_put": "P",
        "weight": 1 / 4,
        "leg_name": "Long ATM Put AAPL 1M",
        "rebal_week_day": [2],
    },
    {
        "day_to_expiry_target": 7 * 4,
        "strike_target": 0.5,
        "strike_col": "delta",
        "call_or_put": "C",
        "weight": 1 / 4,
        "leg_name": "Long ATM Call AAPL 1M",
        "rebal_week_day": [2],
    },
]


# ---------------------------------------------------------------------------
# Core orchestrator
# ---------------------------------------------------------------------------


def generate_dispersion_trades(
    start_date: datetime,
    end_date: datetime,
    index_ticker: str = "SPY",
    component_ticker: str = "AAPL",
    index_legs: list[OptionLegSpec] | None = None,
    component_legs: list[OptionLegSpec] | None = None,
    greek: DispersionGreek = "theta",
    base_notional: float = 1.0,
) -> pd.DataFrame:
    """Generate combined daily positions for a long dispersion trade.

    Pipeline
    --------
    1. Generate raw trades for each leg independently (with Greeks)
    2. Compute dollar-gamma if Γ-neutral flavor
    3. Apply greek-neutral sizing to rescale the component leg
    4. Delta-hedge each leg independently (SPY shares / AAPL shares)
    5. Concatenate into a single positions DataFrame

    Args:
        start_date: Backtest start date.
        end_date: Backtest end date.
        index_ticker: Index underlier (default "SPY").
        component_ticker: Component underlier (default "AAPL").
        index_legs: Leg specs for the index (short vol). Defaults to
                    short 1M ATM straddle.
        component_legs: Leg specs for the component (long vol). Defaults to
                        long 1M ATM straddle.
        greek: Greek flavor for sizing — "theta", "gamma", or "vega".
        base_notional: Fallback sizing ratio when greek ratio is undefined.

    Returns:
        DataFrame with columns:
            ['date', 'option_id', 'entry_date', 'leg_name', 'weight', 'ticker']
        Ready to pass to DispersionBacktester or StrategyBacktester.
    """
    index_legs = index_legs or INDEX_SHORT_STRADDLE_1M
    component_legs = component_legs or COMPONENT_LONG_STRADDLE_1M

    index_cls = _TICKER_TO_TRADE_CLS[index_ticker]
    component_cls = _TICKER_TO_TRADE_CLS[component_ticker]

    # ── Step 1: generate raw trades with Greeks (before hedge) ──
    logging.info("=== DISPERSION [%s-neutral]: Generating index leg (%s) ===", greek, index_ticker)
    df_index = index_cls._generate_trades(
        start_date, end_date, tickers=index_ticker, legs=index_legs
    )

    logging.info("=== DISPERSION [%s-neutral]: Generating component leg (%s) ===", greek, component_ticker)
    df_component = component_cls._generate_trades(
        start_date, end_date, tickers=component_ticker, legs=component_legs
    )

    # Align on common trading dates
    common_dates = set(df_index["date"].unique()) & set(df_component["date"].unique())
    df_index = df_index[df_index["date"].isin(common_dates)].copy()
    df_component = df_component[df_component["date"].isin(common_dates)].copy()
    logging.info(
        "Common trading dates: %d (%s → %s)",
        len(common_dates),
        min(common_dates).date(),
        max(common_dates).date(),
    )

    # ── Step 2: resolve sizing column (dollar_gamma for Γ) ──
    sizing_col = _prepare_sizing_column(df_index, greek)
    _prepare_sizing_column(df_component, greek)
    logging.info("Sizing column: '%s'", sizing_col)

    # ── Step 3: greek-neutral sizing on component leg ──
    logging.info("=== DISPERSION: Applying %s-neutral sizing ===", greek)
    df_component_sized = apply_greek_neutral_sizing(
        df_index_leg=df_index,
        df_component_leg=df_component,
        greek=sizing_col,
        base_notional=base_notional,
    )

    # ── Step 4: delta-hedge each leg independently ──
    logging.info("=== DISPERSION: Delta hedging %s ===", index_ticker)
    df_index_hedged = DeltaHedgedOptionTrade._hedge_trades(df_index)

    logging.info("=== DISPERSION: Delta hedging %s ===", component_ticker)
    df_component_hedged = DeltaHedgedOptionTrade._hedge_trades(df_component_sized)

    # ── Step 5: combine positions ──
    output_cols = ["date", "option_id", "entry_date", "leg_name", "weight", "ticker"]
    df_combined = pd.concat(
        [df_index_hedged[output_cols], df_component_hedged[output_cols]],
        ignore_index=True,
    )

    logging.info(
        "=== DISPERSION: %d total position rows (%s + %s) ===",
        len(df_combined), index_ticker, component_ticker,
    )
    return df_combined.sort_values(["date", "ticker", "leg_name"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Multi-source backtester
# ---------------------------------------------------------------------------


class DispersionBacktester(BacktesterBidAskFromData):
    """Backtester for dispersion trades spanning multiple data sources.

    The standard ``StrategyBacktester._preprocess_positions`` loads option
    data from a single source via ``OptionLoader.load_data``.  A dispersion
    trade has positions on SPY *and* AAPL, which live in separate parquet
    files.  This subclass overrides the data loading to handle both.
    """

    @staticmethod
    def _preprocess_positions(df_positions: pd.DataFrame) -> pd.DataFrame:
        logging.info("DispersionBacktester: multi-source data loading.")
        df_positions_cp = df_positions.copy()
        start = df_positions_cp["date"].min()
        end = df_positions_cp["date"].max()
        tickers = df_positions_cp["ticker"].unique().tolist()

        # Load option data from the right parquet for each ticker
        dfs_options = []
        for ticker in tickers:
            loader_cls = _TICKER_TO_LOADER_CLS.get(ticker)
            if loader_cls is None:
                logging.warning("No dedicated loader for %s, skipping.", ticker)
                continue
            logging.info("Loading %s via %s", ticker, loader_cls.__name__)
            df_t = loader_cls.load_data(start, end, process_kwargs={"ticker": ticker})
            dfs_options.append(df_t)

        df_options = pd.concat(dfs_options, ignore_index=True)

        # Synthetic spot rows for delta-hedge positions (option_id == ticker).
        # Uses first() aggregation instead of groupby().apply(pd.Series)
        # to avoid pandas 2.x dropping groupby columns inside the lambda.
        df_spot = (
            df_options.groupby(["date", "ticker"])[["spot"]]
            .first()
            .reset_index()
        )
        df_spot["option_id"] = df_spot["ticker"]
        df_spot["bid"] = df_spot["spot"]
        df_spot["ask"] = df_spot["spot"]
        df_spot["mid"] = df_spot["spot"]
        # Stocks have trivial greeks: delta=1, no gamma/theta/vega/rho.
        df_spot["delta"] = 1
        df_spot["gamma"] = 0.0
        df_spot["theta"] = 0.0
        df_spot["vega"] = 0.0
        df_spot["implied_volatility"] = 0.0
        df_options_spot = pd.concat([df_options, df_spot], ignore_index=True)

        # Merge positions with option data
        df_extended = df_positions_cp.merge(
            df_options_spot, how="left", on=["ticker", "option_id", "date"]
        )
        # Do not trade after expiration
        df_extended = df_extended[
            (df_extended["date"] <= df_extended["expiration"])
            | df_extended["expiration"].isna()
        ]
        return ffill_options_data(df_extended)


# ---------------------------------------------------------------------------
# Convenience: end-to-end dispersion backtest
# ---------------------------------------------------------------------------


def run_dispersion_backtest(
    start_date: datetime,
    end_date: datetime,
    index_ticker: str = "SPY",
    component_ticker: str = "AAPL",
    index_legs: list[OptionLegSpec] | None = None,
    component_legs: list[OptionLegSpec] | None = None,
    greek: DispersionGreek = "theta",
    base_notional: float = 1.0,
) -> DispersionBacktester:
    """End-to-end: generate trades → run backtest → return fitted backtester.

    Example
    -------
    >>> bt_theta = run_dispersion_backtest(
    ...     datetime(2020,1,2), datetime(2022,12,30), greek="theta"
    ... )
    >>> bt_gamma = run_dispersion_backtest(
    ...     datetime(2020,1,2), datetime(2022,12,30), greek="gamma"
    ... )
    >>> # Compare: bt_theta.pnl vs bt_gamma.pnl

    Args:
        start_date: Backtest start.
        end_date: Backtest end.
        index_ticker: Index ticker (default "SPY").
        component_ticker: Component ticker (default "AAPL").
        index_legs: Index strategy. Defaults to short 1M ATM straddle.
        component_legs: Component strategy. Defaults to long 1M ATM straddle.
        greek: Sizing flavor — "theta", "gamma", or "vega".
        base_notional: Fallback notional.

    Returns:
        Fitted DispersionBacktester with accessible .pnl, .nav, .drifted_positions.
    """
    logging.info(
        "╔══════════════════════════════════════════════════╗\n"
        "║  DISPERSION BACKTEST: %s-neutral                ║\n"
        "║  %s (short) vs %s (long)                        ║\n"
        "║  %s → %s                                        ║\n"
        "╚══════════════════════════════════════════════════╝",
        greek, index_ticker, component_ticker,
        start_date.date(), end_date.date(),
    )

    df_positions = generate_dispersion_trades(
        start_date=start_date,
        end_date=end_date,
        index_ticker=index_ticker,
        component_ticker=component_ticker,
        index_legs=index_legs,
        component_legs=component_legs,
        greek=greek,
        base_notional=base_notional,
    )

    bt = DispersionBacktester(df_positions)
    bt.compute_backtest()
    return bt