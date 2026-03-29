# Dispersion Trading — θ/Γ/ν-neutral SPY/AAPL

A volatility dispersion trading strategy implemented in Python.

## Strategy Overview

This project implements a long dispersion trade consisting of two legs:
- **Short volatility on SPY** (index leg): short ATM straddle, carry-type exposure
- **Long volatility on AAPL** (component leg): long ATM straddle, long gamma exposure

The strategy has a short correlation exposure, which provides a natural hedge during
market stress. Both legs are delta-neutral, with dynamic delta hedging throughout the
trade lifecycle.

Three sizing flavors are implemented to match the greek notional between the two legs:
- **θ-neutral**: theta notional of the short SPY leg matches the long AAPL leg
- **Γ-neutral**: gamma notional matching
- **ν-neutral**: vega notional matching

## Project Structure

```
.
├── data/                          # Parquet data files (options, rates)
├── lectures/                      # Course reference notebooks (Lecture 2–5)
├── notebooks/                     # Project notebooks
│   ├── 01_data_exploration.ipynb
│   ├── 02_single_leg_delta_hedging.ipynb
│   └── 03_dispersion_backtest.ipynb
└── src/
    ├── constants.py           # Trading constants (days/year, tenor mapping)
    ├── rates.py               # Risk-free rate interpolation, forward computation
    ├── specs.py               # TypedDicts: OptionLegSpec, DispersionLegSpec, ...
    ├── util.py                # Shared utilities
    ├── data/                  # Data loaders (options DB, rates DB)
    ├── dispersion/            # Greek-neutral sizing logic (θ/Γ/ν flavors)
    ├── metrics/               # Performance, volatility, distance metrics
    ├── pricing/               # Black-Scholes pricing and Greeks
    ├── stochastic/            # Stochastic process base classes (GBM)
    ├── surface/               # Volatility surface models (SVI, SSVI, SABR)
    └── trading/               # Trade generation and backtesting
        ├── backtest.py        # StrategyBacktester, BacktesterBidAskFromData
        ├── option_trade.py    # OptionTrade, DeltaHedgedOptionTrade, ...
        ├── selection.py       # Option selection utilities
        └── strategies.py     # Pre-defined option strategy leg specs
```

## Installation

```bash
pip install -e .
```

## Running the Notebooks

Launch Jupyter from the repository root:

```bash
jupyter notebook notebooks/
```

- `01_data_exploration.ipynb` — explore the options data, bid-ask spreads, and implied vols
- `02_single_leg_delta_hedging.ipynb` — single-leg straddle with delta hedging backtest
- `03_dispersion_backtest.ipynb` — full dispersion backtest with all three greek-neutral flavors
