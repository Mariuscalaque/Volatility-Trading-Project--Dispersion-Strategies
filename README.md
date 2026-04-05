# Dispersion Trading -- Theta/Gamma/Vega-Neutral Strategies on SPY/AAPL

Academic project carried out for the **Volatility Trading** course of the Master 272 -- Financial Engineering (Ingenierie Economique et Financiere), Universite Paris-Dauphine (PSL).

Forked from [BaptisteZloch/Dauphine-Lecture-Volatility](https://github.com/BaptisteZloch/Dauphine-Lecture-Volatility).

---

## Table of Contents

1. [Overview](#overview)
2. [Scope and Limitations](#scope-and-limitations)
3. [Trade Structure](#trade-structure)
4. [Greek-Neutral Sizing](#greek-neutral-sizing)
5. [Dynamic Allocation Overlay](#dynamic-allocation-overlay)
6. [Robustness and Validation](#robustness-and-validation)
7. [Architecture](#architecture)
8. [Repository Structure](#repository-structure)
9. [Data](#data)
10. [Installation](#installation)
11. [Usage](#usage)
12. [Key Empirical Findings](#key-empirical-findings)
13. [References](#references)
14. [License](#license)

---

## Overview

This repository implements and backtests a **long dispersion strategy** on US equity options over the period January 2020 -- December 2022. The strategy exploits the spread between index-level implied volatility and single-stock implied volatility by constructing a two-leg options portfolio:

- **Index leg (carry):** short an ATM straddle on SPY -- collects the index volatility premium.
- **Component leg (dispersion):** long an ATM straddle on AAPL -- pays the single-name volatility premium.

The notional of the component leg is adjusted daily so that the combined portfolio is neutral in one of three Greek dimensions: theta, dollar-gamma, or vega. Each leg is independently delta-hedged in the underlying shares.

The backtest engine computes daily mark-to-market P&L decomposed into delta, gamma, theta, vega, and residual components, tracks the cumulative NAV, and accounts for realistic transaction costs via bid-ask spreads from historical option data.

Beyond the static strategy, the project implements a **dynamic timing overlay** based on an implied-minus-realized correlation proxy. The overlay modulates position exposure using an expanding-window z-score, and is evaluated in a strictly walk-forward framework to avoid look-ahead bias.

---

## Scope and Limitations

This project is an academic study, not a production-grade correlation trading engine.

- The strategy trades a **single component** (AAPL) against the index (SPY). It does not reconstruct the full index basket or compute true average implied correlation from the full set of constituents.
- The "implied correlation" quantities reported in the notebook are **single-name proxies** derived from a simplified two-component variance decomposition. They should not be interpreted as the actual implied correlation surface of the S&P 500 index.
- The data covers only the January 2020 -- December 2022 window, which includes several unusual regimes (COVID-19 crash, post-pandemic recovery, 2022 rate-hiking drawdown).
- Option data is provided as pre-computed Parquet files with daily granularity; intraday effects are not modeled.

---

## Trade Structure

### Static Dispersion Trade

On each weekly rebalancing date (Wednesdays), the strategy:

1. **Sells** an ATM straddle on SPY with approximately one month to expiration (28 calendar days). The straddle consists of an ATM put (delta ~ -0.50) and an ATM call (delta ~ +0.50), each weighted by -1/4 per week (accumulated over 4 weekly entries to build the full monthly exposure).

2. **Buys** an ATM straddle on AAPL with the same maturity profile. The initial weight is a placeholder; it is overwritten by the Greek-neutral sizing engine.

3. **Delta-hedges** each leg independently by taking an offsetting position in the underlying shares. The hedge quantity is the aggregate portfolio delta across all active entries for each underlier.

At expiration, options are settled at intrinsic value. Between rebalancing dates, positions drift with the market (drifted positions).

### P&L Decomposition

The daily P&L of each position is decomposed using a Taylor expansion of the Black-Scholes price:

```
PnL(t) = w(t) × [ delta(t-1) × dS + 0.5 × gamma(t-1) × dS^2 + theta(t-1) × dt + vega(t-1) × dsigma + residual ]
```

where `w(t)` is the NAV-scaled position weight, `dS` is the daily spot change, `dt` is the number of calendar days between trading dates, and `dsigma` is the change in implied volatility.

---

## Greek-Neutral Sizing

The core of the dispersion trade is the sizing of the component (long) leg relative to the index (short) leg. Three flavors are implemented:

### Theta-Neutral

The AAPL straddle notional is scaled so that the daily theta decay of the long leg offsets that of the short leg:

```
ratio(t) = |sum(w_SPY × theta_SPY)(t)| / |sum(theta_AAPL)(t)|
```

### Dollar-Gamma-Neutral

Raw gamma is not comparable across underliers with different spot prices (SPY ~ $450 vs. AAPL ~ $170). The sizing uses dollar-gamma instead:

```
dollar_gamma = gamma × S^2
ratio(t) = |sum(w_SPY × dollar_gamma_SPY)(t)| / |sum(dollar_gamma_AAPL)(t)|
```

### Vega-Neutral

The AAPL notional is scaled to match the aggregate vega exposure of the SPY leg:

```
ratio(t) = |sum(w_SPY × vega_SPY)(t)| / |sum(vega_AAPL)(t)|
```

In all three cases, the ratio is forward-filled across dates where the denominator is zero (e.g., near expiration) and capped at the 95th percentile to prevent extreme position sizes.

---

## Dynamic Allocation Overlay

The project builds a timing signal to modulate the static dispersion trade's exposure through time.

### Signal: Implied-Minus-Realized Correlation Proxy

A single-name implied correlation proxy is derived from a two-component variance decomposition:

```
rho_impl ~ (sigma_idx^2 - w^2 * sigma_cmp^2 - (1-w)^2 * sigma_idx^2) / (2 * w * (1-w) * sigma_cmp * sigma_idx)
```

where `w ~ 0.07` is AAPL's approximate weight in the S&P 500. The realized leg is a 30-day rolling Pearson correlation of daily returns. The signal is defined as `corr_spread = rho_impl - rho_realized`.

An alternative signal based on the variance risk premium spread (VRP_index - VRP_component) is also implemented, following Driessen et al. (2009) and Carr & Wu (2009).

### Exposure Construction

The raw signal is transformed into a position exposure in [0, 1] via one of three methods:

- **Binary:** exposure = 1 if z-score > threshold, else 0.
- **Continuous rank:** exposure = rolling percentile rank, rescaled to [0, 1].
- **Z-score clipped** (default): exposure = clip(z-score, 0, 1).

### Walk-Forward Evaluation

To avoid in-sample overfitting, the z-score is computed using an **expanding window** (all data from inception to date t). This means the signal parameters (mean, standard deviation) are never estimated with future data. The dynamic return series is:

```
r_dynamic(t) = exposure(t-1) × r_static(t)
```

following the standard timing backtest methodology of Moskowitz, Ooi & Pedersen (2012).

---

## Robustness and Validation

### Signal Analysis

- **Bucket analysis:** trading days are sorted into quintiles by signal level; the average forward P&L in each bucket is computed to check for monotonicity.
- **Forward correlation:** Pearson correlation between signal(t) and cumulative forward P&L at horizons of 1, 5, 10, and 21 days, with significance tests.
- **Conditional performance:** the strategy's return, volatility, Sharpe ratio, and drawdown are computed separately for "favorable" (signal > 0) and "unfavorable" (signal <= 0) regimes.

### Sensitivity Analysis

- **Transaction costs:** the baseline backtest includes 1x bid-ask spread; performance is re-evaluated at 1.5x, 2x, 2.5x, and 3x cost multipliers.
- **Signal threshold:** the binary exposure threshold is swept from -0.5 to 1.5; the clipped z-score is shifted by the same amounts.
- **Z-score window:** the expanding (or rolling) estimation window is varied across 21, 42, 63, 126, and 252 trading days.

### Permutation Test

A non-parametric test shuffles the signal dates 1,000 times (preserving the distribution of values), computes the walk-forward dynamic Sharpe for each permutation, and derives a p-value as the fraction of permuted Sharpes exceeding the actual Sharpe. This tests the null hypothesis that the signal carries no genuine timing information.

---

## Architecture

The codebase is organized as a Python library under `src/` with the following major modules:

| Module | Responsibility |
|---|---|
| `src/data/` | Data loading from Parquet files. Separate loaders for the SPY option universe, the AAPL option universe, and the US Treasury yield curve. |
| `src/pricing/` | Black-Scholes pricing and Greeks (delta, gamma, theta, vega, rho). Newton-Raphson implied volatility solver. |
| `src/trading/` | Option selection (closest strike and maturity to a target), trade generation (conversion to daily time series), delta hedging, and the backtest engine (NAV, P&L, drifted positions). |
| `src/dispersion/` | Dispersion-specific logic: trade orchestration, Greek-neutral sizing, dynamic allocation overlay, signal analysis, and robustness tests. |
| `src/metrics/` | Performance metrics: annualized return, Sharpe ratio, Sortino ratio, Calmar ratio, maximum drawdown, rolling Sharpe, information ratio, hit rate. Realized volatility computation. |
| `src/surface/` | Volatility surface models (SVI, SABR, SSVI) for the lecture material. Not used directly in the dispersion backtest. |
| `src/stochastic/` | Stochastic process simulation (geometric Brownian motion) for the lecture material. |

### Backtest Flow

```
run_dispersion_backtest(start, end, greek)
  |
  +-- generate_dispersion_trades()
  |     |
  |     +-- _SPYTrade._generate_trades()     --> load SPY options, select ATM straddle, build daily positions
  |     +-- _AAPLTrade._generate_trades()    --> load AAPL options, select ATM straddle, build daily positions
  |     +-- _prepare_sizing_column()         --> compute dollar_gamma if needed
  |     +-- apply_greek_neutral_sizing()     --> rescale AAPL weights for Greek neutrality
  |     +-- DeltaHedgedOptionTrade._hedge_trades()  --> delta-hedge SPY positions
  |     +-- DeltaHedgedOptionTrade._hedge_trades()  --> delta-hedge AAPL positions
  |     +-- concatenate both legs
  |
  +-- DispersionBacktester(positions)
        |
        +-- _preprocess_positions()  --> load option data per ticker, merge Greeks and prices
        +-- apply_tcost()            --> replace mid with bid/ask on trade entry/exit
        +-- compute_backtest()       --> daily P&L loop (decomposed), NAV accumulation
```

---

## Repository Structure

```text
.
├── data/
│   ├── aapl_2016_2023.parquet              # AAPL daily option chain (2016-2023)
│   ├── optiondb_2016_2023.parquet/         # Multi-ticker option database
│   ├── par-yield-curve-rates-2020-2023.csv # US Treasury par yield curve
│   ├── spy_2020_2022.parquet/              # SPY daily option chain (full)
│   ├── spy_2020_2022_atm.parquet           # SPY ATM subset
│   └── spy_2020_2022_dte90.parquet         # SPY options with DTE <= 90
├── lectures/
│   ├── Lecture_2.ipynb                     # Course lecture notebooks
│   ├── Lecture_3.ipynb
│   ├── Lecture_4.ipynb
│   └── Lecture_5.ipynb
├── notebooks/
│   └── Dispersion_Backtest_ Project.ipynb  # Main deliverable notebook
├── src/
│   ├── __init__.py
│   ├── constants.py                        # Global constants (TRADING_DAYS_PER_YEAR, etc.)
│   ├── rates.py                            # Forward rate computation from yield curve
│   ├── specs.py                            # TypedDict definitions for leg specifications
│   ├── util.py                             # Shared utilities (forward-fill, assertions)
│   ├── data/
│   │   ├── data_loader.py                  # Abstract Parquet data loader
│   │   ├── option_db.py                    # SPY, AAPL, and generic option loaders
│   │   └── rates_db.py                     # US Treasury yield curve loader
│   ├── dispersion/
│   │   ├── dispersion_trade.py             # Trade orchestrator (generate + backtest)
│   │   ├── greek_sizing.py                 # Theta/gamma-dollar/vega-neutral sizing
│   │   ├── dynamic_allocation.py           # Timing signals and walk-forward overlay
│   │   ├── robustness.py                   # Sensitivity analysis and permutation test
│   │   └── signal_analysis.py              # Bucket analysis, forward correlation, conditional perf.
│   ├── metrics/
│   │   ├── distance.py                     # Distance metrics (MSE, etc.)
│   │   ├── performance.py                  # Sharpe, Sortino, Calmar, drawdown, hit rate
│   │   ├── util.py                         # Returns-to-levels conversion
│   │   └── volatility.py                   # Realized volatility
│   ├── pricing/
│   │   ├── black_scholes.py                # Black-Scholes pricing, Greeks, implied vol (Newton-Raphson)
│   │   └── implied_volatility.py           # Implied volatility utilities
│   ├── stochastic/
│   │   ├── base.py                         # Base stochastic process
│   │   └── geometric_brownian_motion.py    # GBM simulation
│   ├── surface/
│   │   ├── base.py                         # Abstract volatility smoother
│   │   ├── sabr.py                         # SABR model
│   │   ├── ssvi.py                         # SSVI (Surface SVI)
│   │   └── svi.py                          # SVI parametrization (Gatheral)
│   └── trading/
│       ├── backtest.py                     # Backtest engine (NAV, P&L, drifted positions)
│       ├── option_trade.py                 # Trade generation, delta/gamma hedging
│       ├── selection.py                    # Option selection (strike, maturity)
│       └── strategies.py                   # Pre-defined strategy templates
├── test_bt.py                              # Smoke test (3 Greek flavors, Jan-Jun 2021)
├── pyproject.toml
├── requirements.txt
└── LICENSE
```

---

## Data

| File | Description | Period |
|---|---|---|
| `spy_2020_2022_dte90.parquet` | SPY daily option chain, filtered to DTE <= 90 days. Used as the primary index data source. | 2020-2022 |
| `aapl_2016_2023.parquet` | AAPL daily option chain. | 2016-2023 |
| `optiondb_2016_2023.parquet/` | Multi-ticker option database used by the generic `OptionLoader`. | 2016-2023 |
| `par-yield-curve-rates-2020-2023.csv` | US Treasury par yield curve rates, used to compute risk-free rates and forward rates for Black-Scholes pricing. | 2020-2023 |

Each option record includes: date, ticker, option_id, expiration, strike, call/put flag, spot, bid, ask, mid, delta, gamma, theta, vega, implied volatility, and volume.

---

## Installation

**Prerequisites:** Python >= 3.10

```bash
git clone https://github.com/Mariuscalaque/Volatility-Trading-Project--Dispersion-Strategies.git
cd Volatility-Trading-Project--Dispersion-Strategies
python -m venv .venv
source .venv/bin/activate   # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

The `pywin32` dependency is declared as conditional (`sys_platform == "win32"`), so the same requirements file works on Linux, macOS, and in GitHub Codespaces.

---

## Usage

### Running the Main Notebook

The primary deliverable is the Jupyter notebook:

```
notebooks/Dispersion_Backtest_ Project.ipynb
```

It walks through all stages of the analysis:

1. Static backtests in theta-neutral, dollar-gamma-neutral, and vega-neutral configurations.
2. Greek neutrality verification (theta, gamma, vega residuals and delta hedge tracking).
3. Stress period diagnostics (COVID-19 March 2020, rate-hiking drawdowns late 2022).
4. Construction and analysis of the implied-minus-realized correlation proxy.
5. Walk-forward dynamic allocation (expanding z-score, no look-ahead).
6. Robustness: transaction cost sensitivity, signal threshold sensitivity, z-score window sensitivity, and permutation test.

### Running a Backtest Programmatically

```python
from datetime import datetime
from src.dispersion.dispersion_trade import run_dispersion_backtest

bt = run_dispersion_backtest(
    start_date=datetime(2020, 1, 2),
    end_date=datetime(2022, 12, 30),
    greek="theta",   # or "gamma" or "vega"
)

# Access results
bt.nav          # DataFrame with cumulative NAV
bt.pnl          # DataFrame with daily P&L (total, delta, gamma, theta, vega, residual)
bt.drifted_positions  # DataFrame with all drifted daily positions
```

### Smoke Test

To verify the entire pipeline runs correctly outside of Jupyter:

```bash
python test_bt.py
```

This script runs all three Greek-neutral variants on a reduced sample (January--June 2021) and validates that the NAV, P&L, and drifted positions are correctly produced with positive NAV and monotonically increasing date indices.

---

## Key Empirical Findings

- All three static dispersion variants (theta, gamma, vega-neutral) produce non-trivial P&L over the 2020-2022 period, with significant variation across stress regimes.
- The dynamic timing overlay based on the correlation spread proxy improves risk-adjusted performance only for the **theta-neutral variant**, and only when evaluated in the **walk-forward** (expanding z-score) framework.
- In-sample z-score overlays appear to improve performance across all variants, but this improvement does not survive out-of-sample walk-forward evaluation for the gamma and vega variants -- a clear illustration of overfitting risk.
- The permutation test provides a formal statistical check on the timing signal's predictive content.
- Transaction cost sensitivity analysis shows that results are moderately robust to 2x cost multipliers but degrade at 3x.

---

## References

- Driessen, J., Maenhout, P. J., & Vilkov, G. (2009). *The price of correlation risk: Evidence from equity options.* The Journal of Finance, 64(3), 1377-1406.
- Carr, P., & Wu, L. (2009). *Variance risk premiums.* The Review of Financial Studies, 22(3), 1311-1341.
- Gatheral, J. (2004). *A parsimonious arbitrage-free implied volatility parameterization with application to the valuation of volatility derivatives.* Presentation at Global Derivatives & Risk Management.

---

## License

This project is licensed under the **Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International** (CC BY-NC-SA 4.0) license. See the [LICENSE](LICENSE) file for details.
