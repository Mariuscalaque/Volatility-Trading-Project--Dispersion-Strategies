"""Volatility trading infrastructure - flat src layout.

Package structure:
    data/           - Data loaders (OptionLoader, USRatesLoader, ...)
    pricing/        - Black-Scholes pricing and greeks
    metrics/        - Performance, volatility, distance metrics
    dispersion/     - Dispersion trade sizing logic (greek-neutral)
    trading/        - Trade generation, backtesting, selection and strategies
    stochastic/     - Stochastic process models (GBM, ...)
    surface/        - Volatility surface models (SVI, SSVI, SABR)
    constants.py    - Shared constants
    rates.py        - Rate utilities
    util.py         - Shared utilities
    specs.py        - TypedDicts: OptionLegSpec, DispersionLegSpec, ...
"""
