"""dispersion_trading - Volatility trading infrastructure.

Package structure:
    data/           - Data loaders (OptionLoader, USRatesLoader, ...)
    pricing/        - Black-Scholes pricing and greeks
    metrics/        - Performance, volatility, distance metrics
    dispersion/     - Dispersion trade sizing logic (greek-neutral)
    trading/option_trade.py - Trade generation classes (OptionTrade, DeltaHedgedOptionTrade, ...)
    trading/backtest.py     - Strategy backtester
    trading/strategies.py   - Pre-defined option strategy leg specs
    trading/selection.py    - Option selection utilities
    specs.py                - TypedDicts: OptionLegSpec, DispersionLegSpec, ...
"""
