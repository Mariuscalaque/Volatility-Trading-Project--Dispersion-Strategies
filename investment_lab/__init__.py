"""investment_lab - Volatility trading infrastructure.

Package structure:
    data/           - Data loaders (OptionLoader, USRatesLoader, ...)
    pricing/        - Black-Scholes pricing and greeks
    metrics/        - Performance, volatility, distance metrics
    dispersion/     - Dispersion trade sizing logic (greek-neutral)
    option_trade.py - Trade generation classes (OptionTrade, DeltaHedgedOptionTrade, ...)
    backtest.py     - Strategy backtester
    option_strategies.py - Pre-defined option strategy leg specs
    option_selection.py  - Option selection utilities
"""
