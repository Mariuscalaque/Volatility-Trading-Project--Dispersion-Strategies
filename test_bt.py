import gc
import sys
from datetime import datetime

sys.path.insert(0, ".")

from src.dispersion.dispersion_trade import run_dispersion_backtest


START_DATE = datetime(2021, 1, 4)
END_DATE = datetime(2021, 6, 30)
GREEKS = ("theta", "gamma", "vega")


def validate_backtest(backtest, greek: str) -> dict:
    nav = backtest.nav["NAV"].dropna()
    pnl = backtest.pnl["pnl"].dropna()
    positions = backtest.drifted_positions

    assert not nav.empty, f"{greek}: NAV is empty"
    assert not pnl.empty, f"{greek}: PnL is empty"
    assert not positions.empty, f"{greek}: drifted positions are empty"
    assert nav.index.is_monotonic_increasing, f"{greek}: NAV index is not sorted"
    assert pnl.index.is_monotonic_increasing, f"{greek}: PnL index is not sorted"
    assert (nav > 0).all(), f"{greek}: NAV contains non-positive values"
    assert positions["ticker"].isin(["SPY", "AAPL"]).any(), f"{greek}: missing core tickers"

    return {
        "greek": greek,
        "nav_obs": len(nav),
        "pnl_obs": len(pnl),
        "final_nav": float(nav.iloc[-1]),
        "dates": f"{nav.index.min().date()} -> {nav.index.max().date()}",
    }


def main() -> None:
    summaries = []

    for greek in GREEKS:
        print(f"\n=== {greek} ===", flush=True)
        backtest = run_dispersion_backtest(START_DATE, END_DATE, greek=greek)
        summary = validate_backtest(backtest, greek)
        summaries.append(summary)
        print(
            f"NAV obs: {summary['nav_obs']}, PnL obs: {summary['pnl_obs']}, "
            f"final NAV: {summary['final_nav']:.6f}, dates: {summary['dates']}",
            flush=True,
        )
        del backtest
        gc.collect()

    print("\nALL 3 FLAVORS OK", flush=True)


if __name__ == "__main__":
    main()
