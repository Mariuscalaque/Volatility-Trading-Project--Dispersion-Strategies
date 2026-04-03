import sys; sys.path.insert(0, '.')
import gc
from datetime import datetime
from src.dispersion.dispersion_trade import run_dispersion_backtest

for greek in ["theta", "gamma", "vega"]:
    print(f"\n=== {greek} ===", flush=True)
    bt = run_dispersion_backtest(datetime(2020,9,1), datetime(2022,12,30), greek=greek)
    print(f"NAV: {bt.nav.shape}, PNL: {bt.pnl.shape}", flush=True)
    del bt; gc.collect()

print("\nALL 3 FLAVORS OK", flush=True)
