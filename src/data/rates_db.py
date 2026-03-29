import pathlib
from datetime import datetime

import pandas as pd

from src.data.data_loader import DataLoader

# Resolve project root from this file's location: src/data/rates_db.py -> project root
ROOT_PATH = pathlib.Path(__file__).resolve().parents[2]


class USRatesLoader(DataLoader):
    @classmethod
    def _get_path(cls) -> str:
        return str(ROOT_PATH / "data" / "par-yield-curve-rates-2020-2023.csv")

    @classmethod
    def _get_valid_date_range(cls) -> tuple[datetime, datetime]:
        return (datetime(2020, 1, 2), datetime(2023, 12, 30))

    @classmethod
    def _process_loaded_data(cls, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        df["date"] = pd.to_datetime(df["date"], format="mixed")
        df = df.ffill().set_index("date").sort_index() / 100
        return df.reset_index()