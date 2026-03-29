import logging
from typing import Optional

import pandas as pd


def check_is_true(condition: bool, message: Optional[str] = None) -> None:
    if not condition:
        raise ValueError(message or "Condition is not true.")


def ffill_options_data(df: pd.DataFrame) -> pd.DataFrame:
    """Forward fill option data based on dates and id.

    Args:
        df (pd.DataFrame): dataframe of options containing at least
        the columns: date and option_id

    Returns:
        pd.DataFrame: forward-filled dataframe
    """
    missing_cols = set(["option_id", "date"]).difference(df.columns)
    check_is_true(
        len(missing_cols) == 0, f"Data is missing required columns: {missing_cols}"
    )
    logging.info("Forward filling option data for df")

    df_sorted = df.sort_values(by=["option_id", "date"]).copy()

    # Forward-fill only the non-key columns within each option_id group.
    # This avoids groupby().apply() which drops option_id from columns
    # in pandas 2.x due to index absorption.
    key_cols = ["option_id", "date"]
    fill_cols = [c for c in df_sorted.columns if c not in key_cols]
    df_sorted[fill_cols] = df_sorted.groupby("option_id")[fill_cols].ffill()

    return df_sorted