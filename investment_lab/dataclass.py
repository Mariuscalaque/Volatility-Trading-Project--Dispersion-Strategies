from typing import Literal, TypedDict

from investment_lab.dispersion.greek_sizing import GreekFlavor


class OptionLegSpec(TypedDict, total=False):
    day_to_expiry_target: int
    strike_target: float
    strike_col: Literal["strike", "moneyness", "delta"]
    call_or_put: Literal["C", "P"]
    weight: float
    leg_name: str
    rebal_week_day: list[int]


class VarianceSwapLegSpec(TypedDict, total=False):
    day_to_expiry_target: int
    strike_spacing: float | int
    weight: float
    rebal_week_day: list[int]


class DispersionLegSpec(TypedDict, total=False):
    index_ticker: str
    component_ticker: str
    day_to_expiry_target: int
    greek: GreekFlavor
    base_notional: float
    rebal_week_day: list[int]
