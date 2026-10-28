"""Load a generator fleet and compute marginal costs."""
import pandas as pd

DEFAULT_FUEL_PRICES = {  # $/MMBtu, override with real EIA prices
    "gas": 3.00,
    "coal": 2.10,
    "oil": 15.00,
    "nuclear": 0.70,
    "wind": 0.0,
    "solar": 0.0,
    "hydro": 0.0,
}


def load_fleet(path, fuel_prices=None):
    """Read a fleet CSV and add a marginal_cost column in $/MWh.

    marginal_cost = heat_rate (MMBtu/MWh) * fuel_price ($/MMBtu) + vom ($/MWh)
    """
    prices = {**DEFAULT_FUEL_PRICES, **(fuel_prices or {})}
    df = pd.read_csv(path)
    required = {"unit", "fuel", "capacity_mw", "heat_rate", "vom"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"fleet file missing columns: {missing}")
    df["fuel_price"] = df["fuel"].map(prices)
    if df["fuel_price"].isna().any():
        bad = df.loc[df["fuel_price"].isna(), "fuel"].unique()
        raise ValueError(f"no fuel price for: {bad}")
    df["marginal_cost"] = df["heat_rate"] * df["fuel_price"] + df["vom"]
    return df
