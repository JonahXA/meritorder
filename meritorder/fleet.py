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
    "waste": 0.0,  # byproduct / biomass, must-run price takers
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


def monthly_marginal_costs(df, fuel_prices):
    """Marginal cost per unit per month from a state x fuel x month price table.

    fuel_prices: DataFrame with month, state, fuel, price (scripts/fetch_fuel_prices.py).
    Fuels without a series there (nuclear, renewables, waste) keep the
    defaults. Returns {month: array aligned with df rows}.
    """
    out = {}
    for month, p in fuel_prices.groupby("month"):
        price = df.merge(p[["state", "fuel", "price"]], on=["state", "fuel"], how="left")["price"]
        price = price.fillna(df["fuel"].map(DEFAULT_FUEL_PRICES)).values
        out[month] = df["heat_rate"].values * price + df["vom"].values
    return out
