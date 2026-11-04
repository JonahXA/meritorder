"""Monthly delivered fuel prices to power plants, by state, from EIA.

    python scripts/fetch_fuel_prices.py --year 2024

Writes data/fuel_prices_{year}.csv: month, state, fuel, price ($/MMBtu), source.

Route: electricity/electric-power-operational-data, cost-per-btu. EIA withholds
many state-months for the electric power sector total (sectorid 98), e.g.
Arkansas gas, Indiana coal, but publishes other sector cuts. Take the first
available of: 98 electric power, 99 all sectors, 1 electric utilities,
94 independent power producers. Anything still missing falls back to the
mean across MISO states for that fuel and month.
Nuclear has no EIA price series and keeps the fleet.py default.
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fetch_eia import api_key, get_all, cached  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
EIA_FUEL = {"NG": "gas", "COW": "coal", "PEL": "oil"}
SECTORS = ["98", "99", "1", "94"]       # priority order


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--year", type=int, default=2024)
    args = ap.parse_args()
    y = args.year
    states = sorted(pd.read_csv(ROOT / "data/miso_fleet.csv")["state"].unique())

    raw = cached(f"fuel_cost_{y}", lambda: get_all(
        "electricity/electric-power-operational-data",
        {"frequency": "monthly", "data[]": ["cost-per-btu"], "facets[location][]": states,
         "facets[sectorid][]": SECTORS, "facets[fueltypeid][]": list(EIA_FUEL),
         "start": f"{y}-01", "end": f"{y}-12"}, api_key()))
    raw["price"] = pd.to_numeric(raw["cost-per-btu"], errors="coerce")
    raw["fuel"] = raw["fueltypeid"].map(EIA_FUEL)
    raw = raw.rename(columns={"period": "month", "location": "state"}).dropna(subset=["price"])
    raw["rank"] = raw["sectorid"].astype(str).map({s: i for i, s in enumerate(SECTORS)})
    raw = raw.sort_values("rank").drop_duplicates(["month", "state", "fuel"])

    grid = pd.MultiIndex.from_product(
        [[f"{y}-{m:02d}" for m in range(1, 13)], states, list(EIA_FUEL.values())],
        names=["month", "state", "fuel"]).to_frame(index=False)
    out = grid.merge(raw[["month", "state", "fuel", "price", "sectorid"]], how="left")
    out["source"] = "eia_sector_" + out["sectorid"].astype(str)
    out = out.drop(columns="sectorid")
    miss = out["price"].isna()
    fallback = out.groupby(["month", "fuel"])["price"].transform("mean")
    out.loc[miss, "price"] = fallback[miss]
    out.loc[miss, "source"] = "miso_states_mean"
    out.to_csv(ROOT / f"data/fuel_prices_{y}.csv", index=False)

    print(f"Wrote {len(out)} rows. Source counts:", out.source.value_counts().to_dict())
    print(out[out.state == "MI"].pivot(index="month", columns="fuel", values="price").round(2).to_string())
    print("\nMISO-states mean by fuel:", out.groupby("fuel").price.mean().round(2).to_dict())


if __name__ == "__main__":
    main()
