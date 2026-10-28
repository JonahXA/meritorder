"""Build a real Michigan fleet from the EIA API (v2).

Needs a free key: https://www.eia.gov/opendata/register.php
    export EIA_API_KEY=...   (or EIA_API_KEY=... in .env)
    python scripts/fetch_eia.py --year 2024

Writes data/mi_fleet.csv in the same schema as data/sample_fleet.csv.
Raw pulls are cached in data/raw/ (gitignored).

Note: route and field names follow EIA API v2 docs. Check them against
https://www.eia.gov/opendata/browser/ on first run since EIA renames fields
occasionally.
"""
import argparse
import json
import os
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data/raw"
BASE = "https://api.eia.gov/v2"

# EIA energy source codes -> our fuel buckets
FUEL_MAP = {
    "NG": "gas", "BIT": "coal", "SUB": "coal", "LIG": "coal", "RC": "coal",
    "DFO": "oil", "RFO": "oil", "NUC": "nuclear", "WND": "wind",
    "SUN": "solar", "WAT": "hydro",
}
VOM = {"gas": 3.0, "coal": 4.5, "oil": 6.0, "nuclear": 2.5,
       "wind": 0.0, "solar": 0.0, "hydro": 0.0}


def get_all(route, params, key):
    rows, offset = [], 0
    while True:
        p = {**params, "api_key": key, "offset": offset, "length": 5000}
        r = requests.get(f"{BASE}/{route}/data/", params=p, timeout=60)
        r.raise_for_status()
        data = r.json()["response"]["data"]
        rows += data
        if len(data) < 5000:
            return pd.DataFrame(rows)
        offset += 5000


def cached(name, fn):
    RAW.mkdir(parents=True, exist_ok=True)
    path = RAW / f"{name}.json"
    if path.exists():
        return pd.DataFrame(json.loads(path.read_text()))
    df = fn()
    path.write_text(df.to_json(orient="records"))
    return df


def api_key():
    """EIA_API_KEY from the environment, else from .env in the repo root."""
    if os.environ.get("EIA_API_KEY"):
        return os.environ["EIA_API_KEY"]
    env = ROOT / ".env"
    if env.exists():
        for line in env.read_text().splitlines():
            k, _, v = line.partition("=")
            if k.strip() == "EIA_API_KEY" and v.strip():
                return v.strip().strip("'\"")
    raise SystemExit("Set EIA_API_KEY or put EIA_API_KEY=... in .env")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--year", type=int, default=2024)
    args = ap.parse_args()
    key = api_key()

    cap = cached(f"capacity_{args.year}", lambda: get_all(
        "electricity/operating-generator-capacity",
        {"frequency": "monthly", "data[]": ["net-summer-capacity-mw"],
         "facets[stateid][]": "MI", "facets[status][]": "OP",
         "start": f"{args.year}-12", "end": f"{args.year}-12"}, key))

    fuel = cached(f"facility_fuel_{args.year}", lambda: get_all(
        "electricity/facility-fuel",
        {"frequency": "annual", "data[]": ["generation", "total-consumption-btu"],
         "facets[state][]": "MI", "start": str(args.year), "end": str(args.year)}, key))

    cap["fuel"] = cap["energy_source_code"].map(FUEL_MAP)
    cap = cap.dropna(subset=["fuel"])
    cap["capacity_mw"] = pd.to_numeric(cap["net-summer-capacity-mw"], errors="coerce")

    # Plant-level heat rate (MMBtu/MWh) from EIA-923 totals
    fuel["generation"] = pd.to_numeric(fuel["generation"], errors="coerce")
    fuel["btu"] = pd.to_numeric(fuel["total-consumption-btu"], errors="coerce")
    hr = fuel.groupby("plantCode")[["btu", "generation"]].sum()
    hr = (hr["btu"] / hr["generation"]).rename("heat_rate").reset_index()
    hr = hr.rename(columns={"plantCode": "plantid"})

    plants = (cap.groupby(["plantid", "plantName", "fuel"], as_index=False)
                 ["capacity_mw"].sum())
    plants["plantid"] = plants["plantid"].astype(str)
    hr["plantid"] = hr["plantid"].astype(str)
    plants = plants.merge(hr, on="plantid", how="left")

    zero_fuel = plants["fuel"].isin(["wind", "solar", "hydro"])
    plants.loc[zero_fuel, "heat_rate"] = 0.0
    # Fallback heat rates where 923 data is missing or nonsensical
    default_hr = {"gas": 8.0, "coal": 10.5, "oil": 12.5, "nuclear": 10.4}
    bad = plants["heat_rate"].isna() | (plants["heat_rate"] > 25) | (
        (plants["heat_rate"] < 5) & ~zero_fuel)
    plants.loc[bad, "heat_rate"] = plants.loc[bad, "fuel"].map(default_hr)
    plants["vom"] = plants["fuel"].map(VOM)

    out = plants.rename(columns={"plantName": "unit"})[
        ["unit", "fuel", "capacity_mw", "heat_rate", "vom"]]
    out = out[out["capacity_mw"] > 0]
    out.to_csv(ROOT / "data/mi_fleet.csv", index=False)
    print(f"Wrote {len(out)} plants, {out.capacity_mw.sum():,.0f} MW to data/mi_fleet.csv")
    print(out.groupby("fuel").capacity_mw.sum().sort_values(ascending=False))


if __name__ == "__main__":
    main()
