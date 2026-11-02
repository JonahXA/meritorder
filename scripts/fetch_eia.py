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
    "PC": "coal", "SGC": "coal",  # SGC: coal syngas (Edwardsport IGCC)
    # WH: waste heat, the steam half of a combined cycle; joins the gas CC unit
    "WH": "gas", "DFO": "oil", "RFO": "oil", "KER": "oil", "JF": "oil",
    "WO": "oil", "NUC": "nuclear", "WND": "wind", "SUN": "solar", "WAT": "hydro",
    # Waste and byproduct fuels: run whenever available, treated as must-run
    # price takers (fuel price 0 in fleet.py). Blast furnace gas is steel-mill
    # byproduct; the rest are biomass or waste.
    "BFG": "waste", "WDS": "waste", "BLQ": "waste", "LFG": "waste",
    "MSW": "waste", "MSB": "waste", "MSN": "waste", "OBG": "waste",
    "OBL": "waste", "SLW": "waste", "TDF": "waste", "OG": "waste",
    "PUR": "waste", "AB": "waste",
}
VOM = {"gas": 3.0, "coal": 4.5, "oil": 6.0, "nuclear": 2.5,
       "wind": 0.0, "solar": 0.0, "hydro": 0.0, "waste": 5.0}
# Prime movers -> technology class. Combined cycle parts (gas turbine CT,
# steam part CA, single shaft CS) are one unit: EIA-923 puts the fuel on CT
# and much of the generation on CA, so heat rate only makes sense combined.
PM_CLASS = {"CT": "CC", "CA": "CC", "CS": "CC", "GT": "GT", "IC": "IC",
            "ST": "ST", "HY": "HY", "WT": "WT", "PV": "PV"}
# Excluded: pumped storage (PS) and batteries (BA) are storage, not supply.
# They need an intertemporal model; see docs/data.md.
EXCLUDE_PM = {"PS", "BA"}


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
    ap.add_argument("--region", choices=["MI", "MISO"], default="MI",
                    help="MI: every plant in Michigan (data/mi_fleet.csv). "
                         "MISO: every plant in the MISO balancing authority (data/miso_fleet.csv).")
    args = ap.parse_args()
    key = api_key()
    y = args.year
    # Operating plus standby (available when called, e.g. Midland Cogeneration
    # ST2). Out of service (OA, OS) excluded: Palisades was OA through 2024.
    cap_params = {"frequency": "monthly", "data[]": ["net-summer-capacity-mw"],
                  "facets[status][]": ["OP", "SB"], "start": f"{y}-12", "end": f"{y}-12"}
    fuel_params = {"frequency": "annual",
                   "data[]": ["generation", "total-consumption-btu", "consumption-for-eg-btu"],
                   "start": str(y), "end": str(y)}

    if args.region == "MI":
        cap = cached(f"capacity_{y}", lambda: get_all(
            "electricity/operating-generator-capacity",
            {**cap_params, "facets[stateid][]": "MI"}, key))
        fuel = cached(f"facility_fuel_{y}", lambda: get_all(
            "electricity/facility-fuel", {**fuel_params, "facets[state][]": "MI"}, key))
    else:
        cap = cached(f"capacity_miso_{y}", lambda: get_all(
            "electricity/operating-generator-capacity",
            {**cap_params, "facets[balancing_authority_code][]": "MISO"}, key))
        states = sorted(cap["stateid"].unique())
        fuel = cached(f"facility_fuel_miso_{y}", lambda: get_all(
            "electricity/facility-fuel", {**fuel_params, "facets[state][]": states}, key))

    out = build_fleet(cap, fuel)
    name = "mi_fleet.csv" if args.region == "MI" else "miso_fleet.csv"
    out.to_csv(ROOT / "data" / name, index=False)
    print(f"Wrote {len(out)} units, {out.capacity_mw.sum():,.0f} MW to data/{name}")
    print(out.pivot_table(index="fuel", columns="ba", values="capacity_mw",
                          aggfunc="sum", fill_value=0).round(0).to_string())
    if args.region != "MI":
        return

    profile = cached(f"state_profile_{y}", lambda: get_all(
        "electricity/state-electricity-profiles/capability",
        {"frequency": "annual", "data[]": ["capability"],
         "facets[stateId][]": "MI", "facets[producertypeid][]": "TOT",
         "start": str(y), "end": str(y)}, key))
    check = cross_check(out, profile)
    check.to_csv(ROOT / "results/fleet_check.csv", index=False)
    print("\nCapacity by fuel vs EIA state profile (MW):")
    print(check.to_string(index=False))


# EIA state profile categories -> our buckets. Petroleum coke sits under
# Petroleum in the profile but we treat it as coal (burned in coal units).
PROFILE_MAP = {"Coal": "coal", "Natural Gas": "gas", "Nuclear": "nuclear",
               "Wind": "wind", "Solar": "solar", "Hydroelectric": "hydro",
               "Petroleum": "oil", "Other Gas": "waste", "Other Biomass": "waste",
               "Wood": "waste"}


def cross_check(fleet, profile):
    p = profile.copy()
    p["mw"] = pd.to_numeric(p["capability"], errors="coerce")
    p["fuel"] = p["energySourceDescription"].map(PROFILE_MAP)
    p = p.dropna(subset=["fuel"]).groupby("fuel").mw.sum()
    ours = fleet.groupby("fuel").capacity_mw.sum()
    df = pd.DataFrame({"ours_mw": ours, "eia_profile_mw": p}).fillna(0).round(0)
    df["diff_mw"] = df.ours_mw - df.eia_profile_mw
    df.loc["total"] = df.sum()
    return df.reset_index(names="fuel")


def build_fleet(cap, fuel):
    """EIA-860 generators + EIA-923 plant fuel -> one row per (plant, fuel, tech class)."""
    cap = cap.copy()
    cap["capacity_mw"] = pd.to_numeric(cap["net-summer-capacity-mw"], errors="coerce")
    unmapped = cap[~cap["energy_source_code"].isin(FUEL_MAP)
                   & ~cap["prime_mover_code"].isin(EXCLUDE_PM)]
    if len(unmapped):
        print("Dropped unmapped fuels (MW):",
              unmapped.groupby("energy_source_code").capacity_mw.sum().round(0).to_dict())
    excl = cap[cap["prime_mover_code"].isin(EXCLUDE_PM)]
    print("Excluded storage (MW):",
          excl.groupby("prime_mover_code").capacity_mw.sum().round(0).to_dict())
    cap = cap[~cap["prime_mover_code"].isin(EXCLUDE_PM)]
    cap["fuel"] = cap["energy_source_code"].map(FUEL_MAP)
    cap["tech"] = cap["prime_mover_code"].map(PM_CLASS).fillna(cap["prime_mover_code"])
    cap = cap.dropna(subset=["fuel"])

    # Heat rate (MMBtu/MWh) per plant, fuel bucket, tech class from EIA-923.
    # Drop the ALL subtotal rows, which otherwise double and triple count.
    f = fuel[(fuel["fuel2002"] != "ALL") & (fuel["primeMover"] != "ALL")].copy()
    f["fuel"] = f["fuel2002"].map(FUEL_MAP)
    f["tech"] = f["primeMover"].map(PM_CLASS).fillna(f["primeMover"])
    f["generation"] = pd.to_numeric(f["generation"], errors="coerce")
    # Fuel for electricity only. total-consumption-btu also counts steam sold
    # by cogeneration plants (Midland Cogeneration Venture), inflating heat rate.
    f["btu"] = pd.to_numeric(f["consumption-for-eg-btu"], errors="coerce")
    f["plantid"] = f["plantCode"].astype(str)
    hr = f.groupby(["plantid", "fuel", "tech"])[["btu", "generation"]].sum()
    hr = hr[hr["generation"] > 0]
    hr = (hr["btu"] / hr["generation"]).rename("heat_rate").reset_index()

    cap["plantid"] = cap["plantid"].astype(str)
    # ba matters: southwest Michigan (Cook, New Covert, Indeck Niles) is in PJM
    cap = cap.rename(columns={"stateid": "state", "balancing_authority_code": "ba"})
    plants = (cap.groupby(["plantid", "plantName", "state", "ba", "fuel", "tech"], as_index=False)
                 ["capacity_mw"].sum())
    plants = plants.merge(hr, on=["plantid", "fuel", "tech"], how="left")

    zero_fuel = plants["fuel"].isin(["wind", "solar", "hydro", "waste"])
    plants.loc[zero_fuel, "heat_rate"] = 0.0
    # Fallback where 923 has no generation for this plant/fuel/tech (new, idle,
    # or backup units) or the ratio is nonsensical. Values are typical US
    # fleet averages by technology (EIA Electric Power Annual table 8.1).
    default_hr = {("gas", "CC"): 7.6, ("gas", "GT"): 11.0, ("gas", "IC"): 9.0,
                  ("gas", "ST"): 10.5, ("coal", "ST"): 10.6, ("oil", "GT"): 13.0,
                  ("oil", "IC"): 10.5, ("oil", "ST"): 11.0, ("nuclear", "ST"): 10.4}
    typical = pd.Series([default_hr.get((fu, t), 11.0)
                         for fu, t in zip(plants["fuel"], plants["tech"])], index=plants.index)
    # Annual average heat rate of a unit that rarely runs is mostly startup and
    # hot-standby fuel (Dan E Karn steam units: 23 MMBtu/MWh). Cogen plants
    # can go the other way when fuel is split between steam and power (5.0 for
    # a combined cycle, below any real unit). Outside [0.8, 1.5] x typical for
    # the technology, treat it as unreliable.
    bad = ~zero_fuel & (plants["heat_rate"].isna()
                        | (plants["heat_rate"] < 0.8 * typical)
                        | (plants["heat_rate"] > 1.5 * typical))
    plants["hr_source"] = "eia923"
    plants.loc[zero_fuel, "hr_source"] = "zero_fuel"
    plants.loc[bad, "hr_source"] = "default"
    plants.loc[bad, "heat_rate"] = typical[bad]
    plants["vom"] = plants["fuel"].map(VOM)
    plants["unit"] = plants["plantName"] + " " + plants["tech"]
    out = plants[["unit", "fuel", "capacity_mw", "heat_rate", "vom", "tech",
                  "state", "ba", "plantid", "hr_source"]]
    return out[out["capacity_mw"] > 0].reset_index(drop=True)


if __name__ == "__main__":
    main()
