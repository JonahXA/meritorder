"""Pull MISO hub prices and zone loads for a year.

    python scripts/fetch_miso.py --year 2024

Writes
  data/miso_lmp_{year}.csv    hourly day-ahead ex-post LMP, MCC, MLC for the 8 MISO hubs
  data/miso_load_{year}.csv   hourly actual load by local resource zone, plus a
                              Michigan (LRZ7) estimate
  data/miso_renew_{year}.csv  hourly MISO wind and solar output (EIA-930), so
                              renewables aren't modeled at nameplate every hour

Sources are MISO's public market reports (docs.misoenergy.org/marketreports),
no key needed:
  {date}_da_expost_lmp.csv   one file per day, every node, LMP/MCC/MLC by hour ending (EST)
  {date}_df_al.xls           one file per day, forecast and actual load for the
                             week, by zone

MISO reports Michigan (LRZ7) and Wisconsin/Upper Peninsula (LRZ2) only as one
combined zone, LRZ2_7, and so does EIA-930. The Michigan share is estimated
from EIA annual retail sales, MI / (MI + WI). See docs/phase3.md.
"""
import argparse
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import date, timedelta
from io import StringIO
from pathlib import Path

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fetch_eia import api_key, get_all, cached  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data/raw/miso"
URL = "https://docs.misoenergy.org/marketreports/{}"


def download(name):
    """Fetch a market report into data/raw/miso, once."""
    path = RAW / name
    if not path.exists():
        r = requests.get(URL.format(name), timeout=60)
        if r.status_code == 404:
            return None
        r.raise_for_status()
        path.write_bytes(r.content)
    return path


def hub_lmps(day):
    """Hub rows from one day's DA ex-post LMP file, long format.

    The full file is ~1.2 MB (every node), so only the hub extract is cached.
    """
    out = RAW / "lmp" / f"{day:%Y%m%d}_hubs.csv"
    if out.exists():
        return pd.read_csv(out)
    r = requests.get(URL.format(f"{day:%Y%m%d}_da_expost_lmp.csv"), timeout=60)
    r.raise_for_status()
    lines = r.text.splitlines()
    head = next(i for i, l in enumerate(lines) if l.startswith("Node,"))
    df = pd.read_csv(StringIO("\n".join(lines[head:])))
    df = df[df["Type"] == "Hub"]
    df = df[df["Node"].str.endswith(".HUB")]
    df = df.melt(id_vars=["Node", "Value"], value_vars=[c for c in df if c.startswith("HE")],
                 var_name="he", value_name="v")
    df["he"] = df["he"].str[3:].astype(int)
    df = df.pivot_table(index=["Node", "he"], columns="Value", values="v").reset_index()
    df.columns.name = None
    df = df.rename(columns={"Node": "hub", "LMP": "lmp", "MCC": "mcc", "MLC": "mlc"})
    df.insert(0, "date", day.isoformat())
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    return df


def zone_loads(day):
    """Actual load rows from the forecast/actual file published on `day`."""
    path = download(f"{day:%Y%m%d}_df_al.xls")
    if path is None:
        return None
    x = pd.read_excel(path, header=None)
    head = x.index[x[0].astype(str).str.strip() == "Market Day"][0]
    cols = x.iloc[head].astype(str).str.strip()
    df = x.iloc[head + 1:].copy()
    df.columns = cols
    df = df[pd.to_numeric(df["HourEnding"], errors="coerce").notna()]
    df["date"] = pd.to_datetime(df["Market Day"]).dt.date.astype(str)
    df["he"] = df["HourEnding"].astype(int)
    act = [c for c in cols if c.endswith("ActualLoad (MWh)")]
    df = df[["date", "he"] + act].rename(columns={c: c.split()[0] for c in act})
    df["published"] = day.isoformat()
    return df


def michigan_share(year):
    """MI / (MI + WI) annual retail electricity sales, from EIA."""
    key = api_key()
    sales = cached(f"retail_sales_mi_wi_{year}", lambda: get_all(
        "electricity/retail-sales",
        {"frequency": "annual", "data[]": ["sales"], "facets[stateid][]": ["MI", "WI"],
         "facets[sectorid][]": "ALL", "start": str(year), "end": str(year)}, key))
    s = sales.assign(sales=pd.to_numeric(sales["sales"])).set_index("stateid")["sales"]
    return float(s["MI"] / (s["MI"] + s["WI"])), s


def renewables(year):
    """Hourly MISO wind and solar output from EIA-930, keyed to MISO's EST hour ending.

    EIA-930 periods are UTC hour ending. MISO reports in EST year round.
    """
    key = api_key()
    df = cached(f"eia930_miso_wnd_sun_{year}", lambda: get_all(
        "electricity/rto/fuel-type-data",
        {"frequency": "hourly", "data[]": ["value"], "facets[respondent][]": "MISO",
         "facets[fueltype][]": ["WND", "SUN"],
         "start": f"{year - 1}-12-31T00", "end": f"{year + 1}-01-01T06"}, key))
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    end_est = pd.to_datetime(df["period"]) - pd.Timedelta(hours=5)
    start = end_est - pd.Timedelta(hours=1)
    df["date"] = start.dt.date.astype(str)
    df["he"] = start.dt.hour + 1
    out = df.pivot_table(index=["date", "he"], columns="fueltype", values="value").reset_index()
    out.columns.name = None
    out = out.rename(columns={"WND": "wind_mw", "SUN": "solar_mw"})
    # EIA-930 has gaps (all of 2024-07-01 is null for MISO wind and solar).
    # Fill each missing hour with the mean of the same hour the day before and after.
    grid = pd.DataFrame([(d.date().isoformat(), h) for d in
                         pd.date_range(f"{year - 1}-12-31", f"{year + 1}-01-01") for h in range(1, 25)],
                        columns=["date", "he"])
    out = grid.merge(out, on=["date", "he"], how="left")
    out["filled"] = out[["wind_mw", "solar_mw"]].isna().any(axis=1)
    for c in ["wind_mw", "solar_mw"]:
        nb = (out[c].shift(24) + out[c].shift(-24)) / 2
        out[c] = out[c].fillna(nb)
    return out[out["date"].str.startswith(str(year))].reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--year", type=int, default=2024)
    args = ap.parse_args()
    RAW.mkdir(parents=True, exist_ok=True)
    days = [date(args.year, 1, 1) + timedelta(d) for d in range(366)]
    days = [d for d in days if d.year == args.year]

    with ThreadPoolExecutor(8) as ex:
        lmp = pd.concat(ex.map(hub_lmps, days), ignore_index=True)
    lmp["mec"] = lmp["lmp"] - lmp["mcc"] - lmp["mlc"]
    lmp.to_csv(ROOT / f"data/miso_lmp_{args.year}.csv", index=False)
    print(f"LMPs: {len(lmp):,} rows, {lmp.hub.nunique()} hubs, {lmp.date.nunique()} days")

    # Each file holds a week of forecasts with actuals filled in through the day
    # before publication. Read every file into the next year and keep the latest
    # published actual for each hour.
    pub = days + [days[-1] + timedelta(d) for d in range(1, 9)]
    with ThreadPoolExecutor(8) as ex:
        load = pd.concat([d for d in ex.map(zone_loads, pub) if d is not None], ignore_index=True)
    zones = [c for c in load if c.startswith(("LRZ", "MISO"))]
    for c in zones:
        load[c] = pd.to_numeric(load[c], errors="coerce")
    load = (load.dropna(subset=["MISO"]).sort_values("published")
                .groupby(["date", "he"], as_index=False).last())
    load = load[load["date"].str.startswith(str(args.year))].drop(columns="published")

    share, sales = michigan_share(args.year)
    load["MI_est"] = share * load["LRZ2_7"]
    load.to_csv(ROOT / f"data/miso_load_{args.year}.csv", index=False)
    print(f"Load: {len(load):,} hours, zones {zones}")
    print(f"Michigan share of LRZ2_7 from {args.year} retail sales "
          f"(MI {sales['MI']:,.0f}, WI {sales['WI']:,.0f} GWh): {share:.3f}")
    print(load[["LRZ2_7", "MI_est", "MISO"]].describe().round(0).to_string())

    ren = renewables(args.year)
    ren.to_csv(ROOT / f"data/miso_renew_{args.year}.csv", index=False)
    print(f"Renewables: {len(ren):,} hours, {ren.filled.sum()} filled from neighbor days, "
          f"still missing {ren[['wind_mw', 'solar_mw']].isna().any(axis=1).sum()}")
    print(ren[["wind_mw", "solar_mw"]].describe().round(0).to_string())


if __name__ == "__main__":
    main()
