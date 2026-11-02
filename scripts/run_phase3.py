"""Phase 3: Michigan vs rest of MISO, one line, every hour of a year.

    python scripts/run_phase3.py [--year 2024] [--base-limit 4000]

Inputs (from fetch_eia.py --region MISO and fetch_miso.py)
  data/miso_fleet.csv, data/miso_load_{year}.csv, data/miso_renew_{year}.csv,
  data/miso_lmp_{year}.csv

Outputs
  results/phase3_agreement.csv    solvers vs HiGHS + KKT on a sample of real hours
  results/phase3_hourly.csv       hourly model prices/flow at the base limit, with actuals
  results/phase3_limit_sweep.csv  congestion statistics vs line limit
  results/phase3_by_wind.csv      congestion and imports by MISO wind output quintile
  results/phase3_congestion.png   model vs actual congestion
  results/phase3_limit_sweep.png  how often the line binds as the limit changes
"""
import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from meritorder.fleet import load_fleet  # noqa: E402
from meritorder.kkt import two_region_kkt_report, kkt_ok  # noqa: E402
from meritorder.network import merit_two_region, flow_bisection, highs_two_region  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results"

ap = argparse.ArgumentParser()
ap.add_argument("--year", type=int, default=2024)
ap.add_argument("--base-limit", type=float, default=4000.0,
                help="Michigan import/export limit in MW for the hourly run (assumption, see docs)")
args = ap.parse_args()
Y = args.year

fleet = load_fleet(ROOT / "data/miso_fleet.csv")
mi = (fleet["state"] == "MI").values
c, cap0 = fleet["marginal_cost"].values, fleet["capacity_mw"].values
wind, solar = (fleet["fuel"] == "wind").values, (fleet["fuel"] == "solar").values

load = pd.read_csv(ROOT / f"data/miso_load_{Y}.csv")
ren = pd.read_csv(ROOT / f"data/miso_renew_{Y}.csv")
lmp = pd.read_csv(ROOT / f"data/miso_lmp_{Y}.csv")
hours = load.merge(ren, on=["date", "he"])
mich = lmp[lmp.hub == "MICHIGAN.HUB"].set_index(["date", "he"])
other_mcc = lmp[lmp.hub != "MICHIGAN.HUB"].groupby(["date", "he"]).mcc.mean()
hours = hours.join(mich[["lmp", "mec", "mcc", "mlc"]], on=["date", "he"])
hours = hours.join(other_mcc.rename("mcc_other"), on=["date", "he"])
print(f"Fleet: Michigan {cap0[mi].sum():,.0f} MW ({mi.sum()} units), "
      f"rest of MISO {cap0[~mi].sum():,.0f} MW ({(~mi).sum()} units). Hours: {len(hours):,}")


def capacity(row):
    """Wind and solar available = actual MISO output, shared by installed capacity."""
    cap = cap0.copy()
    cap[wind] *= row.wind_mw / cap0[wind].sum()
    cap[solar] *= row.solar_mw / cap0[solar].sum()
    return cap


def split(cap, row):
    return c[mi], cap[mi], row.MI_est, c[~mi], cap[~mi], row.MISO - row.MI_est


# 1. Solver agreement and KKT on a sample of real hours
rows = []
for row in hours.iloc[::97].itertuples():
    cap = capacity(row)
    inst = split(cap, row)
    ref = highs_two_region(*inst, args.base_limit)
    for name, fn in [("merit_two_region", merit_two_region), ("flow_bisection", flow_bisection),
                     ("highs", highs_two_region)]:
        gM, gR, f, cost, lamM, lamR = fn(*inst, args.base_limit)
        rep = two_region_kkt_report(*inst, args.base_limit, gM, gR, f, lamM, lamR)
        rows.append({"date": row.date, "he": row.he, "solver": name, "cost_err": abs(cost - ref[3]),
                     "lamM_err": abs(lamM - ref[4]), "lamR_err": abs(lamR - ref[5]),
                     "flow_err": abs(f - ref[2]), "kkt_ok": kkt_ok(rep),
                     "gap": rep["duality_gap"]})
agree = pd.DataFrame(rows)
agree.to_csv(OUT / "phase3_agreement.csv", index=False)
print(f"\nAgreement vs HiGHS on {agree.date.nunique()} sampled hours, limit {args.base_limit:,.0f} MW:")
print(agree.groupby("solver").agg(max_cost_err=("cost_err", "max"), max_lamM_err=("lamM_err", "max"),
                                  max_lamR_err=("lamR_err", "max"), all_kkt_ok=("kkt_ok", "all"),
                                  max_abs_gap=("gap", lambda s: s.abs().max())).to_string())

# 2. Every hour, for a sweep of line limits
limits = [1000, 2000, 3000, 4000, 6000, 8000, np.inf]
if args.base_limit not in limits:
    limits = sorted(limits + [args.base_limit])
res = {F: np.empty((len(hours), 3)) for F in limits}
for k, row in enumerate(hours.itertuples()):
    cap = capacity(row)
    inst = split(cap, row)
    for F in limits:
        _, _, f, _, lamM, lamR = merit_two_region(*inst, F)
        res[F][k] = (f, lamM, lamR)

act = hours["mcc"] - hours["mcc_other"]      # Michigan congestion relative to the other hubs
sweep = []
for F in limits:
    f, lamM, lamR = res[F].T
    cong = lamM - lamR
    sweep.append({"limit_mw": F, "share_import_binding": np.mean(f >= F - 1e-6),
                  "share_export_binding": np.mean(f <= -F + 1e-6),
                  "mean_congestion": cong.mean(), "mean_abs_congestion": np.abs(cong).mean(),
                  "share_abs_cong_gt_1": np.mean(np.abs(cong) > 1),
                  "mean_cong_when_import_binds": cong[f >= F - 1e-6].mean() if np.any(f >= F - 1e-6) else np.nan,
                  "corr_with_actual": np.corrcoef(cong, act)[0, 1] if cong.std() > 0 else np.nan,
                  "mean_import_mw": f.mean()})
sweep = pd.DataFrame(sweep)
# Two ways to read MISO's congestion: relative to the other hubs (closest to
# the model's lam_M - lam_R), and raw MCC (relative to MISO's reference bus)
for label, a in [("actual: MCC_MI - mean MCC other hubs", act),
                 ("actual: MCC_MI (vs MISO reference bus)", hours["mcc"])]:
    sweep.loc[len(sweep)] = {"limit_mw": label, "mean_congestion": a.mean(),
                             "mean_abs_congestion": a.abs().mean(),
                             "share_abs_cong_gt_1": (a.abs() > 1).mean(),
                             "corr_with_actual": np.corrcoef(a, hours["MI_est"])[0, 1]}
sweep.to_csv(OUT / "phase3_limit_sweep.csv", index=False)
print("\nCongestion vs line limit ($/MWh, Michigan price minus rest of MISO).")
print("For the actual rows, corr_with_actual holds corr(actual congestion, Michigan load):")
print(sweep.round(3).to_string(index=False))

F0 = args.base_limit
f, lamM, lamR = res[F0].T
hourly = hours[["date", "he", "MI_est", "MISO", "wind_mw", "solar_mw", "lmp", "mec", "mcc", "mlc",
                "mcc_other"]].copy()
hourly["flow_mw"], hourly["lam_mi"], hourly["lam_rest"] = f, lamM, lamR
hourly["cong_model"] = lamM - lamR
hourly["cong_actual"] = act
hourly.to_csv(OUT / "phase3_hourly.csv", index=False)
print(f"\nAt {F0:,.0f} MW: model Michigan price mean ${lamM.mean():.2f} vs actual hub LMP "
      f"${hours.lmp.mean():.2f}; rest-of-MISO price mean ${lamR.mean():.2f} vs actual MEC "
      f"${hours.mec.mean():.2f}")
print(f"corr(model Michigan price, actual LMP) = {np.corrcoef(lamM, hours.lmp)[0, 1]:.2f}")
print(f"Michigan imports: mean {f.mean():,.0f} MW, at import limit {np.mean(f >= F0 - 1e-6):.1%} of hours")

# What drives Michigan congestion: MISO-wide wind, not Michigan load
byw = pd.DataFrame({"wind_quintile": pd.qcut(hours.wind_mw, 5).astype(str),
                    "wind_mw": hours.wind_mw, "mi_load": hours.MI_est,
                    "actual_mcc_mi": hours.mcc, "actual_vs_other_hubs": act})
for F in [1000, F0]:
    byw[f"model_cong_{F:.0f}"] = res[F][:, 1] - res[F][:, 2]
    byw[f"model_import_{F:.0f}"] = res[F][:, 0]
byw = byw.groupby("wind_quintile", sort=False).mean().sort_values("wind_mw")
byw.to_csv(OUT / "phase3_by_wind.csv")
print("\nBy MISO wind output quintile ($/MWh, MW):")
print(byw.round(2).to_string())
print(f"corr(MISO wind, MCC_MI) = {hours.wind_mw.corr(hours.mcc):.2f}, "
      f"corr(Michigan load, MCC_MI) = {hours.MI_est.corr(hours.mcc):.2f}")

# Figures
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
ax = axes[0]
bins = pd.qcut(hourly.MI_est, 10)
g = hourly.groupby(bins, observed=True)
mid = g.MI_est.mean()
tight = res[1000][:, 1] - res[1000][:, 2]
ax.plot(mid, g.cong_model.mean(), "o-", color="C3", label=f"model, limit {F0:,.0f} MW")
ax.plot(mid, pd.Series(tight).groupby(bins.values, observed=True).mean().values, "o--",
        color="C1", label="model, limit 1,000 MW")
ax.plot(mid, g.cong_actual.mean(), "o-", color="0.3", label="actual: MCC_MI - other hubs")
ax.plot(mid, g.mcc.mean(), "s:", color="0.5", label="actual: MCC_MI")
ax.axhline(0, color="0.7", lw=0.8)
ax.set_xlabel("Michigan load (MW, decile mean)")
ax.set_ylabel("Michigan congestion ($/MWh)")
ax.set_title("Mean congestion by Michigan load", fontsize=10)
ax.legend()
ax = axes[1]
week = hourly.iloc[24 * 196:24 * 203]       # a July week
x = np.arange(len(week))
ax.plot(x, week.cong_actual, color="0.3", label="actual")
ax.plot(x, week.cong_model, color="C3", label="model")
ax.set_xlabel("hour (July 15 to 21)")
ax.set_ylabel("$/MWh")
ax.set_title("One summer week", fontsize=10)
ax.legend()
fig.tight_layout()
fig.savefig(OUT / "phase3_congestion.png", dpi=150)

fig, ax = plt.subplots(figsize=(8, 4.5))
s = sweep[sweep.limit_mw.apply(lambda v: isinstance(v, (int, float)) and np.isfinite(v))]
ax.plot(s.limit_mw, 100 * s.share_import_binding, "o-", color="C3", label="import limit binds")
ax.plot(s.limit_mw, 100 * s.share_export_binding, "o-", color="C0", label="export limit binds")
ax.plot(s.limit_mw, 100 * s.share_abs_cong_gt_1, "s--", color="0.4", label="model |congestion| > $1")
ax.axhline(100 * (act.abs() > 1).mean(), color="0.2", ls=":",
           label="actual |MCC_MI - other hubs| > $1")
ax.axhline(100 * (hours["mcc"].abs() > 1).mean(), color="0.5", ls="-.",
           label="actual |MCC_MI| > $1")
ax.set_xlabel("Michigan transfer limit (MW)")
ax.set_ylabel("% of hours")
ax.set_title("How often the line binds, by assumed limit", fontsize=10)
ax.legend()
fig.tight_layout()
fig.savefig(OUT / "phase3_limit_sweep.png", dpi=150)
print(f"\nSaved figures and CSVs to {OUT}")
