"""Phase 4: hourly validation against MISO Michigan Hub prices.

    python scripts/run_phase4.py [--year 2024] [--limit 4000]

Main model: Phase 3 two-region LP (Michigan vs rest of MISO, line limit
--limit), hourly load and wind/solar, monthly delivered fuel prices by state.
Ablations change one thing at a time:
  flat_fuel        Phase 1 default fuel prices (gas $3.00, coal $2.10)
  pooled           no transmission limit
  nameplate_renew  wind and solar at full capacity every hour

Outputs
  results/phase4_hourly.csv       hourly actuals and model prices for every variant
  results/phase4_metrics.csv      corr, MAE, RMSE, bias per variant
  results/phase4_breakdown.csv    main-model error by hour, month, load decile, price level
  results/phase4_duration.png     price duration curves
  results/phase4_profiles.png     mean price by hour of day and by month
  results/phase4_errors.png       error vs Michigan load and actual price
  results/phase4_availability.csv availability sweep: fit on Jan-Jun, test on Jul-Dec
  results/phase4_availability.png
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
from meritorder.fleet import load_fleet, monthly_marginal_costs  # noqa: E402
from meritorder.network import merit_two_region  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results"

ap = argparse.ArgumentParser()
ap.add_argument("--year", type=int, default=2024)
ap.add_argument("--limit", type=float, default=4000.0)
args = ap.parse_args()
Y, F0 = args.year, args.limit

fleet = load_fleet(ROOT / "data/miso_fleet.csv")
mi = (fleet["state"] == "MI").values
cap0 = fleet["capacity_mw"].values
wind, solar = (fleet["fuel"] == "wind").values, (fleet["fuel"] == "solar").values
mc = monthly_marginal_costs(fleet, pd.read_csv(ROOT / f"data/fuel_prices_{Y}.csv"))
c_flat = fleet["marginal_cost"].values

load = pd.read_csv(ROOT / f"data/miso_load_{Y}.csv")
ren = pd.read_csv(ROOT / f"data/miso_renew_{Y}.csv")
lmp = pd.read_csv(ROOT / f"data/miso_lmp_{Y}.csv")
mich = lmp[lmp.hub == "MICHIGAN.HUB"][["date", "he", "lmp", "mec", "mcc", "mlc"]]
h = load[["date", "he", "MI_est", "MISO"]].merge(ren, on=["date", "he"]).merge(mich, on=["date", "he"])
h["month"] = h["date"].str[:7]

VARIANTS = {               # name: (fuel prices, line limit, hourly renewables)
    "main": ("monthly", F0, True),
    "flat_fuel": ("flat", F0, True),
    "pooled": ("monthly", np.inf, True),
    "nameplate_renew": ("monthly", F0, False),
}
dispatchable = ~(wind | solar)
SHORT_PRICE, SHORT_MW = 1000.0, 1e6


def run(fuel, F, hourly_ren, avail=1.0):
    """Hourly model. avail scales every non-wind/solar unit's capacity (outages)."""
    lamM, lamR, flow = np.empty(len(h)), np.empty(len(h)), np.empty(len(h))
    for k, r in enumerate(h.itertuples()):
        c = mc[r.month] if fuel == "monthly" else c_flat
        cap = cap0.copy()
        cap[dispatchable] *= avail
        if hourly_ren:
            cap[wind] *= r.wind_mw / cap0[wind].sum()
            cap[solar] *= r.solar_mw / cap0[solar].sum()
        # Shortage unit in each region at MISO's $1,000/MWh energy offer cap,
        # so a derate that leaves an hour short sets a scarcity price
        # instead of making the LP infeasible
        cM_, capM_ = np.r_[c[mi], SHORT_PRICE], np.r_[cap[mi], SHORT_MW]
        cR_, capR_ = np.r_[c[~mi], SHORT_PRICE], np.r_[cap[~mi], SHORT_MW]
        _, _, flow[k], _, lamM[k], lamR[k] = merit_two_region(
            cM_, capM_, r.MI_est, cR_, capR_, r.MISO - r.MI_est, F)
    return lamM, lamR, flow


for name, spec in VARIANTS.items():
    h[f"{name}_mi"], h[f"{name}_rest"], h[f"{name}_flow"] = run(*spec)
    print(f"solved {name}")
h.to_csv(OUT / "phase4_hourly.csv", index=False)


def metrics(pred, act, month=None):
    """Level and shape. within_month_corr removes fuel-price level shifts;
    std_ratio < 1 means the model's swings are smaller than the market's."""
    pred, act = pd.Series(np.asarray(pred)), pd.Series(np.asarray(act))
    month = h["month"].values if month is None else month
    e = pred - act
    wm = pd.DataFrame({"p": pred, "a": act, "m": month}).groupby("m").apply(
        lambda d: pd.Series({"c": d.p.corr(d.a), "r": d.p.std() / d.a.std()}), include_groups=False)
    return {"corr": pred.corr(act), "within_month_corr": wm.c.mean(),
            "std_ratio": wm.r.mean(), "mae": e.abs().mean(),
            "rmse": np.sqrt((e ** 2).mean()), "bias": e.mean(),
            "model_mean": pred.mean(), "actual_mean": act.mean()}


rows = []
for name in VARIANTS:
    rows.append({"variant": name, "target": "Michigan Hub LMP", **metrics(h[f"{name}_mi"], h.lmp)})
    rows.append({"variant": name, "target": "MISO energy price (MEC)",
                 **metrics(h[f"{name}_rest"], h.mec)})
m = pd.DataFrame(rows)
m.to_csv(OUT / "phase4_metrics.csv", index=False)
print("\nAccuracy over", len(h), "hours ($/MWh):")
print(m.round(3).to_string(index=False))

# Availability sweep: how much of the peak gap is the model assuming every
# plant is always available? Pick alpha on Jan-Jun by MAE, score on Jul-Dec.
first = (h["month"] <= f"{Y}-06").values
avail_rows = []
avail_pred = {}
for a in [1.0, 0.9, 0.85, 0.8, 0.75, 0.7]:
    lamM = run("monthly", F0, True, a)[0]
    avail_pred[a] = lamM
    for half, mask in [("fit Jan-Jun", first), ("test Jul-Dec", ~first)]:
        avail_rows.append({"availability": a, "half": half,
                           **metrics(lamM[mask], h.lmp.values[mask], h["month"].values[mask])})
    print(f"solved availability {a}")
av = pd.DataFrame(avail_rows)
av.to_csv(OUT / "phase4_availability.csv", index=False)
best = av[av.half == "fit Jan-Jun"].sort_values("mae").availability.iloc[0]
print("\nAvailability sweep (Michigan Hub LMP):")
print(av.round(3).to_string(index=False))
print(f"Best availability on Jan-Jun by MAE: {best}. Test Jul-Dec at {best} vs 1.0:")
print(av[(av.half == "test Jul-Dec") & av.availability.isin([1.0, best])].round(3).to_string(index=False))
h["avail_best_mi"] = avail_pred[best]

# Error breakdown for the main model vs the Michigan Hub LMP
h["err"] = h.main_mi - h.lmp
h["hour"] = h.he
h["load_decile"] = pd.qcut(h.MI_est, 10, labels=False) + 1
h["price_bucket"] = pd.cut(h.lmp, [-np.inf, 0, 15, 25, 35, 50, 100, np.inf],
                           labels=["<0", "0-15", "15-25", "25-35", "35-50", "50-100", ">100"])
parts = []
for key in ["hour", "month", "load_decile", "price_bucket"]:
    g = h.groupby(key, observed=True).agg(hours=("err", "size"), actual=("lmp", "mean"),
                                          model=("main_mi", "mean"), bias=("err", "mean"),
                                          mae=("err", lambda e: e.abs().mean()),
                                          congestion_plus_loss=("mcc", "mean"))
    g["congestion_plus_loss"] += h.groupby(key, observed=True).mlc.mean()
    parts.append(g.reset_index().rename(columns={key: "level"}).assign(by=key))
br = pd.concat(parts)[["by", "level", "hours", "actual", "model", "bias", "mae",
                       "congestion_plus_loss"]]
br.to_csv(OUT / "phase4_breakdown.csv", index=False)
for key in ["month", "load_decile", "price_bucket"]:
    print(f"\nMain model by {key}:")
    print(br[br.by == key].drop(columns="by").round(2).to_string(index=False))

top = h.lmp >= h.lmp.quantile(0.95)
peak = h.MI_est >= h.MI_est.quantile(0.95)
print(f"\nTop 5% price hours: actual mean ${h.lmp[top].mean():.2f}, model ${h.main_mi[top].mean():.2f}")
print(f"Top 5% Michigan load hours: actual ${h.lmp[peak].mean():.2f}, model ${h.main_mi[peak].mean():.2f}")
print(f"Negative-price hours: actual {np.mean(h.lmp < 0):.1%}, model {np.mean(h.main_mi < 0):.1%}")
# Is the residual about net load level or about ramping? OLS of the error
# (actual - model) on MISO net load and its hour-to-hour up/down change.
h["net_gw"] = (h.MISO - h.wind_mw - h.solar_mw) / 1e3
h["up_gw"] = h.net_gw.diff().clip(lower=0)
h["down_gw"] = (-h.net_gw.diff()).clip(lower=0)
d = h.dropna(subset=["up_gw"])
y = (d.lmp - d.main_mi).values
for cols in [["net_gw"], ["net_gw", "up_gw", "down_gw"]]:
    X = np.c_[np.ones(len(d)), d[cols].values]
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    r2 = 1 - ((y - X @ beta) ** 2).sum() / ((y - y.mean()) ** 2).sum()
    print(f"residual ~ {' + '.join(cols)}: R2 = {r2:.3f}, coefs "
          + ", ".join(f"{n} {v:.2f}" for n, v in zip(["const"] + cols, beta)))
print(f"Mean congestion + loss at Michigan Hub (not in an energy-only model): "
      f"${(h.mcc + h.mlc).mean():.2f}")

# Figures
fig, ax = plt.subplots(figsize=(8, 4.5))
x = np.linspace(0, 100, len(h))
ax.plot(x, np.sort(h.lmp)[::-1], color="0.2", lw=2, label="actual Michigan Hub LMP")
for name, color in [("main", "C3"), ("flat_fuel", "C0"), ("nameplate_renew", "C2")]:
    ax.plot(x, np.sort(h[f"{name}_mi"])[::-1], color=color, label=f"model: {name}")
ax.set_ylim(-20, 150)
ax.set_xlabel("% of hours price is at least this high")
ax.set_ylabel("$/MWh")
ax.set_title(f"Price duration, {Y}", fontsize=10)
ax.legend()
fig.tight_layout()
fig.savefig(OUT / "phase4_duration.png", dpi=150)

fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
for ax, key, xl in [(axes[0], "hour", "hour ending (EST)"), (axes[1], "month", "month")]:
    b = br[br.by == key]
    ax.plot(range(len(b)), b.actual, "o-", color="0.2", label="actual LMP")
    ax.plot(range(len(b)), b.model, "o-", color="C3", label="model (main)")
    ax.set_xticks(range(len(b)))
    ax.set_xticklabels([str(v)[-2:] for v in b.level], fontsize=8)
    ax.set_xlabel(xl)
    ax.set_ylabel("mean $/MWh")
    ax.legend()
axes[0].set_title("By hour of day", fontsize=10)
axes[1].set_title("By month", fontsize=10)
fig.tight_layout()
fig.savefig(OUT / "phase4_profiles.png", dpi=150)

fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
b = br[br.by == "load_decile"]
axes[0].plot(b.level, b.actual, "o-", color="0.2", label="actual LMP")
axes[0].plot(b.level, b.model, "o-", color="C3", label="model (main)")
axes[0].set_xlabel("Michigan load decile")
axes[0].set_ylabel("mean $/MWh")
axes[0].set_title("By Michigan load", fontsize=10)
axes[0].legend()
axes[1].scatter(h.lmp, h.main_mi, s=2, alpha=0.25, color="C3")
lim = [-20, 150]
axes[1].plot(lim, lim, color="0.4", lw=0.8)
axes[1].set_xlim(lim)
axes[1].set_ylim(lim)
axes[1].set_xlabel("actual Michigan Hub LMP")
axes[1].set_ylabel("model Michigan price")
axes[1].set_title("Hourly, main model", fontsize=10)
fig.tight_layout()
fig.savefig(OUT / "phase4_errors.png", dpi=150)
fig, ax = plt.subplots(figsize=(8, 4.5))
x = np.linspace(0, 100, len(h))
ax.plot(x, np.sort(h.lmp)[::-1], color="0.2", lw=2, label="actual")
for a, color in [(1.0, "C3"), (0.85, "C1"), (0.75, "C4")]:
    ax.plot(x, np.sort(avail_pred[a])[::-1], color=color, label=f"model, availability {a}")
ax.set_ylim(-20, 150)
ax.set_xlabel("% of hours price is at least this high")
ax.set_ylabel("$/MWh")
ax.set_title("Derating dispatchable capacity: tighter supply, higher peaks", fontsize=10)
ax.legend()
fig.tight_layout()
fig.savefig(OUT / "phase4_availability.png", dpi=150)
print(f"\nSaved figures and CSVs to {OUT}")
