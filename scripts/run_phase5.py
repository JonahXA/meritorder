"""Phase 5: scenarios on the 2024 hourly model, read through duals.

    python scripts/run_phase5.py               # ~10 minutes
    python scripts/run_phase5.py --plots-only  # redraw figures from the CSV

Two baselines:
  economic    pure cost-based dispatch (Phase 4 main)
  calibrated  30% of each coal unit self-commits (offered at $0, costed at its
              true cost), which matches MISO's 2024 coal share (EIA-930)

Scenarios
  gas         gas fuel price x 0.75, 1.5, 2.0 (envelope theorem check at 1.0)
  coal        retire all Michigan coal (5.8 GW) at a 4 GW and an unlimited line;
              retire the least efficient half of MISO coal
  carbon      carbon price 0 to 100 $/tCO2
  cap         annual CO2 cap 10% and 25% below baseline; its shadow price is
              found by dual decomposition across hours, and checked against a
              full one-week LP in HiGHS

Outputs
  results/phase5_scenarios.csv, results/phase5_cap_check.csv,
  results/phase5_carbon.png, results/phase5_gas.png
"""
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from meritorder.emissions import cap_by_bisection, cap_highs, dispatch_with_carbon  # noqa: E402
from meritorder.hourly import Year  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results"
MUST_RUN = 0.3

rows = []


def record(base, scenario, setting, y, r, **extra):
    rows.append({"baseline": base, "scenario": scenario, "setting": setting,
                 **r.summary(y.fleet), **extra})
    s = rows[-1]
    print(f"  {base:10s} {scenario:8s} {str(setting):22s} MI ${s['mi_price_mean']:6.2f}  "
          f"CO2 {s['co2_mt']:6.1f} Mt  coal {s['share_coal']:.3f}  gas {s['share_gas']:.3f}")


t0 = time.time()
if "--plots-only" in sys.argv:
    res = pd.read_csv(OUT / "phase5_scenarios.csv")
else:
    t0 = time.time()
    years = {"economic": Year(ROOT), "calibrated": Year(ROOT, coal_must_run=MUST_RUN)}
    for base, y in years.items():
        print(f"\n{base} baseline")
        ref = y.run()
        record(base, "baseline", "", y, ref)

        # Gas price. Envelope theorem: d(total cost)/d(gas multiplier) at 1.0
        # equals the baseline's spending on gas fuel. The dispatch shifts, but at
        # an optimum that shift contributes nothing to first order.
        up, dn = y.run(gas_mult=1.01), y.run(gas_mult=0.99)
        record(base, "gas", "x1.00 envelope", y, ref,
               d_cost_d_gasmult_fd=(up.cost - dn.cost) / 0.02, gas_fuel_spend=ref.gas_fuel_spend)
        for mult in [0.75, 1.5, 2.0]:
            record(base, "gas", f"x{mult:.2f}", y, y.run(gas_mult=mult))

        # Coal retirement
        mi_coal = y.mi & y.coal
        for F in [4000.0, np.inf]:
            r = y.run(F=F, retire=mi_coal)
            line_value = np.abs(r.hourly.lam_mi - r.hourly.lam_rest).sum()
            record(base, "coal", f"retire MI, line {F:.0f}", y, r,
                   line_value_usd_per_mw_yr=line_value)
        hr = y.fleet["heat_rate"].where(y.coal, -np.inf).values
        coal_cap = y.cap0 * y.coal
        order = np.argsort(-hr)
        worst = np.zeros(len(hr), bool)
        worst[order[np.cumsum(coal_cap[order]) <= 0.5 * coal_cap.sum()]] = True
        record(base, "coal", "retire worst half MISO", y, y.run(retire=worst))

        # Carbon price
        for tau in [15, 30, 50, 75, 100]:
            record(base, "carbon", f"${tau}/t", y, y.run(tau=tau))

        # Annual cap: shadow price by dual decomposition. With no new clean
        # capacity the only lever is fuel switching, so deep caps can be
        # unreachable at any price; record the floor instead.
        floor = y.run(tau=500.0)
        record(base, "carbon", "$500/t (floor)", y, floor)
        for cut in [0.05, 0.10, 0.20]:
            try:
                tau, r = y.cap((1 - cut) * ref.co2)
            except ValueError:
                print(f"  {base:10s} cap      -{cut:.0%}: unreachable, floor {floor.co2 / 1e6:.1f} Mt "
                      f"= -{1 - floor.co2 / ref.co2:.1%}")
                rows.append({"baseline": base, "scenario": "cap", "setting": f"-{cut:.0%}",
                             "cap_shadow_price": np.nan, "unreachable": True})
                continue
            record(base, "cap", f"-{cut:.0%}", y, r, cap_shadow_price=tau)
        print(f"  ({time.time() - t0:.0f} s)")

    res = pd.DataFrame(rows)
    res.to_csv(OUT / "phase5_scenarios.csv", index=False)

    # Check the cap dual on one real week against a single LP over all 168 hours
    # (pooled MISO, no line, economic baseline, so it fits cap_highs)
    y = years["economic"]
    wk = y.h.iloc[24 * 196:24 * 203]
    c = y.mc[wk.month.iloc[0]]
    caps = np.tile(y.cap0, (len(wk), 1))
    caps[:, y.wind] *= (wk.wind_mw.values / y.cap0[y.wind].sum())[:, None]
    caps[:, y.solar] *= (wk.solar_mw.values / y.cap0[y.solar].sum())[:, None]
    d = wk.MISO.values
    E0 = dispatch_with_carbon(c, caps, d, y.e, 0.0)[2]
    E_floor = dispatch_with_carbon(c, caps, d, y.e, 1000.0)[2]
    checks = []
    for frac in [0.3, 0.6]:       # caps 30% and 60% of the way to the week's floor
        E = E0 - frac * (E0 - E_floor)
        cut = 1 - E / E0
        t1 = time.time()
        _, cost_b, em_b, tau_b, lam_b, lower = cap_by_bisection(c, caps, d, y.e, E)
        t_b = time.time() - t1
        t1 = time.time()
        _, cost_h, tau_h, lam_h = cap_highs(c, caps, d, y.e, E)
        t_h = time.time() - t1
        checks.append({"cut": cut, "tau_bisection": tau_b, "tau_highs": tau_h,
                       "cost_bisection": cost_b, "cost_highs": cost_h, "dual_bound": lower,
                       "max_price_diff": np.abs(lam_b - lam_h).max(),
                       "seconds_bisection": t_b, "seconds_highs": t_h})
    chk = pd.DataFrame(checks)
    chk.to_csv(OUT / "phase5_cap_check.csv", index=False)
    print(f"\nCap dual check, one July week, {caps.shape[1]} units x {len(wk)} hours:")
    print(chk.to_string(index=False))


# Figures
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
for base, color in [("economic", "C0"), ("calibrated", "C3")]:
    s = res[(res.baseline == base) & res.scenario.isin(["baseline", "carbon"])].copy()
    s["setting"] = s.setting.fillna("")
    s = s[~s.setting.str.contains("floor")]
    s["tau"] = [0.0 if v == "" else float(v.strip("$/t")) for v in s.setting]
    s = s.sort_values("tau")
    axes[0].plot(s.tau, s.co2_mt, "o-", color=color, label=base)
    axes[1].plot(s.tau, s.mi_price_mean, "o-", color=color, label=base)
    caps_ = res[(res.baseline == base) & (res.scenario == "cap")].dropna(subset=["co2_mt"])
    axes[0].scatter(caps_.cap_shadow_price, caps_.co2_mt, marker="*", s=150, color=color,
                    zorder=3, label=f"{base}: annual caps at their shadow prices")
axes[0].set_xlabel("carbon price ($/tCO2)")
axes[0].set_ylabel("MISO CO2 (Mt/yr)")
axes[0].set_title("Emissions vs carbon price", fontsize=10)
axes[0].legend(fontsize=8)
axes[1].set_xlabel("carbon price ($/tCO2)")
axes[1].set_ylabel("mean Michigan price ($/MWh)")
axes[1].set_title("Michigan price vs carbon price", fontsize=10)
axes[1].legend(fontsize=8)
fig.tight_layout()
fig.savefig(OUT / "phase5_carbon.png", dpi=150)

fig, ax = plt.subplots(figsize=(7, 4.5))
for base, color in [("economic", "C0"), ("calibrated", "C3")]:
    s = res[(res.baseline == base) & (res.scenario == "gas")].copy()
    s["mult"] = [float(v.split()[0][1:]) for v in s.setting]
    s = s.sort_values("mult")
    ax.plot(s.mult, s.mi_price_mean, "o-", color=color, label=f"{base}: Michigan price")
ax.set_xlabel("gas price multiplier (1.0 = 2024 actual)")
ax.set_ylabel("mean Michigan price ($/MWh)")
ax.set_title("Gas price pass-through", fontsize=10)
ax.legend()
fig.tight_layout()
fig.savefig(OUT / "phase5_gas.png", dpi=150)
print(f"\nSaved to {OUT} ({time.time() - t0:.0f} s)")
