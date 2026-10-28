"""Phase 1: solve single-region dispatch across all demand levels.

Outputs
  results/phase1_supply_curve.png   price vs demand (the supply stack)
  results/phase1_dual_function.png  concave dual q(lambda) at one demand level
  results/phase1_agreement.csv      solver agreement, KKT checks, timing
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from meritorder.dispatch import solve_all, dual_function  # noqa: E402
from meritorder.fleet import load_fleet  # noqa: E402
from meritorder.kkt import kkt_report, kkt_ok  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results"
OUT.mkdir(exist_ok=True)

fleet_path = sys.argv[1] if len(sys.argv) > 1 else ROOT / "data/sample_fleet.csv"
df = load_fleet(fleet_path)
c, cap = df["marginal_cost"].values, df["capacity_mw"].values
total = cap.sum()

breakpoints = np.cumsum(cap[np.argsort(c)])

rows = []
for d in np.linspace(0.02, 0.98, 49) * total:
    out = solve_all(c, cap, d)
    ref = out["highs"]
    for name, r in out.items():
        rep = kkt_report(c, cap, d, r["g"], r["lam"])
        rows.append({
            "demand_mw": d, "solver": name, "price": r["lam"], "cost": r["cost"],
            "cost_err_vs_highs": abs(r["cost"] - ref["cost"]),
            "price_err_vs_highs": abs(r["lam"] - ref["lam"]),
            "kkt_ok": kkt_ok(rep), "duality_gap": rep["duality_gap"],
            "ms": 1000 * r["seconds"],
            # demand on a capacity breakpoint: price interval, dual not unique
            "degenerate": bool(np.isclose(d, breakpoints, atol=1e-6).any()),
        })
res = pd.DataFrame(rows)
res.to_csv(OUT / "phase1_agreement.csv", index=False)

nd = res[~res.degenerate]
summary = nd.groupby("solver").agg(
    max_cost_err=("cost_err_vs_highs", "max"),
    max_price_err=("price_err_vs_highs", "max"),
    all_kkt_ok=("kkt_ok", "all"),
    max_abs_gap=("duality_gap", lambda s: s.abs().max()),
    mean_ms=("ms", "mean"),
)
print(f"Fleet: {len(df)} units, {total:,.0f} MW")
print(summary.to_string())
deg = res[res.degenerate]
if len(deg):
    print("\nDegenerate demand levels (any price in the interval satisfies KKT):")
    print(deg.pivot_table(index="demand_mw", columns="solver", values="price").to_string())

# Supply curve
order = np.argsort(c)
x = np.concatenate([[0], np.cumsum(cap[order])])
y = np.concatenate([c[order], [c[order][-1]]])
fig, ax = plt.subplots(figsize=(8, 4.5))
ax.step(x, y, where="post", color="0.3", label="merit order (supply stack)")
hp = res[res.solver == "simplex"]
ax.scatter(hp.demand_mw, hp.price, s=14, color="C0", zorder=3,
           label="simplex dual on demand constraint")
ax.set_xlabel("Demand (MW)")
ax.set_ylabel("Price ($/MWh)")
ax.set_title("Phase 1: price equals the marginal unit's cost")
ax.legend()
fig.tight_layout()
fig.savefig(OUT / "phase1_supply_curve.png", dpi=150)

# Dual function at a mid demand level
d0 = 0.6 * total
lams = np.linspace(0, c.max() * 1.1, 600)
q = [dual_function(l, c, cap, d0) for l in lams]
star = solve_all(c, cap, d0)["merit_order"]
fig, ax = plt.subplots(figsize=(8, 4.5))
ax.plot(lams, q, color="C1")
ax.axvline(star["lam"], ls="--", color="0.4")
ax.axhline(star["cost"], ls=":", color="0.4")
ax.set_xlabel("lambda ($/MWh)")
ax.set_ylabel("q(lambda)")
ax.set_title(f"Dual function at d = {d0:,.0f} MW: max equals primal cost (strong duality)")
fig.tight_layout()
fig.savefig(OUT / "phase1_dual_function.png", dpi=150)
print(f"\nSaved figures and CSV to {OUT}")
