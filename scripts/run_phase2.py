"""Phase 2: quadratic costs, first-order and dual methods.

Outputs
  results/phase2_agreement.csv     solver agreement, KKT, iterations, timing per demand level
  results/phase2_price_curve.png   LP step price vs QP price across demand
  results/phase2_convergence.png   projected gradient and dual ascent error vs iteration, by kappa
  results/phase2_newton.png        dual Newton vs bisection, |lambda_k - lambda*|
  results/phase2_convergence.csv   observed vs theoretical linear rates
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from meritorder import dispatch, qp  # noqa: E402
from meritorder.fleet import load_fleet  # noqa: E402
from meritorder.kkt import qp_kkt_report, kkt_ok  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results"
OUT.mkdir(exist_ok=True)
SPREAD, FLOOR = 0.10, 1.0

fleet_path = sys.argv[1] if len(sys.argv) > 1 else ROOT / "data/sample_fleet.csv"
df = load_fleet(fleet_path)
c, cap = df["marginal_cost"].values, df["capacity_mw"].values
a, b = qp.quadratic_coeffs(c, cap, SPREAD, FLOOR)
total = cap.sum()
print(f"Fleet: {len(df)} units, {total:,.0f} MW. "
      f"Marginal cost rises max({SPREAD:.0%} of c, ${FLOOR:.0f}) from zero to full load. "
      f"kappa = max b / min b = {b.max() / b.min():,.0f}")

# 1. Agreement and KKT across demand
rows = []
for d in np.linspace(0.02, 0.98, 49) * total:
    out = qp.solve_all(a, b, cap, d)
    ref = out["scipy"]
    free = ((ref["g"] > 0) & (ref["g"] < cap)).sum()
    for name, r in out.items():
        rep = qp_kkt_report(a, b, cap, d, r["g"], r["lam"])
        rows.append({
            "demand_mw": d, "solver": name, "price": r["lam"], "cost": r["cost"],
            "cost_err_vs_scipy": abs(r["cost"] - ref["cost"]),
            "price_err_vs_scipy": abs(r["lam"] - ref["lam"]),
            "max_g_err_mw": float(np.max(np.abs(r["g"] - ref["g"]))),
            "kkt_ok": kkt_ok(rep, tol=1e-4), "duality_gap": rep["duality_gap"],
            "iters": r["iters"], "ms": 1000 * r["seconds"],
            # no unit strictly between its bounds: price is an interval again
            "no_free_unit": free == 0,
        })
res = pd.DataFrame(rows)
res.to_csv(OUT / "phase2_agreement.csv", index=False)
ok = res[~res.no_free_unit]
print(ok.groupby("solver").agg(
    max_cost_err=("cost_err_vs_scipy", "max"),
    max_price_err=("price_err_vs_scipy", "max"),
    max_g_err_mw=("max_g_err_mw", "max"),
    all_kkt_ok=("kkt_ok", "all"),
    max_abs_gap=("duality_gap", lambda s: s.abs().max()),
    median_iters=("iters", "median"),
    mean_ms=("ms", "mean"),
).to_string())
n_gap = res[res.solver == "scipy"].no_free_unit.sum()
print(f"\nDemand levels with no free unit (price not unique): {n_gap} of 49")

# 2. Price curve: LP step vs QP
ds = np.linspace(0.005, 0.995, 600) * total
lp_price = [dispatch.merit_order(c, cap, d)[2] for d in ds]
qp_price = [qp.dual_newton(a, b, cap, d)[2] for d in ds]
fig, ax = plt.subplots(figsize=(8, 4.5))
ax.plot(ds, lp_price, color="0.5", drawstyle="steps-post", label="Phase 1 LP (constant cost)")
ax.plot(ds, qp_price, color="C0", label=f"Phase 2 QP (cost rises {SPREAD:.0%} over each unit's range)")
ax.set_xlabel("Demand (MW)")
ax.set_ylabel("Price ($/MWh)")
ax.set_title("Quadratic costs turn flat steps into ramps")
ax.legend()
fig.tight_layout()
fig.savefig(OUT / "phase2_price_curve.png", dpi=150)

# 3. Convergence vs conditioning on synthetic instances
rng = np.random.default_rng(382)
n = 30
a0 = rng.uniform(0, 80, n)
cap0 = rng.uniform(100, 800, n)
d0 = 0.55 * cap0.sum()
kappas = [1, 10, 100, 1000]
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
conv = []
for j, kappa in enumerate(kappas):
    b0 = 0.02 * np.exp(rng.uniform(0, np.log(kappa), n)) if kappa > 1 else np.full(n, 0.02)
    b0[0], b0[-1] = 0.02, 0.02 * kappa
    g_star, _, lam_star, _ = qp.dual_newton(a0, b0, cap0, d0)
    for ax, name in zip(axes, ["projected_gradient", "dual_ascent"]):
        errs = []
        qp.QP_SOLVERS[name](a0, b0, cap0, d0, tol=1e-13, max_iter=20_000,
                            callback=lambda g, lam: errs.append(np.linalg.norm(g - g_star)))
        # stop at the projection's accuracy floor (~1e-10 MW)
        errs = np.array(errs)
        below = np.flatnonzero(errs < 1e-10)
        errs = np.maximum(errs[:below[0] + 1] if len(below) else errs, 1e-10)
        k = np.arange(1, len(errs) + 1)
        ax.semilogy(k, errs, color=f"C{j}", label=f"kappa = {kappa}")
        if name == "projected_gradient":
            theory = 1 - b0.min() / b0.max()
            ax.semilogy(k, errs[0] * theory ** (k - 1), color=f"C{j}", ls="--", lw=0.8)
        else:
            free = (g_star > 0) & (g_star < cap0)
            theory = 1 - np.sum(1 / b0[free]) / np.sum(1 / b0)
        # observed rate: geometric mean of error ratios over the clean middle stretch
        mask = (errs > 1e-9) & (errs < errs[0] * 1e-1)
        obs = np.exp(np.mean(np.diff(np.log(errs[mask])))) if mask.sum() > 2 else np.nan
        conv.append({"method": name, "kappa": kappa, "iters": len(errs),
                     "observed_rate": obs, "theory_rate": theory})
for ax, title in zip(axes, ["Projected gradient, step 1/L (dashed: (1 - 1/kappa)^k bound)",
                            "Dual ascent, step 1 / sum(1/b_i)"]):
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("iteration")
    ax.set_xscale("log")
    ax.set_ylim(1e-10, None)
    ax.legend()
axes[0].set_ylabel("||g_k - g*|| (MW)")
fig.tight_layout()
fig.savefig(OUT / "phase2_convergence.png", dpi=150)
conv = pd.DataFrame(conv)
conv.to_csv(OUT / "phase2_convergence.csv", index=False)
print("\nLinear rates (dual ascent theory = 1 - sum_free(1/b) / sum_all(1/b), local):")
print(conv.to_string(index=False))

# 4. Newton vs bisection on the dual
fig, ax = plt.subplots(figsize=(7, 4.5))
_, _, lam_star, _ = qp.dual_newton(a, b, cap, 0.6 * total, tol=1e-15)
iters = {}
for name, color in [("dual_bisection", "C1"), ("dual_newton", "C2")]:
    lams = []
    qp.QP_SOLVERS[name](a, b, cap, 0.6 * total,
                        callback=lambda g, lam: lams.append(abs(lam - lam_star)))
    lams = np.maximum(lams, 1e-16)
    iters[name] = len(lams)
    ax.semilogy(range(1, len(lams) + 1), lams, "o-", ms=3, color=color, label=name)
ax.set_xlabel("iteration")
ax.set_ylabel("|lambda_k - lambda*| ($/MWh)")
ax.set_title("Dual is piecewise quadratic: Newton finishes in a few steps")
ax.legend()
fig.tight_layout()
fig.savefig(OUT / "phase2_newton.png", dpi=150)
newton = res[res.solver == "dual_newton"].iters
bis = res[res.solver == "dual_bisection"].iters
print(f"\nIterations across 49 demand levels: Newton median {newton.median():.0f} "
      f"(max {newton.max()}), bisection median {bis.median():.0f}")
print(f"Saved figures and CSVs to {OUT}")
