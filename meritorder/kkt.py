"""Verify KKT conditions and strong duality for a dispatch solution."""
import numpy as np

from .dispatch import dual_function


def kkt_report(c, cap, d, g, lam, tol=1e-6):
    c, cap, g = (np.asarray(v, float) for v in (c, cap, g))
    # Stationarity c_i - lam - mu_lo_i + mu_hi_i = 0 pins the bound multipliers
    mu_hi = np.maximum(0.0, lam - c)
    mu_lo = np.maximum(0.0, c - lam)
    primal_obj = float(c @ g)
    dual_obj = float(dual_function(lam, c, cap, d))
    return {
        "primal_feasible": abs(g.sum() - d) < tol
                           and (g > -tol).all() and (g < cap + tol).all(),
        "stationarity": float(np.max(np.abs(c - lam - mu_lo + mu_hi))),
        "comp_slack_upper": float(np.max(mu_hi * (cap - g))),
        "comp_slack_lower": float(np.max(mu_lo * g)),
        "primal_obj": primal_obj,
        "dual_obj": dual_obj,
        "duality_gap": primal_obj - dual_obj,
    }


def kkt_ok(report, tol=1e-5):
    scale = max(1.0, abs(report["primal_obj"]))
    return (report["primal_feasible"]
            and report["stationarity"] < tol
            and report["comp_slack_upper"] < tol * scale
            and report["comp_slack_lower"] < tol * scale
            and abs(report["duality_gap"]) < tol * scale)
