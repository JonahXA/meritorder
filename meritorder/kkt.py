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


def qp_kkt_report(a, b, cap, d, g, lam, tol=1e-6):
    """Same checks for the quadratic-cost problem (Phase 2).

    Marginal cost m_i = a_i + b_i g_i replaces c_i in stationarity.
    """
    from .qp import cost, dual_function as qp_dual
    a, b, cap, g = (np.asarray(v, float) for v in (a, b, cap, g))
    m = a + b * g
    mu_hi = np.maximum(0.0, lam - m)
    mu_lo = np.maximum(0.0, m - lam)
    primal_obj = cost(a, b, g)
    dual_obj = qp_dual(lam, a, b, cap, d)
    return {
        "primal_feasible": abs(g.sum() - d) < tol * max(1.0, d)
                           and (g > -tol).all() and (g < cap + tol).all(),
        "stationarity": float(np.max(np.abs(m - lam - mu_lo + mu_hi))),
        "comp_slack_upper": float(np.max(mu_hi * (cap - g))),
        "comp_slack_lower": float(np.max(mu_lo * g)),
        "primal_obj": primal_obj,
        "dual_obj": dual_obj,
        "duality_gap": primal_obj - dual_obj,
    }


def two_region_kkt_report(cM, capM, dM, cR, capR, dR, F, gM, gR, f, lamM, lamR, tol=1e-6):
    """KKT and strong duality for the two-region LP (Phase 3).

    Units: same conditions as Phase 1, each against its own region's price.
    Line: lam_M - lam_R = nu_hi - nu_lo with nu_hi (F - f) = 0, nu_lo (F + f) = 0.
    Dual: q = q_M(lam_M) + q_R(lam_R) - F |lam_M - lam_R|.
    """
    from .dispatch import dual_function
    cM, capM, cR, capR, gM, gR = (np.asarray(v, float) for v in (cM, capM, cR, capR, gM, gR))
    c, cap, g = np.r_[cM, cR], np.r_[capM, capR], np.r_[gM, gR]
    lam = np.r_[np.full(len(cM), lamM), np.full(len(cR), lamR)]
    mu_hi = np.maximum(0.0, lam - c)
    mu_lo = np.maximum(0.0, c - lam)
    cong = lamM - lamR
    if abs(cong) < tol:     # rounding, not congestion (matters when F = inf)
        cong = 0.0
    nu_hi, nu_lo = max(0.0, cong), max(0.0, -cong)
    slack_hi = nu_hi * (F - f) if nu_hi else 0.0
    slack_lo = nu_lo * (F + f) if nu_lo else 0.0
    line_term = F * abs(cong) if cong else 0.0
    primal_obj = float(c @ g)
    dual_obj = float(dual_function(lamM, cM, capM, dM) + dual_function(lamR, cR, capR, dR)
                     - line_term)
    scale = max(1.0, dM + dR)
    return {
        "primal_feasible": abs(gM.sum() + f - dM) < tol * scale
                           and abs(gR.sum() - f - dR) < tol * scale
                           and abs(f) <= F + tol * scale
                           and (g > -tol).all() and (g < cap + tol).all(),
        "stationarity": float(np.max(np.abs(c - lam - mu_lo + mu_hi))),
        "comp_slack_upper": float(max(np.max(mu_hi * (cap - g)), slack_hi)),
        "comp_slack_lower": float(max(np.max(mu_lo * g), slack_lo)),
        "primal_obj": primal_obj,
        "dual_obj": dual_obj,
        "duality_gap": primal_obj - dual_obj,
    }
