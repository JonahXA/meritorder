"""Phase 5: carbon prices and emissions caps.

An emissions cap over many hours couples them:

    (C)  min  sum_t sum_i c_i g_it
         s.t. sum_i g_it = d_t                 for every hour t   (prices lambda_t)
              0 <= g_it <= cap_it
              sum_t sum_i e_i g_it <= E        one cap for the whole period   (tau >= 0)

Relaxing only the cap with multiplier tau separates (C) by hour:

    q(tau) = -tau E + sum_t min_g { sum_i (c_i + tau e_i) g_it : hour t constraints }

so for a given tau each hour is a Phase 1 merit order with costs c + tau e.
That is a carbon price. Emissions E(tau) are nonincreasing in tau, and the
supergradient of q is E(tau) - E, so bisection on tau solves the dual. The
optimal tau is the shadow price of the cap, in $/tCO2.
"""
import numpy as np
from scipy import sparse
from scipy.optimize import linprog

from .dispatch import merit_order

# EIA CO2 emission coefficients, lb CO2 per MMBtu. Waste/byproduct fuels are
# counted as zero (biogenic or already emitted upstream); see docs/phase5.md.
CO2_LB_PER_MMBTU = {"coal": 210.2, "gas": 117.0, "oil": 161.3}
TONNE_PER_LB = 0.00045359237


def emission_rates(fleet):
    """tCO2 per MWh for each unit: heat rate x fuel emission coefficient."""
    lb = fleet["fuel"].map(CO2_LB_PER_MMBTU).fillna(0.0).values
    return fleet["heat_rate"].values * lb * TONNE_PER_LB


def dispatch_with_carbon(c, caps, demands, e, tau):
    """Hourly merit order with costs c + tau e.

    caps: (T, n) available capacity per hour. Returns (g (T, n), fuel cost,
    emissions, hourly prices). Prices include the carbon adder.
    """
    c, e = np.asarray(c, float), np.asarray(e, float)
    T = len(demands)
    g = np.empty((T, len(c)))
    lam = np.empty(T)
    for t in range(T):
        g[t], _, lam[t] = merit_order(c + tau * e, caps[t], demands[t])
    return g, float((g @ c).sum()), float((g @ e).sum()), lam


def cap_by_bisection(c, caps, demands, e, E, tau_max=1000.0, tol=1e-6, max_iter=100):
    """Dual decomposition: find the carbon price tau that meets cap E.

    E(tau) is a step function, so at the optimal tau some units tie on
    c + tau e and no single Lagrangian minimizer hits the cap exactly: just
    below tau emissions are over the cap, just above they're under. Both
    dispatches minimize the Lagrangian at tau, so any blend of them does too.
    The blend with emissions exactly E is feasible and satisfies
    complementary slackness, so it's optimal (primal recovery).

    Returns (g, cost, emissions, tau, lam, lower_bound). lower_bound is the
    dual value q(tau); with exact primal recovery it equals cost.
    """
    g, cost, em, lam = dispatch_with_carbon(c, caps, demands, e, 0.0)
    if em <= E:                       # cap not binding: tau = 0
        return g, cost, em, 0.0, lam, cost
    if dispatch_with_carbon(c, caps, demands, e, tau_max)[2] > E:
        raise ValueError(f"cap {E} not reachable at tau = {tau_max}")
    lo, hi = 0.0, tau_max
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        if dispatch_with_carbon(c, caps, demands, e, mid)[2] > E:
            lo = mid
        else:
            hi = mid
        if hi - lo < tol * max(1.0, hi):
            break
    tau = hi
    g_hi, cost_hi, em_hi, lam = dispatch_with_carbon(c, caps, demands, e, hi)
    g_lo, _, em_lo, _ = dispatch_with_carbon(c, caps, demands, e, lo)
    lower = cost_hi + tau * (em_hi - E)     # q(tau): weak duality bound
    theta = 0.0 if em_lo <= em_hi else (E - em_hi) / (em_lo - em_hi)
    g = theta * g_lo + (1 - theta) * g_hi
    return g, float((g @ c).sum()), float((g @ e).sum()), tau, lam, lower


def cap_highs(c, caps, demands, e, E):
    """Reference: the whole multi-hour LP with the cap, in HiGHS (sparse).

    Returns (g, cost, tau, lam) with tau = shadow price of the cap (>= 0) and
    lam the hourly prices.
    """
    c, e = np.asarray(c, float), np.asarray(e, float)
    T, n = caps.shape
    rows = np.repeat(np.arange(T), n)
    A_eq = sparse.csr_matrix((np.ones(T * n), (rows, np.arange(T * n))), shape=(T, T * n))
    A_ub = sparse.csr_matrix(np.tile(e, T)[None, :])
    res = linprog(np.tile(c, T), A_ub=A_ub, b_ub=[E], A_eq=A_eq, b_eq=demands,
                  bounds=list(zip(np.zeros(T * n), caps.ravel())), method="highs")
    if not res.success:
        raise RuntimeError(res.message)
    return (res.x.reshape(T, n), float(res.fun), float(-res.ineqlin.marginals[0]),
            res.eqlin.marginals.copy())
