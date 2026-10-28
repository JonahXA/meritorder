"""Single-region economic dispatch, four ways.

Problem (P):  min  sum_i c_i g_i
              s.t. sum_i g_i = d          (multiplier lambda = price)
                   0 <= g_i <= cap_i      (multipliers mu_lo, mu_hi >= 0)

Every solver returns (g, cost, lam).
"""
import time
import numpy as np
from scipy.optimize import linprog

from .simplex import simplex


def _check(c, cap, d):
    c, cap = np.asarray(c, float), np.asarray(cap, float)
    if d < 0 or d > cap.sum() + 1e-9:
        raise ValueError(f"demand {d} outside [0, {cap.sum()}]")
    return c, cap


def merit_order(c, cap, d):
    """Closed form from KKT: fill cheapest units first.

    Price = cost of the marginal (partially loaded) unit.
    """
    c, cap = _check(c, cap, d)
    order = np.argsort(c, kind="stable")
    g = np.zeros_like(c)
    remaining = d
    lam = c[order[0]]
    for i in order:
        if remaining <= 0:
            break
        g[i] = min(cap[i], remaining)
        remaining -= g[i]
        lam = c[i]
    return g, float(c @ g), float(lam)


def dual_function(lam, c, cap, d):
    """q(lambda) = lambda*d + sum_i cap_i * min(0, c_i - lambda). Concave."""
    return lam * d + np.sum(cap * np.minimum(0.0, c - lam))


def dual_bisection(c, cap, d, tol=1e-10, max_iter=200):
    """Maximize the dual by bisection on its supergradient.

    Supergradient of q at lambda: d - sum_{c_i < lambda} cap_i.
    Positive means lambda is too low.
    """
    c, cap = _check(c, cap, d)
    lo, hi = c.min() - 1.0, c.max() + 1.0
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        if d - cap[c < mid].sum() > 0:
            lo = mid
        else:
            hi = mid
        if hi - lo < tol:
            break
    lam = 0.5 * (lo + hi)
    # Snap to the nearest unit cost (the true lambda* is a breakpoint)
    lam = float(c[np.argmin(np.abs(c - lam))])
    # Recover primal: units below price at cap, marginal units fill the rest
    g = np.where(c < lam, cap, 0.0)
    marg = np.isclose(c, lam)
    resid = d - g.sum()
    if marg.any() and resid > 0:
        g[marg] = cap[marg] * resid / cap[marg].sum()
    return g, float(c @ g), lam


def simplex_dispatch(c, cap, d):
    """Our own simplex on standard form: vars [g, s], rows [sum g = d; g + s = cap]."""
    c, cap = _check(c, cap, d)
    n = len(c)
    A = np.zeros((n + 1, 2 * n))
    A[0, :n] = 1.0
    A[1:, :n] = np.eye(n)
    A[1:, n:] = np.eye(n)
    b = np.concatenate([[d], cap])
    cost = np.concatenate([c, np.zeros(n)])
    x, obj, y = simplex(cost, A, b)
    return x[:n], obj, float(y[0])


def highs_dispatch(c, cap, d):
    """Reference answer from scipy's HiGHS."""
    c, cap = _check(c, cap, d)
    res = linprog(c, A_eq=np.ones((1, len(c))), b_eq=[d],
                  bounds=list(zip(np.zeros_like(cap), cap)), method="highs")
    if not res.success:
        raise RuntimeError(res.message)
    return res.x, float(res.fun), float(res.eqlin.marginals[0])


SOLVERS = {
    "merit_order": merit_order,
    "dual_bisection": dual_bisection,
    "simplex": simplex_dispatch,
    "highs": highs_dispatch,
}


def solve_all(c, cap, d):
    out = {}
    for name, fn in SOLVERS.items():
        t0 = time.perf_counter()
        g, cost, lam = fn(c, cap, d)
        out[name] = {"g": g, "cost": cost, "lam": lam,
                     "seconds": time.perf_counter() - t0}
    return out
