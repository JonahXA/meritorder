"""Phase 2: dispatch with quadratic costs.

Problem (Q):  min  sum_i a_i g_i + (b_i/2) g_i^2       (b_i > 0)
              s.t. sum_i g_i = d                       (multiplier lambda = price)
                   0 <= g_i <= cap_i

Strictly convex, so g* is unique. Marginal cost of unit i is a_i + b_i g_i.
Every solver returns (g, cost, lam, iters) and takes an optional
callback(g, lam) called once per iteration (used for convergence plots).
"""
import time
import numpy as np
from scipy.optimize import Bounds, LinearConstraint, minimize


def _check(a, b, cap, d):
    a, b, cap = (np.asarray(v, float) for v in (a, b, cap))
    if (b <= 0).any():
        raise ValueError("all b_i must be > 0 (strict convexity)")
    if d < 0 or d > cap.sum() + 1e-9:
        raise ValueError(f"demand {d} outside [0, {cap.sum()}]")
    return a, b, cap


def quadratic_coeffs(c, cap, spread=0.10, floor=1.0):
    """Turn Phase 1 constant costs into a_i + b_i g_i.

    Marginal cost starts at a_i = c_i at zero output and rises by
    max(spread * c_i, floor) $/MWh by full load. The floor keeps b_i > 0
    for zero-cost units (wind, solar, hydro).
    """
    c, cap = np.asarray(c, float), np.asarray(cap, float)
    return c.copy(), np.maximum(spread * c, floor) / cap


def cost(a, b, g):
    return float(a @ g + 0.5 * b @ (g * g))


def project(v, cap, d, tol=1e-13, max_iter=200):
    """Euclidean projection onto {sum g = d, 0 <= g <= cap}.

    KKT of the projection gives g = clip(v - t, 0, cap) for a scalar t.
    s(t) = sum clip(v - t, 0, cap) is continuous and nonincreasing, so find
    s(t) = d by bisection. Returns (g, t).
    """
    lo, hi = np.min(v - cap), np.max(v)   # s(lo) = sum cap >= d, s(hi) = 0 <= d
    scale = max(1.0, np.max(np.abs(v)))
    for _ in range(max_iter):
        t = 0.5 * (lo + hi)
        if np.clip(v - t, 0, cap).sum() > d:
            lo = t
        else:
            hi = t
        if hi - lo < tol * scale:
            break
    t = 0.5 * (lo + hi)
    return np.clip(v - t, 0, cap), t


def projected_gradient(a, b, cap, d, tol=1e-10, max_iter=200_000, callback=None):
    """g <- P(g - grad f(g) / L) with L = max b_i.

    At a fixed point every interior unit has a_i + b_i g_i = -L t, so the
    price is lambda = -L t from the last projection.

    Stops on the error bound ||g - g*|| <= (L/mu) ||g_{k+1} - g_k||,
    mu = min b_i, so a slow (ill-conditioned) run does not stop early.
    """
    a, b, cap = _check(a, b, cap, d)
    L, kappa = b.max(), b.max() / b.min()
    g, t = project(np.zeros_like(a), cap, d)
    for k in range(1, max_iter + 1):
        g_new, t = project(g - (a + b * g) / L, cap, d)
        step = np.linalg.norm(g_new - g)
        g = g_new
        if callback:
            callback(g, -L * t)
        if kappa * step < tol * (1 + np.linalg.norm(g)):
            break
    return g, cost(a, b, g), float(-L * t), k


def dispatch_at(lam, a, b, cap):
    """Minimizer of the Lagrangian at price lam: g_i(lam) = clip((lam - a_i)/b_i, 0, cap_i)."""
    return np.clip((lam - a) / b, 0, cap)


def dual_function(lam, a, b, cap, d):
    """q(lambda) = lambda d + sum_i min_{0<=g<=cap} (a g + b g^2/2 - lambda g). Concave, C^1."""
    g = dispatch_at(lam, a, b, cap)
    return float(lam * d + a @ g + 0.5 * b @ (g * g) - lam * g.sum())


def _bracket(a, b, cap):
    # Below min a every unit is off; above max(a + b cap) every unit is at cap
    return a.min() - 1.0, (a + b * cap).max() + 1.0


def dual_ascent(a, b, cap, d, tol=1e-9, max_iter=200_000, callback=None):
    """lambda <- lambda + alpha (d - sum g(lambda)), alpha = 1 / sum(1/b_i).

    q'(lambda) = d - sum g_i(lambda) is Lipschitz with constant sum(1/b_i),
    so alpha = 1/L_q is the textbook step.
    """
    a, b, cap = _check(a, b, cap, d)
    alpha = 1.0 / np.sum(1.0 / b)
    lam = np.median(a)
    for k in range(1, max_iter + 1):
        g = dispatch_at(lam, a, b, cap)
        r = d - g.sum()
        lam += alpha * r
        if callback:
            callback(dispatch_at(lam, a, b, cap), lam)
        if abs(r) < tol * max(1.0, d):
            break
    g = dispatch_at(lam, a, b, cap)
    return g, cost(a, b, g), float(lam), k


def dual_bisection(a, b, cap, d, tol=1e-12, max_iter=200, callback=None):
    """Bisection on q'(lambda) = d - sum g(lambda), nonincreasing in lambda."""
    a, b, cap = _check(a, b, cap, d)
    lo, hi = _bracket(a, b, cap)
    for k in range(1, max_iter + 1):
        lam = 0.5 * (lo + hi)
        if d - dispatch_at(lam, a, b, cap).sum() > 0:
            lo = lam
        else:
            hi = lam
        if callback:
            callback(dispatch_at(0.5 * (lo + hi), a, b, cap), 0.5 * (lo + hi))
        if hi - lo < tol * max(1.0, abs(hi)):
            break
    lam = 0.5 * (lo + hi)
    g = dispatch_at(lam, a, b, cap)
    return g, cost(a, b, g), float(lam), k


def dual_newton(a, b, cap, d, tol=1e-12, max_iter=200, callback=None):
    """Safeguarded Newton on q'(lambda) = 0.

    q''(lambda) = -sum_{i free} 1/b_i, where free means 0 < g_i(lambda) < cap_i.
    q' is piecewise linear, so a Newton step lands exactly on the root of the
    current piece. Falls back to bisection when q'' = 0 (no free units) or the
    step leaves the bracket.
    """
    a, b, cap = _check(a, b, cap, d)
    lo, hi = _bracket(a, b, cap)
    lam = 0.5 * (lo + hi)
    for k in range(1, max_iter + 1):
        g = dispatch_at(lam, a, b, cap)
        r = d - g.sum()
        if r > 0:
            lo = lam
        else:
            hi = lam
        if abs(r) < tol * max(1.0, d) or hi - lo < tol * max(1.0, abs(hi)):
            break
        free = (g > 0) & (g < cap)
        curv = np.sum(1.0 / b[free])
        step = lam + r / curv if curv > 0 else np.nan
        lam = step if lo < step < hi else 0.5 * (lo + hi)
        if callback:
            callback(dispatch_at(lam, a, b, cap), lam)
    g = dispatch_at(lam, a, b, cap)
    return g, cost(a, b, g), float(lam), k


def scipy_qp(a, b, cap, d, callback=None):
    """Reference answer from scipy trust-constr plus an active-set crossover.

    trust-constr is interior point and stops with units ~1e-4 MW off their
    bounds whatever the tolerances. So take its active set (units within
    1e-3 MW of a bound), pin those, and solve the remaining equality-
    constrained QP exactly: on free units a_i + b_i g_i = lambda and
    sum g = d give lambda = (d - pinned + sum a/b) / sum 1/b.
    """
    a, b, cap = _check(a, b, cap, d)
    n = len(a)
    x0 = cap * d / cap.sum()
    res = minimize(lambda g: cost(a, b, g), x0, method="trust-constr",
                   jac=lambda g: a + b * g, hess=lambda g: np.diag(b),
                   constraints=[LinearConstraint(np.ones((1, n)), d, d)],
                   bounds=Bounds(np.zeros(n), cap),
                   options={"gtol": 1e-12, "xtol": 1e-14, "maxiter": 5000})
    g = np.clip(res.x, 0, cap)
    g[g < 1e-3] = 0.0
    g[g > cap - 1e-3] = cap[g > cap - 1e-3]
    free = (g > 0) & (g < cap)
    if free.any():
        lam = (d - g[~free].sum() + np.sum(a[free] / b[free])) / np.sum(1 / b[free])
        g[free] = (lam - a[free]) / b[free]
    else:
        # trust-constr multipliers use the opposite sign convention to ours
        lam = -res.v[0][0]
    return g, cost(a, b, g), float(lam), res.nit


QP_SOLVERS = {
    "projected_gradient": projected_gradient,
    "dual_ascent": dual_ascent,
    "dual_bisection": dual_bisection,
    "dual_newton": dual_newton,
    "scipy": scipy_qp,
}


def solve_all(a, b, cap, d):
    out = {}
    for name, fn in QP_SOLVERS.items():
        t0 = time.perf_counter()
        g, f, lam, k = fn(a, b, cap, d)
        out[name] = {"g": g, "cost": f, "lam": lam, "iters": k,
                     "seconds": time.perf_counter() - t0}
    return out
