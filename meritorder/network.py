"""Phase 3: two regions joined by one transmission line.

Michigan (M) and the rest of MISO (R), flow f into Michigan, limit F:

    min  sum_{i in M} c_i g_i + sum_{j in R} c_j h_j
    s.t. sum g + f = d_M              multiplier lam_M  (Michigan price)
         sum h - f = d_R              multiplier lam_R  (rest-of-MISO price)
         -F <= f <= F                 multipliers nu_lo, nu_hi >= 0
         0 <= g <= cap_M, 0 <= h <= cap_R

Stationarity in f gives lam_M - lam_R = nu_hi - nu_lo: the price difference
IS the shadow price of the line (the congestion price).

Every solver takes (cM, capM, dM, cR, capR, dR, F) and returns
(gM, gR, f, cost, lam_M, lam_R).
"""
import numpy as np
from scipy.optimize import linprog

from .dispatch import merit_order
from .simplex import simplex


def _check(cM, capM, dM, cR, capR, dR, F):
    cM, capM, cR, capR = (np.asarray(v, float) for v in (cM, capM, cR, capR))
    if F < 0:
        raise ValueError("line limit F must be >= 0")
    if dM < 0 or dR < 0 or dM + dR > capM.sum() + capR.sum() + 1e-9:
        raise ValueError("total demand outside [0, total capacity]")
    if dM > capM.sum() + F + 1e-9 or dR > capR.sum() + F + 1e-9:
        raise ValueError("a region's demand exceeds its capacity plus the line limit")
    return cM, capM, cR, capR


def merit_two_region(cM, capM, dM, cR, capR, dR, F):
    """Closed form from KKT.

    Either the line is slack, so lam_M = lam_R and the two regions act as one
    pooled merit order, or the line is at its limit and each region runs its
    own merit order against its demand net of the fixed flow.
    """
    cM, capM, cR, capR = _check(cM, capM, dM, cR, capR, dR, F)
    nM = len(cM)
    g, cost, lam = merit_order(np.r_[cM, cR], np.r_[capM, capR], dM + dR)
    f = dM - g[:nM].sum()
    if abs(f) <= F + 1e-9:
        return g[:nM], g[nM:], float(f), cost, lam, lam
    f = float(np.sign(f) * F)
    gM, costM, lamM = merit_order(cM, capM, dM - f)
    gR, costR, lamR = merit_order(cR, capR, dR + f)
    return gM, gR, f, costM + costR, lamM, lamR


def flow_bisection(cM, capM, dM, cR, capR, dR, F, tol=1e-9, max_iter=200):
    """Primal decomposition: pick the flow, let each region dispatch itself.

    V(f) = C_M(d_M - f) + C_R(d_R + f) is convex in f, with derivative
    V'(f) = lam_R(d_R + f) - lam_M(d_M - f). Bisect on its sign: if Michigan's
    price is higher, import more.
    """
    cM, capM, cR, capR = _check(cM, capM, dM, cR, capR, dR, F)
    return _flow_bisection(lambda d: merit_order(cM, capM, d),
                           lambda d: merit_order(cR, capR, d),
                           capM.sum(), dM, capR.sum(), dR, F, tol, max_iter)


def flow_bisection_qp(aM, bM, capM, dM, aR, bR, capR, dR, F, tol=1e-9, max_iter=200):
    """Same decomposition with Phase 2 quadratic costs; each region solved by dual Newton.

    Regional prices are now continuous in net demand wherever cost ranges
    overlap, so V'(f) is continuous and the slack-line case is a plain root.
    """
    from .qp import dual_newton
    capM, capR = np.asarray(capM, float), np.asarray(capR, float)
    return _flow_bisection(lambda d: dual_newton(aM, bM, capM, d)[:3],
                           lambda d: dual_newton(aR, bR, capR, d)[:3],
                           capM.sum(), dM, capR.sum(), dR, F, tol, max_iter)


def _flow_bisection(solveM, solveR, totM, dM, totR, dR, F, tol, max_iter):
    """solveX(d) returns (g, cost, lam) for region X at net demand d."""
    lo, hi = max(-F, dM - totM, -dR), min(F, dM, totR - dR)
    if lo > hi + 1e-9:
        raise ValueError("no feasible flow")

    def slope(f):
        return solveR(dR + f)[2] - solveM(dM - f)[2]

    a, b = lo, hi
    if slope(b) <= 0:          # Michigan still dearer at max import
        f = b
    elif slope(a) >= 0:        # rest of MISO dearer at max export
        f = a
    else:
        for _ in range(max_iter):
            m = 0.5 * (a + b)
            if slope(m) < 0:
                a = m
            else:
                b = m
            if b - a < tol * max(1.0, abs(m)):
                break
        f = 0.5 * (a + b)
    gM, costM, lamM = solveM(dM - f)
    gR, costR, lamR = solveR(dR + f)
    if abs(f) < F - 1e-6:
        # Line slack: one price. f* sits where one region's supply curve has a
        # kink (step) or hits an edge (zero or full output); that region's
        # price is then any value in a range, and the other region's price
        # lies in it. Use the other region's.
        netM, netR = dM - f, dR + f
        edgeM = netM < 1e-6 or netM > totM - 1e-6
        edgeR = netR < 1e-6 or netR > totR - 1e-6
        if edgeM != edgeR:
            lamM = lamR = lamR if edgeM else lamM
        else:
            eps = 1e-6
            jumpM = abs(solveM(netM - eps)[2] - solveM(netM + eps)[2])
            jumpR = abs(solveR(netR + eps)[2] - solveR(netR - eps)[2])
            if jumpM > jumpR:
                lamM = lamR
            elif jumpR > jumpM:
                lamR = lamM
            else:
                lamM = lamR = 0.5 * (lamM + lamR)
    return gM, gR, float(f), costM + costR, float(lamM), float(lamR)


def _standard_form(cM, capM, dM, cR, capR, dR, F):
    """x = [g, h, f+, f-, s_g, s_h, s+, s-] >= 0, f = f+ - f-."""
    nM, nR = len(cM), len(cR)
    n = nM + nR
    ncol = 2 * n + 4
    A = np.zeros((n + 4, ncol))
    A[0, :nM] = 1.0
    A[0, n], A[0, n + 1] = 1.0, -1.0              # sum g + f = dM
    A[1, nM:n] = 1.0
    A[1, n], A[1, n + 1] = -1.0, 1.0              # sum h - f = dR
    A[2:2 + n, :n] = np.eye(n)
    A[2:2 + n, n + 2:2 * n + 2] = np.eye(n)       # g + s = cap
    A[n + 2, [n, n + 1, 2 * n + 2]] = [1.0, -1.0, 1.0]    # f + s+ = F
    A[n + 3, [n, n + 1, 2 * n + 3]] = [-1.0, 1.0, 1.0]    # -f + s- = F
    b = np.r_[dM, dR, capM, capR, F, F]
    cost = np.r_[cM, cR, np.zeros(ncol - n)]
    return cost, A, b


def simplex_two_region(cM, capM, dM, cR, capR, dR, F):
    """Our own simplex on the standard form above. Duals of rows 0, 1 are the prices."""
    cM, capM, cR, capR = _check(cM, capM, dM, cR, capR, dR, F)
    nM, n = len(cM), len(cM) + len(cR)
    # An unlimited line is the same as one sized to total capacity
    F = min(F, capM.sum() + capR.sum())
    cost, A, b = _standard_form(cM, capM, dM, cR, capR, dR, F)
    x, obj, y = simplex(cost, A, b)
    return x[:nM], x[nM:n], float(x[n] - x[n + 1]), obj, float(y[0]), float(y[1])


def highs_two_region(cM, capM, dM, cR, capR, dR, F):
    """Reference answer from scipy's HiGHS. Variables [g, h, f]."""
    cM, capM, cR, capR = _check(cM, capM, dM, cR, capR, dR, F)
    nM, nR = len(cM), len(cR)
    A = np.zeros((2, nM + nR + 1))
    A[0, :nM], A[0, -1] = 1.0, 1.0
    A[1, nM:nM + nR], A[1, -1] = 1.0, -1.0
    Fb = None if np.isinf(F) else F
    bounds = (list(zip(np.zeros(nM), capM)) + list(zip(np.zeros(nR), capR))
              + [(None if Fb is None else -Fb, Fb)])
    res = linprog(np.r_[cM, cR, 0.0], A_eq=A, b_eq=[dM, dR], bounds=bounds, method="highs")
    if not res.success:
        raise RuntimeError(res.message)
    x = res.x
    lam = res.eqlin.marginals
    return x[:nM], x[nM:nM + nR], float(x[-1]), float(res.fun), float(lam[0]), float(lam[1])


TWO_REGION_SOLVERS = {
    "merit_two_region": merit_two_region,
    "flow_bisection": flow_bisection,
    "simplex": simplex_two_region,
    "highs": highs_two_region,
}
