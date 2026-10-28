"""Two-phase tableau simplex with Bland's rule.

Solves   min c^T x   s.t.  A x = b,  x >= 0
and returns the primal solution plus the dual vector y (B^T y = c_B),
which is what gives us electricity prices.
"""
import numpy as np


class LPError(Exception):
    pass


def _pivot(T, basis, r, k):
    T[r] /= T[r, k]
    for i in range(T.shape[0]):
        if i != r and T[i, k] != 0:
            T[i] -= T[i, k] * T[r]
    basis[r] = k


def _run(T, basis, n_cols, tol, max_iter):
    """Iterate on tableau T whose last row is the reduced-cost row."""
    for _ in range(max_iter):
        reduced = T[-1, :n_cols]
        entering = np.where(reduced < -tol)[0]
        if entering.size == 0:
            return
        k = entering[0]  # Bland: smallest index
        col = T[:-1, k]
        pos = col > tol
        if not pos.any():
            raise LPError("unbounded")
        ratios = np.full(col.shape, np.inf)
        ratios[pos] = T[:-1, -1][pos] / col[pos]
        best = ratios.min()
        ties = np.where(np.isclose(ratios, best, atol=tol))[0]
        r = min(ties, key=lambda i: basis[i])  # Bland tie break
        _pivot(T, basis, r, k)
    raise LPError("iteration limit reached")


def simplex(c, A, b, tol=1e-9, max_iter=10_000):
    c = np.asarray(c, float)
    A = np.asarray(A, float).copy()
    b = np.asarray(b, float).copy()
    m, n = A.shape

    neg = b < 0
    A[neg] *= -1
    b[neg] *= -1

    # Phase I: minimize sum of artificials
    T = np.zeros((m + 1, n + m + 1))
    T[:m, :n] = A
    T[:m, n:n + m] = np.eye(m)
    T[:m, -1] = b
    T[-1, :n] = -A.sum(axis=0)
    T[-1, -1] = -b.sum()
    basis = list(range(n, n + m))
    _run(T, basis, n + m, tol, max_iter)
    if -T[-1, -1] > 1e-7:
        raise LPError("infeasible")

    # Drive leftover artificials out of the basis, drop redundant rows
    keep = []
    for r in range(m):
        if basis[r] >= n:
            nz = np.where(np.abs(T[r, :n]) > tol)[0]
            if nz.size:
                _pivot(T, basis, r, nz[0])
                keep.append(r)
        else:
            keep.append(r)
    T = np.vstack([T[keep][:, list(range(n)) + [-1]], np.zeros(n + 1)])
    basis = [basis[r] for r in keep]

    # Phase II: original objective in reduced form
    T[-1, :n] = c
    for i, j in enumerate(basis):
        T[-1] -= c[j] * T[i]
    _run(T, basis, n, tol, max_iter)

    x = np.zeros(n)
    x[basis] = T[:-1, -1]

    # Duals: solve B^T y = c_B on the original (sign-corrected) rows
    B = A[keep][:, basis]
    y_kept = np.linalg.solve(B.T, c[basis])
    y = np.zeros(m)
    y[keep] = y_kept
    y[neg] *= -1  # undo the row flips
    return x, float(c @ x), y
