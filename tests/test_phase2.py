import numpy as np
import pytest

from meritorder import dispatch
from meritorder.kkt import qp_kkt_report, kkt_ok
from meritorder.qp import (QP_SOLVERS, solve_all, project, quadratic_coeffs,
                           dual_newton, dual_bisection, projected_gradient)

rng = np.random.default_rng(2)


def random_instance(n, kappa=None):
    a = rng.uniform(0, 100, n)
    cap = rng.uniform(50, 1000, n)
    if kappa is None:
        b = rng.uniform(0.005, 0.1, n)
    else:
        b = 0.01 * np.exp(rng.uniform(0, np.log(kappa), n))
        b[0], b[-1] = 0.01, 0.01 * kappa
    d = rng.uniform(0.05, 0.95) * cap.sum()
    return a, b, cap, d


@pytest.mark.parametrize("n", [1, 2, 5, 20, 60])
def test_qp_solvers_agree(n):
    for _ in range(10):
        a, b, cap, d = random_instance(n)
        out = solve_all(a, b, cap, d)
        ref = out["scipy"]
        for name, r in out.items():
            assert r["cost"] == pytest.approx(ref["cost"], rel=1e-7, abs=1e-6), name
            np.testing.assert_allclose(r["g"], ref["g"], atol=1e-3 * cap.max(), err_msg=name)


@pytest.mark.parametrize("name", list(QP_SOLVERS))
def test_qp_kkt_and_strong_duality(name):
    for _ in range(10):
        a, b, cap, d = random_instance(15)
        g, _, lam, _ = QP_SOLVERS[name](a, b, cap, d)
        assert kkt_ok(qp_kkt_report(a, b, cap, d, g, lam), tol=1e-4), name


def test_projection_matches_kkt_and_is_closest():
    for _ in range(50):
        n = rng.integers(1, 30)
        v = rng.normal(0, 500, n)
        cap = rng.uniform(10, 300, n)
        d = rng.uniform(0, 1) * cap.sum()
        p, t = project(v, cap, d)
        assert p.sum() == pytest.approx(d, abs=1e-6)
        assert (p >= 0).all() and (p <= cap + 1e-12).all()
        # Variational inequality: (v - p) . (x - p) <= 0 for any feasible x
        x = cap * d / cap.sum()
        assert (v - p) @ (x - p) <= 1e-6 * (1 + np.abs(v).sum())


def test_newton_needs_fewer_iterations_than_bisection():
    for _ in range(10):
        a, b, cap, d = random_instance(40)
        assert dual_newton(a, b, cap, d)[3] < dual_bisection(a, b, cap, d)[3]


def test_projected_gradient_rate_bound():
    # ||g_k - g*|| <= (1 - 1/kappa)^k ||g_0 - g*|| with kappa = max b / min b
    a, b, cap, d = random_instance(20, kappa=20)
    g_star = dual_newton(a, b, cap, d)[0]
    errs = []
    projected_gradient(a, b, cap, d, tol=1e-12,
                       callback=lambda g, lam: errs.append(np.linalg.norm(g - g_star)))
    g0 = project(np.zeros_like(a), cap, d)[0]
    e0 = np.linalg.norm(g0 - g_star)
    rate = 1 - b.min() / b.max()
    for k, e in enumerate(errs[:200], start=1):
        assert e <= rate ** k * e0 + 1e-6


def test_qp_tends_to_lp_as_b_shrinks():
    # 0 <= QP cost - LP cost <= sum b cap^2 / 2, and |lam_QP - lam_LP| <= max b cap
    a, _, cap, d = random_instance(10)
    _, lp_cost, lp_lam = dispatch.merit_order(a, cap, d)
    b = np.full(10, 1e-6)
    _, qp_cost, qp_lam, _ = dual_newton(a, b, cap, d)
    assert -1e-6 <= qp_cost - lp_cost <= 0.5 * b @ cap**2
    assert abs(qp_lam - lp_lam) <= (b * cap).max() + 1e-9


def test_quadratic_coeffs_spread_and_floor():
    c = np.array([0.0, 20.0, 50.0])
    cap = np.array([100.0, 200.0, 400.0])
    a, b = quadratic_coeffs(c, cap, spread=0.1, floor=1.0)
    np.testing.assert_allclose(a + b * cap, [1.0, 22.0, 55.0])
    assert (b > 0).all()
