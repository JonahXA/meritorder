import numpy as np
import pytest
from scipy.optimize import minimize_scalar

from meritorder.dispatch import merit_order
from meritorder.kkt import two_region_kkt_report, kkt_ok
from meritorder.network import (TWO_REGION_SOLVERS, merit_two_region, highs_two_region,
                                flow_bisection_qp)
from meritorder.qp import dual_newton, quadratic_coeffs

rng = np.random.default_rng(3)


def random_instance(nM, nR, F=None):
    cM, cR = rng.uniform(0, 100, nM), rng.uniform(0, 100, nR)
    capM, capR = rng.uniform(50, 500, nM), rng.uniform(50, 500, nR)
    if F is None:
        F = rng.uniform(0, 300)
    # Michigan often short on capacity so the line matters. Stay off exact
    # edges (demand = capacity + F), where prices are degenerate intervals
    # (covered in Phase 1).
    dM = rng.uniform(0.3, 1.2) * capM.sum()
    dM = min(dM, 0.98 * capM.sum() + 0.9 * min(F, 1e9))
    dR = rng.uniform(0.1, 0.8) * capR.sum()
    dR = min(dR, capR.sum() - max(0.0, dM - capM.sum()) - 1.0)
    return cM, capM, dM, cR, capR, dR, F


@pytest.mark.parametrize("F", [0.0, 50.0, 200.0, np.inf])
def test_two_region_solvers_agree_and_pass_kkt(F):
    for _ in range(20):
        inst = random_instance(rng.integers(1, 8), rng.integers(1, 12), F)
        out = {k: fn(*inst) for k, fn in TWO_REGION_SOLVERS.items()}
        ref = out["highs"]
        for name, r in out.items():
            assert r[3] == pytest.approx(ref[3], rel=1e-7, abs=1e-6), name
            assert r[4] == pytest.approx(ref[4], abs=1e-6), name
            assert r[5] == pytest.approx(ref[5], abs=1e-6), name
            rep = two_region_kkt_report(*inst, r[0], r[1], r[2], r[4], r[5])
            assert kkt_ok(rep), name


def test_unlimited_line_is_one_pooled_market():
    cM, capM, dM, cR, capR, dR, _ = random_instance(5, 8, np.inf)
    _, _, _, cost, lamM, lamR = merit_two_region(cM, capM, dM, cR, capR, dR, np.inf)
    _, pooled_cost, pooled_lam = merit_order(np.r_[cM, cR], np.r_[capM, capR], dM + dR)
    assert lamM == lamR == pooled_lam
    assert cost == pytest.approx(pooled_cost)


def test_zero_limit_is_two_islands():
    cM, capM, dM, cR, capR, dR, _ = random_instance(5, 8, 0.0)
    dM = min(dM, capM.sum())
    _, _, f, _, lamM, lamR = merit_two_region(cM, capM, dM, cR, capR, dR, 0.0)
    assert f == 0.0
    assert lamM == merit_order(cM, capM, dM)[2]
    assert lamR == merit_order(cR, capR, dR)[2]


def test_congestion_price_is_value_of_more_line():
    # d(cost)/dF = -(lam_M - lam_R) when the import limit binds
    cM = np.array([80.0, 90.0])
    capM = np.array([300.0, 300.0])
    cR = np.array([10.0, 20.0, 30.0])
    capR = np.array([400.0, 400.0, 400.0])
    dM, dR, F = 450.0, 500.0, 100.0
    gM, gR, f, cost, lamM, lamR = highs_two_region(cM, capM, dM, cR, capR, dR, F)
    assert f == pytest.approx(F)
    assert (lamM, lamR) == (pytest.approx(90.0), pytest.approx(20.0))
    cost2 = highs_two_region(cM, capM, dM, cR, capR, dR, F + 1.0)[3]
    assert cost - cost2 == pytest.approx(lamM - lamR)


def test_qp_two_region_matches_1d_minimization():
    for _ in range(10):
        cM, capM, dM, cR, capR, dR, F = random_instance(6, 10)
        aM, bM = quadratic_coeffs(cM, capM, spread=0.3)
        aR, bR = quadratic_coeffs(cR, capR, spread=0.3)
        gM, gR, f, cost, lamM, lamR = flow_bisection_qp(aM, bM, capM, dM, aR, bR, capR, dR, F)
        lo, hi = max(-F, dM - capM.sum(), -dR), min(F, dM, capR.sum() - dR)
        V = lambda x: dual_newton(aM, bM, capM, dM - x)[1] + dual_newton(aR, bR, capR, dR + x)[1]
        ref = minimize_scalar(V, bounds=(lo, hi), method="bounded",
                              options={"xatol": 1e-10})
        assert cost == pytest.approx(min(ref.fun, V(lo), V(hi)), rel=1e-8)
        # KKT: free units' marginal cost equals their region's price
        for a, b, cap, g, lam in [(aM, bM, capM, gM, lamM), (aR, bR, capR, gR, lamR)]:
            free = (g > 1e-6) & (g < cap - 1e-6)
            np.testing.assert_allclose((a + b * g)[free], lam, atol=1e-6)
        # Line: prices equal unless the line is at its limit, and then the
        # importing side is dearer
        if abs(f) < F - 1e-6:
            assert lamM == pytest.approx(lamR, abs=1e-6)
        else:
            assert np.sign(f) * (lamM - lamR) >= -1e-6
