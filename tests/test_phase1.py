import numpy as np
import pytest

from meritorder.dispatch import SOLVERS, solve_all, merit_order
from meritorder.fleet import load_fleet
from meritorder.kkt import kkt_report, kkt_ok
from meritorder.simplex import simplex, LPError

rng = np.random.default_rng(382)


def random_instance(n):
    c = rng.uniform(0, 100, n)
    cap = rng.uniform(50, 1000, n)
    d = rng.uniform(0.05, 0.95) * cap.sum()
    return c, cap, d


@pytest.mark.parametrize("n", [1, 2, 5, 20, 60])
def test_solvers_agree(n):
    for _ in range(25):
        c, cap, d = random_instance(n)
        out = solve_all(c, cap, d)
        ref = out["highs"]
        for name, r in out.items():
            assert r["cost"] == pytest.approx(ref["cost"], rel=1e-7, abs=1e-6), name
            assert r["lam"] == pytest.approx(ref["lam"], rel=1e-7, abs=1e-6), name


@pytest.mark.parametrize("name", list(SOLVERS))
def test_kkt_and_strong_duality(name):
    for _ in range(25):
        c, cap, d = random_instance(15)
        g, cost, lam = SOLVERS[name](c, cap, d)
        assert kkt_ok(kkt_report(c, cap, d, g, lam)), name


def test_price_is_marginal_unit_cost():
    c = np.array([10.0, 30.0, 20.0])
    cap = np.array([100.0, 100.0, 100.0])
    g, cost, lam = merit_order(c, cap, 150)
    assert lam == 20.0
    np.testing.assert_allclose(g, [100, 0, 50])


def test_demand_out_of_range():
    with pytest.raises(ValueError):
        merit_order([1, 2], [10, 10], 25)


def test_simplex_textbook_lp():
    # min -x1 - 2x2  s.t. x1 + x2 + s1 = 4, x1 + 3x2 + s2 = 6
    c = [-1, -2, 0, 0]
    A = [[1, 1, 1, 0], [1, 3, 0, 1]]
    x, obj, y = simplex(c, A, [4, 6])
    np.testing.assert_allclose(x[:2], [3, 1], atol=1e-9)
    assert obj == pytest.approx(-5)
    np.testing.assert_allclose(y, [-0.5, -0.5], atol=1e-9)


def test_simplex_infeasible():
    with pytest.raises(LPError):
        simplex([1, 1], [[1, 1]], [-1])


def test_sample_fleet_loads():
    df = load_fleet("data/sample_fleet.csv")
    assert (df["marginal_cost"] >= 0).all()
    out = solve_all(df["marginal_cost"].values, df["capacity_mw"].values, 12000)
    assert out["simplex"]["lam"] == pytest.approx(out["highs"]["lam"])


def test_degenerate_breakpoint_price_interval():
    # Demand exactly fills units 1 and 2: any lambda in [20, 30] is a valid price
    c = np.array([10.0, 20.0, 30.0])
    cap = np.array([100.0, 100.0, 100.0])
    d = 200.0
    g, _, _ = merit_order(c, cap, d)
    for lam in [20.0, 25.0, 30.0]:
        assert kkt_ok(kkt_report(c, cap, d, g, lam))
    assert not kkt_ok(kkt_report(c, cap, d, g, 31.0))
