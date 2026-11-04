import numpy as np
import pandas as pd
import pytest

from meritorder.emissions import (cap_by_bisection, cap_highs, dispatch_with_carbon,
                                  emission_rates)

rng = np.random.default_rng(5)


def random_week(n=12, T=24):
    c = rng.uniform(5, 60, n)
    e = rng.uniform(0, 1.1, n)
    e[c < 15] = 0.0          # some clean units, so tight caps are reachable
    caps = np.tile(rng.uniform(50, 300, n), (T, 1)) * rng.uniform(0.7, 1.0, (T, n))
    d = rng.uniform(0.3, 0.6, T) * caps.sum(1)
    return c, caps, d, e


@pytest.mark.parametrize("cut", [0.95, 0.85, 0.7])
def test_cap_dual_decomposition_matches_full_lp(cut):
    for _ in range(5):
        c, caps, d, e = random_week()
        E = cut * dispatch_with_carbon(c, caps, d, e, 0.0)[2]
        try:
            g, cost, em, tau, lam, lower = cap_by_bisection(c, caps, d, e, E)
        except ValueError:
            continue
        gh, cost_h, tau_h, lam_h = cap_highs(c, caps, d, e, E)
        assert tau == pytest.approx(tau_h, rel=1e-5, abs=1e-6)
        assert cost == pytest.approx(cost_h, rel=1e-9)
        assert lower == pytest.approx(cost_h, rel=1e-7)      # strong duality
        assert em == pytest.approx(E, rel=1e-9)              # cap binds exactly
        np.testing.assert_allclose(lam, lam_h, atol=1e-4)
        np.testing.assert_allclose(g.sum(1), d, rtol=1e-9)


def test_loose_cap_has_zero_price():
    c, caps, d, e = random_week()
    E0 = dispatch_with_carbon(c, caps, d, e, 0.0)[2]
    _, _, em, tau, _, _ = cap_by_bisection(c, caps, d, e, 1.1 * E0)
    assert tau == 0.0 and em == pytest.approx(E0)


def test_emissions_fall_as_carbon_price_rises():
    c, caps, d, e = random_week()
    em = [dispatch_with_carbon(c, caps, d, e, t)[2] for t in [0, 10, 25, 50, 100, 200]]
    assert all(a >= b - 1e-9 for a, b in zip(em, em[1:]))


def test_emission_rates():
    fleet = pd.DataFrame({"fuel": ["coal", "gas", "wind", "waste"],
                          "heat_rate": [10.0, 7.0, 0.0, 0.0]})
    np.testing.assert_allclose(emission_rates(fleet),
                               [10 * 210.2 * 0.00045359237, 7 * 117.0 * 0.00045359237, 0, 0])
