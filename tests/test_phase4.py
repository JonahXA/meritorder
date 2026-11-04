import numpy as np
import pandas as pd
import pytest

from meritorder.fleet import monthly_marginal_costs, DEFAULT_FUEL_PRICES


def test_monthly_marginal_costs_use_state_prices_and_fall_back():
    fleet = pd.DataFrame({"state": ["MI", "IN", "MI", "MI"],
                          "fuel": ["gas", "gas", "nuclear", "coal"],
                          "heat_rate": [7.0, 7.0, 10.4, 10.0], "vom": [3.0, 3.0, 2.5, 4.5]})
    prices = pd.DataFrame({"month": ["2024-01", "2024-01", "2024-02"],
                           "state": ["MI", "IN", "MI"], "fuel": ["gas", "gas", "gas"],
                           "price": [4.0, 5.0, 2.0]})
    mc = monthly_marginal_costs(fleet, prices)
    assert set(mc) == {"2024-01", "2024-02"}
    np.testing.assert_allclose(mc["2024-01"][:2], [31.0, 38.0])
    # nuclear and coal have no row: defaults
    assert mc["2024-01"][2] == pytest.approx(10.4 * DEFAULT_FUEL_PRICES["nuclear"] + 2.5)
    assert mc["2024-02"][3] == pytest.approx(10.0 * DEFAULT_FUEL_PRICES["coal"] + 4.5)
    # Indiana gas has no February row: default too
    assert mc["2024-02"][1] == pytest.approx(7.0 * DEFAULT_FUEL_PRICES["gas"] + 3.0)
