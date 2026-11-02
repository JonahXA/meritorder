# Phase 1: Single-region economic dispatch

## Problem

n generators, unit i has marginal cost c_i ($/MWh) and capacity cap_i (MW). Demand is d.

    (P)  min  sum_i c_i g_i
         s.t. sum_i g_i = d            multiplier lambda (free)
              g_i <= cap_i             multiplier mu_hi_i >= 0
              g_i >= 0                 multiplier mu_lo_i >= 0

The feasible set is a polytope and the objective is linear, so (P) is convex and KKT is necessary and sufficient. Feasible iff 0 <= d <= sum cap_i.

## KKT conditions

Lagrangian:

    L = sum c_i g_i - lambda (sum g_i - d) + sum mu_hi_i (g_i - cap_i) - sum mu_lo_i g_i

1. Stationarity: c_i - lambda + mu_hi_i - mu_lo_i = 0 for every i
2. Primal feasibility: sum g_i = d, 0 <= g_i <= cap_i
3. Dual feasibility: mu_hi_i, mu_lo_i >= 0
4. Complementary slackness: mu_hi_i (cap_i - g_i) = 0, mu_lo_i g_i = 0

## What KKT says about prices

From stationarity, three cases per unit:

- c_i < lambda: then mu_hi_i = lambda - c_i > 0, so g_i = cap_i. Cheap units run flat out.
- c_i > lambda: then mu_lo_i = c_i - lambda > 0, so g_i = 0. Expensive units sit idle.
- 0 < g_i < cap_i: both mu are zero, so lambda = c_i.

So the price lambda equals the cost of the partially loaded (marginal) unit. This is exactly the merit order rule, derived instead of assumed. mu_hi_i = lambda - c_i is unit i's profit per MWh, the "inframarginal rent."

Sensitivity: lambda = d(optimal cost)/dd, the cost of serving one more MWh. That's the definition of a wholesale electricity price.

## Dual problem

Minimizing L over the box 0 <= g_i <= cap_i gives

    q(lambda) = lambda d + sum_i cap_i min(0, c_i - lambda)

q is concave and piecewise linear with kinks at each c_i. Its supergradient is d - sum_{c_i < lambda} cap_i, which is positive when lambda is too low (not enough cheap supply) and negative when too high. The dual (D) max_lambda q(lambda) is solved by bisection on that sign. Strong duality holds (LP), so max q = optimal cost of (P). Figure: results/phase1_dual_function.png.

## Degeneracy

If d lands exactly on a cumulative capacity breakpoint, no unit is marginal and every lambda between the last running unit's cost and the next unit's cost satisfies KKT. The price is not unique. In the sample fleet this happens at d = 15,600 MW, where simplex returns 35.50 and HiGHS returns 39.00. Both are correct. Real markets break this tie with offer rules, which is worth noting when comparing to MISO prices in Phase 4.

## Solvers implemented

| Solver | Idea | File |
|---|---|---|
| merit_order | closed form from KKT | meritorder/dispatch.py |
| dual_bisection | maximize q(lambda), recover g from KKT cases | meritorder/dispatch.py |
| simplex | own two-phase tableau simplex, Bland's rule, duals from B^T y = c_B | meritorder/simplex.py |
| highs | scipy reference | meritorder/dispatch.py |

All four agree on cost to ~1e-11 and on price at every non-degenerate demand level; every solution passes the KKT and zero duality gap check in meritorder/kkt.py.

## Real Michigan fleet (2024)

data/mi_fleet.csv: 285 units, 29,148 MW, built from EIA-860M and EIA-923 (see docs/data.md). Run: `python scripts/run_phase1.py data/mi_fleet.csv`. Outputs are tagged `phase1_mi_fleet_*` in results/.

All four solvers agree on cost to 4e-10 and exactly on price at all 49 demand levels, and every solution passes KKT with zero duality gap. Simplex is the slow one (78 ms per solve on 570 variables) but still correct. None of the 49 grid points landed on a capacity breakpoint.

The supply stack (results/phase1_mi_fleet_supply_curve.png), at default fuel prices (gas $3.00, coal $2.10/MMBtu):

| Block | Cumulative MW | Marginal cost ($/MWh) |
|---|---|---|
| hydro, wind, solar (nameplate) | 0 to 5,165 | 0 |
| waste and byproduct fuels | to 5,905 | 5 |
| nuclear (Cook, Fermi) | to 9,223 | 9.8 |
| gas combined cycle and coal, interleaved | to 22,844 | 21 to 29 |
| gas steam and gas turbines | to 28,760 | 34 to 48 |
| oil peakers | to 29,148 | 132 to 255 |

What stands out:

- **The middle of the stack is nearly flat.** 13.6 GW of combined cycle and coal sit within $8/MWh of each other. Any demand in that range gives a price between $21 and $29. Real LMPs vary a lot more than that, so expect the model to underpredict price variation in Phase 4 unless something else (imports, outages, renewables output, startup costs) moves the marginal unit.
- **Gas vs coal ordering depends on the gas price.** The average Michigan CC (7.2 MMBtu/MWh, $3 VOM) breaks even with the average coal unit (10.6 MMBtu/MWh at $2.10, $4.50 VOM) at a gas price of about $3.30/MMBtu. Below that, gas undercuts coal; above it, coal runs first. That's a natural Phase 5 scenario.
- **Wind and solar at nameplate overstate free supply.** 4.9 GW of wind and solar at zero cost assumes they're all producing. Their average output is closer to a third of that, so in practice the stack shifts left and prices rise. Phase 4 needs hourly renewable output.

## Next

Phase 2 replaces c_i g_i with a_i g_i + (b_i/2) g_i^2. The marginal unit condition becomes lambda = a_i + b_i g_i, prices become continuous in demand where units' cost ranges overlap (see docs/phase2.md), and the problem becomes a strictly convex QP solved by projected gradient and dual ascent.
