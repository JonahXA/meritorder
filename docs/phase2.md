# Phase 2: Quadratic costs

## Problem

Real units get less efficient as they push toward full output, so marginal cost rises with output. Model it with a quadratic cost:

    (Q)  min  sum_i a_i g_i + (b_i/2) g_i^2       b_i > 0
         s.t. sum_i g_i = d                        multiplier lambda (free)
              0 <= g_i <= cap_i                    multipliers mu_lo_i, mu_hi_i >= 0

Unit i's marginal cost is a_i + b_i g_i. The Hessian is diag(b), positive definite, so (Q) is strictly convex: g* is unique. Constraints are linear, so KKT is necessary and sufficient (Slater not even needed for affine constraints).

## Choosing b_i

There's no public heat-rate curve per unit, so this is an assumption. Keep a_i = c_i from Phase 1 (marginal cost at zero output) and let marginal cost rise by max(s c_i, floor) $/MWh from zero to full load:

    b_i = max(s c_i, floor) / cap_i,     s = 10%, floor = $1/MWh

The floor keeps b_i > 0 for zero-cost units (wind, solar, hydro), which is needed for strict convexity. As s and floor go to 0, (Q) goes back to the Phase 1 LP. The test suite checks this limit: 0 <= cost_QP - cost_LP <= sum b_i cap_i^2 / 2 and |lambda_QP - lambda_LP| <= max b_i cap_i. The 10% is a placeholder until Phase 4 calibrates it against MISO prices.

## KKT conditions

Stationarity: a_i + b_i g_i - lambda + mu_hi_i - mu_lo_i = 0. Same three cases as Phase 1, with c_i replaced by the marginal cost at the current output:

- a_i + b_i cap_i <= lambda: g_i = cap_i
- a_i >= lambda: g_i = 0
- otherwise a_i + b_i g_i = lambda, so g_i = (lambda - a_i) / b_i

All three collapse into one formula:

    g_i(lambda) = clip((lambda - a_i) / b_i, 0, cap_i)

Every free (partially loaded) unit has marginal cost exactly lambda. In Phase 1 at most one unit was marginal. Now many can be at once, each sitting partway up its own ramp.

## Dual function

Minimizing the Lagrangian over the box separates by unit, and the minimizer is g_i(lambda) above:

    q(lambda) = lambda d + sum_i [ a_i g_i(lambda) + (b_i/2) g_i(lambda)^2 - lambda g_i(lambda) ]

q is concave and, unlike Phase 1, differentiable (Danskin, since the inner minimizer is unique):

    q'(lambda)  = d - sum_i g_i(lambda)
    q''(lambda) = - sum_{i free} 1/b_i

q' is continuous and piecewise linear with kinks where a unit hits 0 or cap. Its Lipschitz constant is L_q = sum_i 1/b_i.

## Prices: mostly continuous, not always

Total supply at price lambda, D(lambda) = sum g_i(lambda), is continuous and nondecreasing. The price for demand d is any lambda with D(lambda) = d. Where D is strictly increasing (at least one free unit), the price is unique and continuous in d. That's the "ramps instead of steps" in results/phase2_price_curve.png.

But if one unit's range [a_i, a_i + b_i cap_i] ends before the next unit's a_j starts, D is flat on that gap, and the price jumps there just like Phase 1. On the sample fleet this happens at four boundaries: renewables to nuclear (3,600 MW: renewables top out at $1, nuclear starts at $9.78), nuclear to gas combined cycle (6,750 MW), coal to gas combustion turbines (15,000 MW), and gas CT to the oil peaker (16,050 MW). Between 6,750 and 15,000 MW the gas CC and coal ranges overlap (gas_cc_C runs $24.40 to $26.84, coal_A starts at $25.92), so several units are marginal at once and the price really is continuous there. So quadratic costs smooth the curve where units' cost ranges overlap but don't remove jumps between fuels with a cost gap. At a jump the price is an interval again (degenerate, same as Phase 1). run_phase2.py flags demand levels with no free unit; none of the 49 grid points land exactly on one.

## Projection onto the feasible set

Projected gradient needs P(v) = argmin { ||g - v||^2 / 2 : sum g = d, 0 <= g <= cap }. This is itself a strictly convex QP, with a = -v and b = 1 in the notation above. With t the multiplier on sum g = d, stationarity is g_i - v_i + t + mu_hi_i - mu_lo_i = 0, so by the same three cases

    P(v)_i = clip(v_i - t, 0, cap_i)

for the scalar t that makes sum_i clip(v_i - t, 0, cap_i) = d. The left side s(t) is continuous and nonincreasing in t, with s(min(v - cap)) = sum cap >= d and s(max v) = 0 <= d, so bisection on t finds it. Cost O(n log(1/tol)). meritorder/qp.py:project. Tests check feasibility and the projection inequality (v - P(v))^T (x - P(v)) <= 0 for feasible x.

## Solvers implemented

| Solver | Idea | Theory |
|---|---|---|
| projected_gradient | g <- P(g - grad f / L), L = max b_i | f is mu-strongly convex (mu = min b) and L-smooth, so \|\|g_k - g*\|\| <= (1 - 1/kappa)^k \|\|g_0 - g*\|\|, kappa = max b / min b |
| dual_ascent | lambda <- lambda + (d - sum g(lambda)) / L_q | step 1/L_q on a concave C^1 function; locally linear, see below |
| dual_bisection | bisection on sign of q'(lambda) | one bit per step |
| dual_newton | lambda <- lambda + (d - sum g) / sum_{free} 1/b, bisection fallback when no unit is free or the step leaves the bracket | q' piecewise linear, so a step that stays on one piece lands exactly on the root |
| scipy | trust-constr plus an active-set crossover | reference |

Price recovery for projected gradient: at a fixed point g = P(g - grad f / L), so every free unit has v_i - t = g_i, i.e. a_i + b_i g_i = -L t. So lambda = -L t from the last projection. Units at cap or 0 satisfy the KKT inequalities with the same lambda, so this lambda is always a valid price.

Stopping rule for projected gradient: strong convexity gives ||g - g*|| <= (L/mu) ||g_{k+1} - g_k||, so the code stops on kappa * step < tol. Stopping on the step alone stops too early when kappa is large (caught by the agreement tests).

Reference solver note: scipy's trust-constr is interior point and stops with units about 1e-4 MW off their bounds no matter how tight the tolerances, which made it look *more expensive* than our solvers by about 0.004 $/h. SLSQP fails outright on these scales. So the reference takes trust-constr's active set (units within 1e-3 MW of a bound), pins them, and solves the remaining equality-constrained QP exactly: lambda = (d - pinned + sum_free a/b) / sum_free 1/b. This is the standard interior point "crossover." The active set comes from scipy, not from our code.

## Results (sample fleet, 15 units, 16,250 MW, kappa = 251)

All five solvers agree with the reference to within 1e-3 $/h in cost (dual ascent is the loosest since it stops on the dual residual; the others agree to 1e-6 or better, Newton to 1e-10), 1e-7 $/MWh in price, and every solution passes the KKT and duality gap check. Full table: results/phase2_agreement.csv.

| | median iterations | mean time |
|---|---|---|
| dual_newton | 7 | 0.06 ms |
| dual_bisection | 44 | 0.14 ms |
| dual_ascent | 154 | 1.0 ms |
| projected_gradient | 1,655 | 184 ms |
| scipy trust-constr | 69 | 42 ms |

Exploiting the structure (one scalar dual variable) beats the general-purpose QP solver by about 700x.

## Convergence vs theory

results/phase2_convergence.png and results/phase2_convergence.csv. 30 synthetic units, b_i log-uniform with kappa in {1, 10, 100, 1000}.

| kappa | PG observed rate | PG bound 1 - 1/kappa | dual ascent observed | dual ascent local theory |
|---|---|---|---|---|
| 1 | converges in 1 step | 0 | 0.8324 | 0.8333 |
| 10 | 0.759 | 0.900 | 0.8674 | 0.8694 |
| 100 | 0.951 | 0.990 | 0.8452 | 0.8452 |
| 1000 | 0.988 | 0.999 | 0.9538 | 0.9538 |

Two findings:

1. **Projected gradient beats its bound, and the bound is the wrong number to look at.** The bound uses the global kappa = max b / min b over all units. Near the solution only the free units move, and the iteration lives on the affine set sum g = d, so the local rate is governed by the b_i of the free units only. Units pinned at 0 or cap stop mattering once identified. With kappa = 1, P(g - grad f / L) solves the problem in one step, since the gradient step lands on the unconstrained optimum shifted by a constant, and projection removes the constant.

2. **Dual ascent's rate is predicted almost exactly by local theory.** Near lambda*, q' is linear with slope -sum_{free} 1/b_i, so the step 1/L_q contracts the error by exactly 1 - (sum_{free} 1/b_i) / (sum_{all} 1/b_i). Observed matches to 4 digits at kappa = 100 and 1000. This rate depends on *which* units are marginal, not on kappa. It's slow when the marginal units are stiff (big b) and many flexible units are pinned at their bounds. Those pinned units inflate L_q but don't help locally. That's why kappa = 1000 is slowest for dual ascent, but kappa = 10 is slower than kappa = 100.

Newton on the dual converges in 3 to 11 steps at every demand level (median 7), against 44 for bisection (results/phase2_newton.png). Each Newton step solves the current linear piece of q' exactly, so it terminates once it lands on the right piece instead of converging asymptotically.

## Next

Phase 3: two regions with a transmission limit. The dual on the line constraint is the congestion price.
