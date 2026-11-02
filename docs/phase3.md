# Phase 3: Two regions and a transmission limit

## Problem

Split MISO into Michigan (M) and the rest of MISO (R), joined by one line carrying flow f into Michigan, with limit F in both directions:

    (T)  min  sum_{i in M} c_i g_i + sum_{j in R} c_j h_j
         s.t. sum_i g_i + f = d_M        multiplier lambda_M  (Michigan price)
              sum_j h_j - f = d_R        multiplier lambda_R  (rest-of-MISO price)
              -F <= f <= F               multipliers nu_lo, nu_hi >= 0
              0 <= g_i <= cap_i,  0 <= h_j <= cap_j

Still an LP, so KKT is necessary and sufficient. Feasible iff each region's demand is within its capacity plus F, and total demand is within total capacity.

## KKT conditions

Lagrangian:

    L = sum c_i g_i + sum c_j h_j - lambda_M (sum g + f - d_M) - lambda_R (sum h - f - d_R)
        + nu_hi (f - F) - nu_lo (f + F) + (unit bound terms as in Phase 1)

Units: exactly Phase 1, each unit against its own region's price. Units cheaper than their region's price run flat out, dearer ones sit idle, and partly loaded ones set the price.

The line: stationarity in f gives

    -lambda_M + lambda_R + nu_hi - nu_lo = 0,   so   lambda_M - lambda_R = nu_hi - nu_lo

with nu_hi (F - f) = 0 and nu_lo (F + f) = 0. Three cases:

- |f| < F: both nu are zero, so lambda_M = lambda_R. An unconstrained line means one price everywhere. The two regions act as one pooled merit order.
- f = F (importing at the limit): lambda_M - lambda_R = nu_hi >= 0. Michigan is dearer by exactly the line's shadow price.
- f = -F (exporting at the limit): lambda_R - lambda_M = nu_lo >= 0.

So **the price difference between the regions is the congestion price**: the multiplier on the line. It's also the value of more line. By the sensitivity interpretation of multipliers, d(cost)/dF = -(nu_hi + nu_lo) = -|lambda_M - lambda_R|. One more MW of transfer capacity saves |lambda_M - lambda_R| dollars per hour. tests/test_phase3.py checks this numerically. The congestion rent (lambda_M - lambda_R) f is what MISO collects and hands back through financial transmission rights.

## Dual

Minimizing L over the unit boxes separates exactly as in Phase 1, and minimizing over f in [-F, F] gives min_f f (lambda_R - lambda_M) = -F |lambda_M - lambda_R|. So

    q(lambda_M, lambda_R) = q_M(lambda_M) + q_R(lambda_R) - F |lambda_M - lambda_R|

where q_M, q_R are the Phase 1 dual functions of each region on its own. Concave in both prices. The last term is the price of disagreement: the two prices can differ, but only at a cost proportional to the line limit. F = 0 decouples into two independent markets. As F goes to infinity, any difference costs infinitely much, which forces one price. meritorder/kkt.py:two_region_kkt_report checks strong duality with this q.

## Solvers

| Solver | Idea | File |
|---|---|---|
| merit_two_region | KKT closed form: solve the pooled merit order; if its implied flow fits within F, done (one price). Otherwise fix f = sign * F and run each region's merit order on its net demand | meritorder/network.py |
| flow_bisection | primal decomposition. V(f) = C_M(d_M - f) + C_R(d_R + f) is convex in f with derivative lambda_R(d_R + f) - lambda_M(d_M - f), so bisect on its sign. Each step is two independent regional dispatches | meritorder/network.py |
| simplex | own simplex from Phase 1 on the standard form (f split as f+ - f-, line slacks) | meritorder/network.py |
| highs | scipy reference | meritorder/network.py |
| flow_bisection_qp | same decomposition with Phase 2 quadratic costs, each region by dual Newton | meritorder/network.py |

Why the closed form is right: if the pooled flow exceeds F, convexity of V says V'(f) <= 0 for every f below the pooled optimum, including F. So importing at the limit is optimal and lambda_M >= lambda_R there, which matches the sign KKT requires.

Price recovery in flow_bisection takes care. With step supply curves, the optimal f lands on a kink of one region's curve (or an edge where that region is at zero or full output). There, that region's price is any value in a range, and the merit order returns one end of it. When the line is slack the code detects which region is on the kink and uses the other region's price, which lies inside the range. The first version got the flows and costs right but returned two different prices with a slack line in 72 of 267 random instances. The KKT check caught it.

Agreement: on 267 random instances with F = 0, finite and unlimited, all four LP solvers agree with HiGHS on cost and both prices, and every solution passes KKT and zero duality gap. On 91 real 2024 hours (every 97th hour, 2,344 units), merit_two_region and flow_bisection match HiGHS on both prices exactly and on cost to 6e-7 (results/phase3_agreement.csv). My dense tableau simplex is too slow for 2,344 units, so it's only verified on the test instances.

## Data

| What | Source | Notes |
|---|---|---|
| Fleet | EIA-860M by balancing authority (fetch_eia.py --region MISO) | 2,344 units, 181 GW. Michigan = state MI within MISO: 271 units, 24,260 MW. 4.9 GW of Michigan plants (Cook, New Covert, Indeck Niles) are in PJM and excluded |
| Hub prices | MISO day-ahead ex-post LMP files, 2024 | LMP = MEC + MCC + MLC. MEC is identical across hubs (max spread $0.008) |
| Zone load | MISO forecast/actual load files, 2024 | Michigan (LRZ7) is only published combined with LRZ2 as LRZ2_7, by both MISO and EIA-930. Michigan = 0.592 x LRZ2_7, the MI share of MI + WI 2024 retail sales. This assumes Michigan and Wisconsin have the same hourly load shape. It also ignores that the Upper Peninsula is in LRZ2 and the PJM part of Michigan is in neither (both a few %, opposite signs) |
| Wind, solar | EIA-930 MISO hourly output | each wind/solar unit's available capacity = its share of installed capacity x actual MISO output that hour. All of 2024-07-01 is null in EIA-930, filled from the neighbor days |

Fuel prices are still the Phase 1 defaults. Ludington pumped storage is still out (see docs/data.md).

## The line limit

There's no public hourly Michigan transfer limit. MISO's real limits are many flowgates whose ratings change with conditions. MISO's annual Planning Resource Auction publishes a Zone 7 Capacity Import Limit, which is the natural calibration target (not looked up yet). Instead of guessing one number, the runner sweeps F from 1,000 to 8,000 MW plus unlimited. The base case is 4,000 MW.

## Results (2024, 8,784 hours)

### Actual Michigan congestion

| | mean | mean abs | share of hours with abs > $1 | corr with Michigan load |
|---|---|---|---|---|
| MCC_MI (vs MISO reference bus) | $0.54 | $2.04 | 52% | -0.10 |
| MCC_MI - mean MCC of other 7 hubs | $2.00 | $3.66 | 73% | 0.30 |

The first row is what MISO reports. The second is closer to what the model computes (Michigan relative to everyone else), but it also absorbs congestion elsewhere. MISO South hubs average -$2.20 to -$3.40 MCC, which makes Michigan look dearer relative to them.

### Model congestion vs the limit

| Limit (MW) | import limit binds | mean congestion | mean congestion when import binds | corr with actual (vs other hubs) |
|---|---|---|---|---|
| 1,000 | 46% | $0.18 | $0.50 | 0.25 |
| 2,000 | 27% | $0.09 | $0.36 | 0.15 |
| 3,000 | 10% | $0.02 | $0.18 | 0.07 |
| 4,000 | 1.4% | $0.001 | $0.10 | 0.01 |
| 6,000 and up | 0% | 0 | | |

results/phase3_limit_sweep.csv, results/phase3_limit_sweep.png.

**Finding 1: one Michigan line can't produce Michigan's congestion.** Even at 1,000 MW, far tighter than any plausible Michigan import capability, the line binds in 46% of hours but the price difference averages $0.50 when it does. Actual congestion averages $2 to $3.66 in absolute value. The reason is the supply stack from Phase 1. Both regions' marginal units sit in the same flat $21 to $29 band of gas combined cycle and coal. Cutting the line forces Michigan up its own curve, but that curve is nearly flat, so its price barely moves. Real Michigan congestion must come from constraints the model doesn't have: the many internal flowgates inside MISO and inside Michigan, which bind at specific nodes rather than on one interface.

**Finding 2: what drives Michigan congestion is MISO wind, not Michigan load.** results/phase3_by_wind.csv:

| MISO wind quintile (mean MW) | actual MCC_MI | model congestion, 1 GW | model Michigan imports, 4 GW |
|---|---|---|---|
| 3,694 | -$0.93 | $0.13 | -43 MW |
| 7,233 | -$0.25 | $0.04 | 249 MW |
| 10,779 | $0.47 | $0.07 | 658 MW |
| 14,806 | $1.60 | $0.23 | 1,455 MW |
| 19,591 | $1.82 | $0.40 | 1,967 MW |

corr(MISO wind, MCC_MI) = 0.29 against -0.10 for Michigan load. When the wind blows in Iowa, Minnesota and the Dakotas, power in the rest of MISO gets cheap, and Michigan (with only 3.8 GW of wind) wants to import it. The model gets this mechanism right: imports climb from roughly zero to 2 GW across the wind quintiles, and congestion rises with them. It's just about 5x too small. That's the same flat-stack story as Finding 1.

**Finding 3: prices are already in the right neighborhood.** At F = 4,000 MW the model's Michigan price averages $26.75 against an actual hub LMP of $29.92, and the rest-of-MISO price averages $26.74 against an actual MEC of $28.63. Hourly correlation of the model's Michigan price with the actual LMP is 0.58. That's before any fuel price calibration, unit commitment, or storage, which is Phase 4's starting point.

## What this means for Phase 4

Most of Michigan's hourly price variation is system-wide (MEC), not congestion. So the two-region model's value is mainly that it gets the system price right with hourly load and wind, not that it predicts Michigan congestion. Phase 4 should:

1. use monthly delivered fuel prices instead of flat defaults
2. explain the residual in the energy price (peaks: no startup costs, ramp limits, or Ludington)
3. treat congestion as a known gap, or use the hub-relative MCC as an exogenous adder

## Reproduce

    python scripts/fetch_eia.py --region MISO
    python scripts/fetch_miso.py --year 2024
    python scripts/run_phase3.py              # ~45 s
