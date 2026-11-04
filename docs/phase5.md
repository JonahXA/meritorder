# Phase 5: Scenarios, read through duals

## Setup

Every scenario reruns all 8,784 hours of 2024 with the Phase 4 model: two regions, 4,000 MW line, monthly state fuel prices, hourly load and wind/solar (meritorder/hourly.py). One full year takes about 7 seconds. A shortage unit at MISO's $1,000/MWh offer cap sits in each region, and it never ran in any scenario here.

**Emissions**: unit CO2 rate = heat rate x EIA emission coefficient (coal 210.2, gas 117.0, oil 161.3 lb CO2/MMBtu). Waste and byproduct fuels count as zero: biogenic, or emitted upstream anyway. That's an accounting choice, and it covers 2.4 GW.

**Two baselines.** Phase 4's model burns too little coal: 15.5% of MISO generation vs 26.3% actual in 2024 (EIA-930), and 45.2% gas vs 39.6%. At 2024 gas prices a combined cycle ($18/MWh) undercuts coal ($29 to $30), so pure economics says switch. Real coal units don't, because long startups, minimum run levels and fuel contracts make them self-commit and run through cheap hours. So every scenario runs on two baselines:

| | economic | calibrated |
|---|---|---|
| Rule | pure cost-based dispatch (Phase 4) | 30% of each coal unit self-commits: offered at $0, costed at its true cost |
| Coal / gas share | 15.5% / 45.2% | 25.0% / 35.8% (actual 26.3% / 39.6%) |
| MISO CO2 | 224 Mt | 261 Mt |
| Mean Michigan price | $25.50 | $23.55 |

The 30% was picked to match the coal share. Price data wasn't used. It also fixes Phase 4's overnight overprediction: the bias in hours ending 1 to 5 goes from +$2.10 to -$0.14. That's out-of-sample support for the self-commitment story. The calibrated baseline treats self-commitment as fixed even when policy changes, which is its main limitation: a high enough carbon price would eventually end it. So the two baselines bracket the answer. "Economic" is how a fully flexible fleet would respond; "calibrated" is how today's behavior would respond if it didn't adapt.

Levels from this model are biased (Phase 4: average prices about $4 low, peaks far too low). Changes between scenarios are more trustworthy than levels, because most of the bias is common to both sides.

## Duals used

| Dual | Meaning here | Where |
|---|---|---|
| lambda_t, hourly price | d(cost)/d(demand) in hour t. What load pays | every scenario |
| lambda_M - lambda_R | congestion price on the Michigan line. Summed over the year, it's the value of 1 MW more line | coal retirement |
| tau, shadow price of an annual CO2 cap | d(cost)/d(cap). The carbon price that would achieve the cap | caps |
| envelope theorem | d(optimal cost)/d(fuel price) = fuel burned at the optimum | gas |

### Envelope theorem check (gas)

For an LP min c(p)^T g, the derivative of the optimal cost with respect to a cost parameter p is (dc/dp)^T g* at the optimal g*. The dispatch response drops out to first order. Scaling the gas fuel price by m, d(cost)/dm at m = 1 should equal the year's spending on gas fuel.

| | finite difference, m = 0.99 to 1.01 | gas fuel spend at baseline | gap |
|---|---|---|---|
| economic | $5.361 B | $5.358 B | 0.04% |
| calibrated | $4.139 B | $4.139 B | 0.02% |

The small gap is the curvature over a 2% step. The LP's cost is piecewise linear in m, so the finite difference straddles a few kinks.

### Annual cap by dual decomposition

One cap on annual emissions couples all 8,784 hours. Written as one LP it would be 2,344 units x 8,784 hours = 20.6 million variables. Relaxing only the cap with multiplier tau gives

    q(tau) = -tau E + sum_t min_g { sum_i (c_i + tau e_i) g_it : hour t's constraints }

which separates by hour, and each hour is just the Phase 3 merit order with costs c + tau e. **The cap's multiplier is a carbon price.** Annual emissions E(tau) are nonincreasing, and the supergradient of q is E(tau) - E, so bisection on tau maximizes q.

One subtlety. E(tau) is a step function, so at the optimal tau* some units tie on c + tau* e and no single Lagrangian minimizer hits the cap exactly: just below tau* emissions are over, just above they're under. The first version returned the "just above" dispatch, which was feasible but over-abated, and cost up to 14% more than optimal in random tests. Both dispatches minimize the Lagrangian at tau*, so any blend does too. The blend with emissions exactly E is feasible and satisfies complementary slackness, so it's optimal (primal recovery). meritorder/emissions.py.

Verification (tests/test_phase5.py, results/phase5_cap_check.csv). On random multi-hour instances the decomposition matches a single HiGHS LP on cost to about 1e-15 relative and on tau to 1e-5, and the dual bound q(tau) equals the optimal cost (strong duality). On one real July week (MISO pooled, 2,344 units x 168 hours = 394k variables):

| cap | tau, decomposition | tau, HiGHS | cost gap | max hourly price gap |
|---|---|---|---|---|
| -4.8% | $5.8725/t | $5.8725/t | 0 | 8e-7 |
| -9.6% | $13.8076/t | $13.8076/t | 0 | 5e-6 |

To be fair about speed: for one week HiGHS is faster (0.6 s vs about 5 s for the bisection). The decomposition matters at full-year scale, where the single LP is 52 times bigger but each bisection step is still 8,784 closed-form merit orders (about 7 s).

## Results

results/phase5_scenarios.csv, phase5_carbon.png, phase5_gas.png.

### Gas price

| gas price | Michigan price (econ / calib) | MISO CO2 (econ / calib) | coal share (econ / calib) |
|---|---|---|---|
| x0.75 | $21.68 / $19.28 | 195 / 246 Mt | 6.0% / 20.3% |
| x1.00 | $25.50 / $23.55 | 224 / 261 Mt | 15.5% / 25.0% |
| x1.50 | $30.96 / $29.91 | 295 / 316 Mt | 34.4% / 39.0% |
| x2.00 | $36.85 / $35.82 | 343 / 354 Mt | 45.5% / 47.9% |

- **Doubling gas raises Michigan prices 45% and MISO emissions 53%** (economic). Coal comes back into the merit order ahead of gas. In the other direction, gas at 0.75x cuts emissions as much as a $15/t carbon price (195 Mt in both), and it lowers prices instead of raising them.
- Doubling gas (mean MISO delivered gas $2.48/MMBtu) adds $11.35/MWh to the Michigan price, about $4.60/MWh per $1/MMBtu. That's below a combined cycle's 7.2 MMBtu/MWh, because as gas gets dearer coal sets the price more often and coal's cost doesn't move.

### Retiring Michigan coal (5.8 GW: Monroe, Campbell, Belle River and small units)

| | economic, 4 GW line | economic, no limit | calibrated, 4 GW line | calibrated, no limit |
|---|---|---|---|---|
| Michigan price change | +$1.05 | +$0.10 | +$1.41 | +$0.40 |
| CO2 from plants in Michigan | -3.3 Mt (-13%) | -3.4 Mt | -15.5 Mt (-47%) | -15.7 Mt |
| MISO CO2 | -0.75 Mt (-0.3%) | -0.70 Mt | -7.0 Mt (-2.7%) | -6.9 Mt |
| Mean Michigan imports | 928 -> 1,288 MW | 1,305 MW | 629 -> 2,222 MW | 2,277 MW |
| Hours line binds | 4.0% | 0 | 9.9% | 0 |
| Value of 1 MW more line (sum of hourly congestion price) | $8,453/MW-yr | 0 | $9,031/MW-yr | 0 |

- **Most of the reduction leaks.** Michigan's in-state emissions fall 13 to 47%, but 55 to 77% of those tonnes reappear in other MISO states, because the energy is replaced by imports, much of it from out-of-state coal. A state-level emissions number overstates what retiring in-state coal does for total emissions.
- **This is where the line matters.** In Phase 3 the Michigan interface barely bound. Without 5.8 GW of in-state coal, imports rise by 0.4 to 1.6 GW and the line binds up to 10% of hours. The congestion dual summed over the year says each extra MW of transfer capacity is worth about $8,500 to $9,000/MW-yr. With an unlimited line the retirement costs Michigan load almost nothing (+$0.10 to +$0.40/MWh). That's the dual's way of saying the binding constraint after retirement is transmission, not generation.
- Retiring the least efficient half of MISO coal instead cuts MISO CO2 by 5% (economic) to 14% (calibrated), several times more than the Michigan-only retirement. The calibrated result is big because that's the coal that was self-committing.

### Carbon price

| $/tCO2 | 0 | 15 | 30 | 50 | 100 | floor ($500) |
|---|---|---|---|---|---|---|
| MISO CO2, economic | 224 | 195 (-13%) | 186 (-17%) | 181 (-19%) | 177 (-21%) | 177 (-21%) |
| MISO CO2, calibrated | 261 | 245 (-6%) | 240 (-8%) | 237 (-9%) | 236 (-10%) | 236 (-10%) |
| Michigan price, economic | $25.50 | $35.69 | $45.19 | $57.31 | $86.63 | |
| Michigan price, calibrated | $23.55 | $31.62 | $39.32 | $49.20 | $73.07 | |

- **Abatement saturates fast.** With no new clean capacity, a carbon price can only switch coal to gas. Economic dispatch gets almost all of it by $30 to $50/t and bottoms out at -21%. If coal self-commitment doesn't respond to the price, the floor is -10%. Deeper cuts need new zero-carbon capacity or retirements, not a higher price on this fleet.
- **Prices rise linearly in tau, with slope = the marginal unit's emission rate.** From the KKT conditions, lambda_t = c_m + tau e_m for the marginal unit m, so d(lambda)/d(tau) = e_m. The fitted slope is about 0.61 tCO2/MWh (economic), between a combined cycle (0.38) and a gas turbine (0.62): gas sets the price in most hours once coal is out.
- **The bill falls on consumers far more than the resource cost does.** At $15/t, MISO's production cost rises $0.17 B for 29 Mt abated, about $6/t. Michigan load alone pays $0.97 B more, because every MWh is priced at the marginal unit's carbon-inclusive cost, including the inframarginal nuclear and wind. Under a real carbon price that gap is carbon revenue plus inframarginal rent, and it matters for who should get the revenue back.
- **Michigan becomes a net exporter** at $30/t and above (economic). Its fleet is gas-heavy relative to the rest of MISO, so a carbon price makes Michigan's combined cycles cheaper than out-of-state coal. In-state CO2 rises (24.4 to 29.7 Mt) while MISO's falls. That's the mirror image of the retirement leakage.

### Annual cap: its shadow price

| cap vs baseline | economic: tau | economic: Michigan price | calibrated: tau |
|---|---|---|---|
| -5% | $3.91/t | $28.24 | $10.06/t |
| -10% | $9.60/t | $32.12 | unreachable (floor -9.8%) |
| -20% | $62.06/t | $64.46 | unreachable |

The shadow price traces the marginal abatement cost curve: cheap at first ($4/t for 5%), then steep as the coal-to-gas switching runs out ($62/t for 20%, against a floor of 21%). The average cost of the -10% cap is $4.20/t ($94 M for 22 Mt), below its $9.60/t marginal price, as it should be for a convex cost curve. The stars in results/phase5_carbon.png sit on the carbon price curve. That's the decomposition's point: a cap and a carbon price are dual descriptions of the same policy.

## Limits

- Fleet fixed at 2024: no new wind, solar, storage or gas build. Every abatement number is fuel switching only.
- Self-commitment is a fixed 30% block, not a decision. A unit commitment model with startup costs would let it respond to prices.
- Peak prices are too low (Phase 4), so price changes driven by peaks (scarcity during retirement) are understated.
- Emissions from waste/byproduct fuels counted as zero.

## Reproduce

    python scripts/run_phase5.py          # ~10 minutes
