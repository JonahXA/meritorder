# CLAUDE.md

Context for Claude Code sessions on this repo. Read this first.

## Who and why

Owner: Jonah Alsfasser (GitHub: JonahXA), Computational Data Science student at Michigan State.
Purpose: research project demonstrating competence in MSU CMSE 382, Optimization Methods in Data Science (textbook: Beck, Introduction to Nonlinear Optimization). Course topics: optimality conditions, least squares, gradient method, Newton's method, convex optimization, optimization over convex sets, KKT, basic LP, duality.
Goal: a portfolio piece (report PDF + repo + page on jonahx-dev.vercel.app/research, matching his other projects like PredictEdge and ClosingLine).

Showing the math matters as much as results. Every phase should: derive optimality conditions by hand in docs/, implement solvers from scratch, verify against a reference solver, and check KKT/duality numerically.

## Research question

Can a cost-minimizing dispatch model built only from public plant data reproduce Michigan's real hourly electricity prices (MISO Michigan Hub LMPs), and where does it break? Key idea: the dual variable on the demand constraint IS the electricity price.

## Phases

1. DONE. Single-region dispatch LP. Four solvers in meritorder/dispatch.py: merit_order (KKT closed form), dual_bisection (maximize concave dual q(lambda)), simplex (own two-phase tableau, Bland's rule, duals via B^T y = c_B, in meritorder/simplex.py), highs (scipy reference). KKT + duality checks in meritorder/kkt.py. Derivation in docs/phase1.md.
2. DONE (sample and real fleet). Quadratic costs a_i g + (b_i/2) g^2 in meritorder/qp.py: projected_gradient (projection by bisection on scalar t, lambda = -L t at the fixed point), dual_ascent, dual_bisection, dual_newton (safeguarded), scipy (trust-constr + active-set crossover, since trust-constr alone sits ~1e-4 MW off bounds). b_i = max(10% c_i, $1) / cap_i, documented as an assumption to calibrate in Phase 4. KKT in kkt.py:qp_kkt_report. Derivation and findings in docs/phase2.md, runner scripts/run_phase2.py.
3. DONE. Two regions (Michigan in MISO vs rest of MISO) in meritorder/network.py: merit_two_region (KKT closed form: pooled merit order, else line at limit and separate merit orders), flow_bisection (primal decomposition on f), simplex, highs, flow_bisection_qp. KKT/duality in kkt.py:two_region_kkt_report, dual q = q_M + q_R - F|lam_M - lam_R|. Runner scripts/run_phase3.py runs all 8,784 hours of 2024 for a sweep of line limits. Derivation and findings in docs/phase3.md.
4. DONE. scripts/run_phase4.py: Phase 3 model, 8,784 hours of 2024, monthly state fuel prices (scripts/fetch_fuel_prices.py -> data/fuel_prices_2024.csv, fleet.py:monthly_marginal_costs), ablations (flat fuel, no line limit, nameplate renewables), availability sweep fit Jan-Jun / test Jul-Dec with a $1,000 shortage unit, error breakdown, residual regression. Findings in docs/phase4.md.
5. DONE. meritorder/emissions.py (CO2 rates; annual cap by dual decomposition across hours with primal recovery; HiGHS multi-hour reference), meritorder/hourly.py (Year: full-year two-region runs, emissions, Michigan payment, congestion rent, cap bisection, optional coal must-run). scripts/run_phase5.py runs gas x0.75/1.5/2, Michigan coal retirement at 4 GW and unlimited line, worst-half MISO coal retirement, carbon $15 to $100 (+$500 floor), caps -5/-10/-20%, on two baselines (economic; calibrated = 30% coal must-run, matches 2024 coal share 25.0% vs 26.3% actual). Findings in docs/phase5.md.

## Data

- data/sample_fleet.csv: MADE UP illustrative fleet for tests. Never present its numbers as real results.
- scripts/fetch_eia.py: builds data/mi_fleet.csv from EIA API v2 (EIA-860M capacity, OP + SB status; EIA-923 electricity-only heat rates per plant/fuel/tech class). Key from EIA_API_KEY env or .env (gitignored). Tested and working for 2024: 285 units, 29,148 MW, reconciles with the EIA state profile by fuel (results/fleet_check.csv). Decisions documented in docs/data.md: pumped storage (Ludington 2,180 MW) excluded, Palisades out of service in 2024, waste/byproduct fuels as must-run at $0 fuel, heat-rate outliers outside [0.8, 1.5]x typical replaced by tech defaults (11% of thermal MW).
- scripts/fetch_miso.py: MISO public market reports (no key): DA ex-post LMP/MCC/MLC for the 8 hubs (data/miso_lmp_YEAR.csv), actual load by zone (data/miso_load_YEAR.csv), plus EIA-930 MISO hourly wind/solar (data/miso_renew_YEAR.csv). Michigan LRZ7 load is not published alone (only LRZ2_7, by MISO and EIA-930); MI_est = 0.592 x LRZ2_7 from MI/(MI+WI) retail sales. EIA-930 has 2024-07-01 null for MISO wind/solar, filled from neighbor days.
- data/miso_fleet.csv from fetch_eia.py --region MISO: 2,344 units, 181 GW. Fleet CSVs have state and ba columns; 4.9 GW of Michigan plants (Cook, New Covert, Indeck Niles) are in PJM, so Phase 3+ uses state MI within MISO (24,260 MW) as Michigan.
- data/raw/ is gitignored cache.

## Findings so far

- All four Phase 1 solvers agree on cost to ~1e-11 and on price at every non-degenerate demand level. All pass KKT with ~0 duality gap.
- Degeneracy: when demand lands exactly on a cumulative capacity breakpoint, the price is not unique (any lambda between adjacent unit costs satisfies KKT). On the sample fleet at 15,600 MW, simplex returns 35.50 and HiGHS 39.00. Both correct. run_phase1.py flags these points; a test covers it. Relevant later when comparing to MISO prices.

- Phase 2: QP prices are continuous only where units' cost ranges overlap. Between fuels with a cost gap (renewables to nuclear, nuclear to gas CC, coal to gas CT) the price still jumps. Dual Newton median 7 iterations vs 44 for bisection, ~700x faster than scipy trust-constr. Projected gradient beats its (1 - 1/kappa)^k bound because only free units matter locally. Dual ascent's local rate 1 - sum_free(1/b)/sum_all(1/b) matches observed to 4 digits; it depends on which units are marginal, not on kappa.
- Real fleet (2024): Phase 1 solvers agree to 4e-10. Merit order middle is flat: gas CC and coal interleave at $21 to $29 over 13.6 GW, so model prices will be insensitive to load; CC/coal breakeven gas price ~$3.30/MMBtu. Phase 2: kappa = 35,594 (tiny solar sites have huge b_i), projected gradient does not converge in 20,000 iterations; dual methods fine. Hessian-preconditioned PG is exactly the dual method (docs/phase2.md).

- Phase 3: a single Michigan line cannot produce Michigan congestion. Even at a 1 GW limit (binds 46% of hours) model congestion averages $0.50 when binding vs actual |MCC| ~$2 to $3.7, because both sides sit on the same flat gas/coal band. MISO wind output drives MCC_MI (corr 0.29) more than Michigan load (-0.10); the model reproduces that direction (imports 0 to 2 GW across wind quintiles) at ~1/5 the size. Model Michigan price mean $26.75 vs actual LMP $29.92, hourly corr 0.58, before fuel price calibration. No public hourly Michigan transfer limit; runner sweeps 1 to 8 GW, base 4 GW. TODO: look up the MISO PRA Zone 7 Capacity Import Limit to calibrate.

- Phase 4: model gets level (bias -$4.4, MAE $7.58) and within-month timing (corr 0.70 vs Michigan Hub, 0.79 vs MEC) but swings are ~1/7 the market's (std ratio 0.14). Top 5% price hours $83 actual vs $31 model. Cause: 2024 stack goes $18 to $35 over 40 to 120 GW (flat), then a cliff. Residual tracks net load level (corr 0.46, R2 0.21), ramps add little (R2 0.23). Uniform availability derate: best 0.9 on Jan-Jun, modest gain on Jul-Dec; below 0.8 summer hits shortage pricing. Hourly renewables are the most important input (nameplate: bias -$10.7). Line limit irrelevant to prices. Missing: unit commitment, reserve scarcity pricing, offer premiums, seasonal outages.

- Phase 5: economic dispatch burns too little coal (15.5% vs 26.3% actual); a 30% coal must-run block fixes the mix and Phase 4's overnight bias. Envelope theorem check: d cost/d gas multiplier = gas fuel spend to 0.04%. Cap dual decomposition matches a 394k-variable HiGHS week LP to 1e-6 $/t. Gas x2: Michigan price +45%, CO2 +53%. Michigan coal retirement: in-state CO2 -13 to -47% but 55 to 77% leaks to other states; line then binds up to 10% of hours, worth ~$8.5k to $9k/MW-yr. Carbon: abatement saturates at -21% (economic) or -10% (calibrated) since only fuel switching is possible; d lambda/d tau ~ 0.61 t/MWh = marginal unit's emission rate; at $15/t Michigan load pays +$0.97 B vs +$0.17 B MISO resource cost. Cap shadow prices: -5% $3.91, -10% $9.60, -20% $62.06 per t (economic).

## Commands

    pip install -r requirements.txt
    python -m pytest -q
    python scripts/run_phase1.py [fleet.csv]
    python scripts/run_phase2.py [fleet.csv]
    python scripts/fetch_eia.py --region MISO
    python scripts/fetch_miso.py --year 2024
    python scripts/run_phase3.py [--base-limit 4000]
    python scripts/fetch_fuel_prices.py --year 2024
    python scripts/run_phase4.py [--limit 4000]
    python scripts/run_phase5.py [--plots-only]
    python scripts/fetch_eia.py --year 2024

## Conventions

- Package code in meritorder/, runners in scripts/run_phaseN.py, derivations in docs/phaseN.md, outputs in results/.
- Every new solver gets an agreement test vs a scipy reference and a KKT check.
- Writing style for docs and README: concise, plain, conversational. No em dashes.
- Jonah is terse and wants you to move fast and make decisions rather than ask questions. Prefer direct code over long explanations. Keep summaries short.
- One commit per meaningful step with a clear message.

## Immediate todo

1. Report PDF (all five phases: derivations from docs/, key figures, findings) and the portfolio page on jonahx-dev.vercel.app/research.
2. Look up the MISO PRA Zone 7 Capacity Import Limit to set the line limit (affects Phase 3 and the Phase 5 retirement line value).
3. Optional model upgrades: unit commitment with startup costs (would replace the fixed must-run block and fix peak amplitude), seasonal availability, reserves, Ludington pumped storage.
