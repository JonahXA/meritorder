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
4. Hourly validation over a full year (8,760 solves) against actual Michigan Hub LMPs: correlation, MAE, error breakdown by hour/season/load. Expect underprediction at peaks (no startup costs, ramp limits, unit commitment). Treat that gap as a finding.
5. Scenarios using duals: gas price shock, coal retirement, carbon price.

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

## Commands

    pip install -r requirements.txt
    python -m pytest -q
    python scripts/run_phase1.py [fleet.csv]
    python scripts/run_phase2.py [fleet.csv]
    python scripts/fetch_eia.py --region MISO
    python scripts/fetch_miso.py --year 2024
    python scripts/run_phase3.py [--base-limit 4000]
    python scripts/fetch_eia.py --year 2024

## Conventions

- Package code in meritorder/, runners in scripts/run_phaseN.py, derivations in docs/phaseN.md, outputs in results/.
- Every new solver gets an agreement test vs a scipy reference and a KKT check.
- Writing style for docs and README: concise, plain, conversational. No em dashes.
- Jonah is terse and wants you to move fast and make decisions rather than ask questions. Prefer direct code over long explanations. Keep summaries short.
- One commit per meaningful step with a clear message.

## Immediate todo

1. Phase 4: monthly delivered fuel prices (EIA-923 fuel receipts or EIA natural gas/coal price routes) instead of flat defaults, then hourly validation of model Michigan price vs MICHIGAN.HUB LMP (correlation, MAE, by hour/season/load). Error breakdown, especially peaks.
2. Look up the MISO PRA Zone 7 Capacity Import Limit to set the Phase 3 base limit.
3. Consider Ludington pumped storage (2,180 MW) as a simple peak-shaving rule or an intertemporal LP.
