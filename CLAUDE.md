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
2. NEXT. Quadratic costs: cost_i = a_i g_i + (b_i/2) g_i^2. Strictly convex QP over {sum g = d, 0 <= g <= cap}. Implement:
   - Projected gradient. Projection onto that set = clip(v - t, 0, cap) with scalar t found by bisection (derive this from KKT of the projection problem, put derivation in docs).
   - Dual ascent: g_i(lambda) = clip((lambda - a_i)/b_i, 0, cap_i), update lambda with the supergradient d - sum g_i(lambda).
   - Newton on the dual where it is smooth (piecewise linear derivative), compare to bisection.
   - Empirical convergence vs theory: step 1/L with L = max b_i, linear rate tied to condition number max b / min b. Plot error vs iteration, vary conditioning.
   - Prices are now continuous in demand (lambda = a_i + b_i g_i at the marginal unit). Show this vs Phase 1 step curve.
   - Need a way to set b_i. Simple defensible choice: b_i so marginal cost rises some % from min to max load, document the assumption.
3. Two regions (Michigan vs rest of MISO) with a transmission limit. Dual on the line constraint = congestion price. Compare to the congestion component of MISO LMPs.
4. Hourly validation over a full year (8,760 solves) against actual Michigan Hub LMPs: correlation, MAE, error breakdown by hour/season/load. Expect underprediction at peaks (no startup costs, ramp limits, unit commitment). Treat that gap as a finding.
5. Scenarios using duals: gas price shock, coal retirement, carbon price.

## Data

- data/sample_fleet.csv: MADE UP illustrative fleet for tests. Never present its numbers as real results.
- scripts/fetch_eia.py: builds data/mi_fleet.csv from EIA API v2 (EIA-860 capacity, EIA-923 heat rates). Needs env var EIA_API_KEY. UNTESTED (written in a sandbox with no network access to EIA). Verify route and field names against the EIA API browser on first run and fix as needed.
- MISO data (load, LMPs incl. energy/congestion/loss components) not pulled yet; needed for Phase 3 and 4.
- data/raw/ is gitignored cache.

## Findings so far

- All four Phase 1 solvers agree on cost to ~1e-11 and on price at every non-degenerate demand level. All pass KKT with ~0 duality gap.
- Degeneracy: when demand lands exactly on a cumulative capacity breakpoint, the price is not unique (any lambda between adjacent unit costs satisfies KKT). On the sample fleet at 15,600 MW, simplex returns 35.50 and HiGHS 39.00. Both correct. run_phase1.py flags these points; a test covers it. Relevant later when comparing to MISO prices.

## Commands

    pip install -r requirements.txt
    python -m pytest -q
    python scripts/run_phase1.py [fleet.csv]
    python scripts/fetch_eia.py --year 2024

## Conventions

- Package code in meritorder/, runners in scripts/run_phaseN.py, derivations in docs/phaseN.md, outputs in results/.
- Every new solver gets an agreement test vs a scipy reference and a KKT check.
- Writing style for docs and README: concise, plain, conversational. No em dashes.
- Jonah is terse and wants you to move fast and make decisions rather than ask questions. Prefer direct code over long explanations. Keep summaries short.
- One commit per meaningful step with a clear message.

## Immediate todo

1. Confirm remote is github.com/JonahXA/meritorder and pushed.
2. Run fetch_eia.py with Jonah's key, fix any API field issues, rerun Phase 1 on data/mi_fleet.csv, sanity check capacity by fuel against known Michigan totals.
3. Start Phase 2.
