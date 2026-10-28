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
2. DONE (on sample fleet). Quadratic costs a_i g + (b_i/2) g^2 in meritorder/qp.py: projected_gradient (projection by bisection on scalar t, lambda = -L t at the fixed point), dual_ascent, dual_bisection, dual_newton (safeguarded), scipy (trust-constr + active-set crossover, since trust-constr alone sits ~1e-4 MW off bounds). b_i = max(10% c_i, $1) / cap_i, documented as an assumption to calibrate in Phase 4. KKT in kkt.py:qp_kkt_report. Derivation and findings in docs/phase2.md, runner scripts/run_phase2.py.
3. Two regions (Michigan vs rest of MISO) with a transmission limit. Dual on the line constraint = congestion price. Compare to the congestion component of MISO LMPs.
4. Hourly validation over a full year (8,760 solves) against actual Michigan Hub LMPs: correlation, MAE, error breakdown by hour/season/load. Expect underprediction at peaks (no startup costs, ramp limits, unit commitment). Treat that gap as a finding.
5. Scenarios using duals: gas price shock, coal retirement, carbon price.

## Data

- data/sample_fleet.csv: MADE UP illustrative fleet for tests. Never present its numbers as real results.
- scripts/fetch_eia.py: builds data/mi_fleet.csv from EIA API v2 (EIA-860 capacity, EIA-923 heat rates). Needs EIA_API_KEY in env or in .env (gitignored). UNTESTED (written in a sandbox with no network access to EIA). Verify route and field names against the EIA API browser on first run and fix as needed.
- MISO data (load, LMPs incl. energy/congestion/loss components) not pulled yet; needed for Phase 3 and 4.
- data/raw/ is gitignored cache.

## Findings so far

- All four Phase 1 solvers agree on cost to ~1e-11 and on price at every non-degenerate demand level. All pass KKT with ~0 duality gap.
- Degeneracy: when demand lands exactly on a cumulative capacity breakpoint, the price is not unique (any lambda between adjacent unit costs satisfies KKT). On the sample fleet at 15,600 MW, simplex returns 35.50 and HiGHS 39.00. Both correct. run_phase1.py flags these points; a test covers it. Relevant later when comparing to MISO prices.

- Phase 2: QP prices are continuous only where units' cost ranges overlap. Between fuels with a cost gap (renewables to nuclear, nuclear to gas CC, coal to gas CT) the price still jumps. Dual Newton median 7 iterations vs 44 for bisection, ~700x faster than scipy trust-constr. Projected gradient beats its (1 - 1/kappa)^k bound because only free units matter locally. Dual ascent's local rate 1 - sum_free(1/b)/sum_all(1/b) matches observed to 4 digits; it depends on which units are marginal, not on kappa.

## Commands

    pip install -r requirements.txt
    python -m pytest -q
    python scripts/run_phase1.py [fleet.csv]
    python scripts/run_phase2.py [fleet.csv]
    python scripts/fetch_eia.py --year 2024

## Conventions

- Package code in meritorder/, runners in scripts/run_phaseN.py, derivations in docs/phaseN.md, outputs in results/.
- Every new solver gets an agreement test vs a scipy reference and a KKT check.
- Writing style for docs and README: concise, plain, conversational. No em dashes.
- Jonah is terse and wants you to move fast and make decisions rather than ask questions. Prefer direct code over long explanations. Keep summaries short.
- One commit per meaningful step with a clear message.

## Immediate todo

1. Run fetch_eia.py with Jonah's key (.env), fix any API field issues, rerun Phase 1 and 2 on data/mi_fleet.csv, sanity check capacity by fuel against known Michigan totals.
2. Start Phase 3: pull MISO data, two-region model.
