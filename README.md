# MeritOrder

Can a cost-minimizing optimization model built only from public plant data reproduce Michigan's real hourly electricity prices, and where does it break?

Research project applying CMSE 382 (Optimization Methods in Data Science, MSU): LP, KKT, duality, convex QP, gradient and Newton methods.

## Status

- [x] Phase 1: single-region dispatch LP, KKT derivation, four solvers including a from-scratch simplex ([docs/phase1.md](docs/phase1.md))
- [x] Phase 2: quadratic costs, projected gradient, dual ascent, dual Newton, convergence vs conditioning ([docs/phase2.md](docs/phase2.md))
- [ ] Phase 3: two regions with a transmission limit, congestion prices
- [ ] Phase 4: hourly validation against MISO Michigan Hub LMPs
- [ ] Phase 5: scenarios (gas prices, retirements, carbon price)

## Quickstart

    pip install -r requirements.txt
    python -m pytest -q
    python scripts/run_phase1.py                  # sample fleet
    export EIA_API_KEY=...                        # free key from eia.gov
    python scripts/fetch_eia.py --year 2024       # builds data/mi_fleet.csv
    python scripts/run_phase1.py data/mi_fleet.csv
    python scripts/run_phase2.py data/mi_fleet.csv

## Layout

    meritorder/   dispatch solvers, simplex, KKT checks, fleet loader
    scripts/      data pulls and phase runners
    tests/        solver agreement, KKT, duality, degeneracy
    docs/         derivations per phase
    results/      figures and CSVs

## Data note

data/sample_fleet.csv is an illustrative, made-up fleet for testing. Real results use data/mi_fleet.csv built from EIA-860 and EIA-923.
