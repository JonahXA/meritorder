# Data: the Michigan fleet

Built by scripts/fetch_eia.py from EIA API v2. Output: data/mi_fleet.csv (one row per plant, fuel, technology class), plus results/fleet_check.csv.

## Sources

- **Capacity**: EIA-860M via `electricity/operating-generator-capacity`, Michigan, December of the chosen year, net summer capacity. Status operating (OP) or standby (SB). Out of service (OA, OS) excluded.
- **Heat rates**: EIA-923 via `electricity/facility-fuel`, annual, `consumption-for-eg-btu / generation` per plant, fuel, and technology class.
- **Cross-check**: EIA state electricity profile via `electricity/state-electricity-profiles/capability`.

## Decisions

1. **Technology classes.** Units are grouped by prime mover: CC (combined cycle: CT + CA + CS together), GT, IC, ST, and so on. Combined cycle parts have to be grouped, because EIA-923 books the fuel to the gas turbine (CT) and much of the generation to the steam part (CA). Gas CC and gas GT at the same plant stay separate, since their heat rates differ by about 4 MMBtu/MWh.
2. **Drop the ALL rows.** EIA-923 facility-fuel returns subtotal rows with `fuel2002 = ALL` and `primeMover = ALL`. Summing them with the detail rows double or triple counts. The first version of the script did this, and it also blended coal and gas into one heat rate per plant.
3. **Electricity-only fuel.** `total-consumption-btu` includes fuel for steam sold by cogeneration plants. With it, Midland Cogeneration Venture came out at 9.2 MMBtu/MWh; with `consumption-for-eg-btu` it's 7.6.
4. **Heat rate outliers.** An annual average heat rate is a poor marginal heat rate for a unit that barely runs, because startup and hot-standby fuel dominate (Dan E Karn's gas steam units: 23.3). In the other direction, cogen fuel allocation can push a unit below anything physically possible (5.0 for a combined cycle). Values outside [0.8, 1.5] x the technology's typical value fall back to that typical value: CC 7.6, GT 11.0, IC 9.0, gas ST 10.5, coal ST 10.6, nuclear 10.4, oil GT 13.0. These are US fleet averages from EIA Electric Power Annual table 8.1. 11% of thermal MW uses a fallback, flagged in the `hr_source` column.
5. **Storage excluded.** Ludington pumped storage (2,180 MW, EIA code WAT + prime mover PS) was originally mapped to zero-cost hydro, which would have made it the single biggest free resource in the state. It's storage: it buys power at night and sells at peak, and is a net consumer. Storage needs an intertemporal model (state of charge linking hours), so it's out of the single-hour dispatch. Worth revisiting in Phase 4, since Ludington clips peak prices.
6. **Waste and byproduct fuels** (blast furnace gas, wood, black liquor, landfill gas, municipal waste; 711 MW) are a `waste` bucket with fuel price 0 and VOM $5/MWh. They're must-run price takers: they burn whatever the mill or landfill produces. Previously these were silently dropped.
7. **Petroleum coke** is mapped to coal, since it's burned in coal units.

## Result (2024)

285 units, 29,148 MW.

| Fuel | Ours (MW) | EIA state profile (MW) | Difference explained by |
|---|---|---|---|
| gas | 13,757 | 13,780 | 24 MW out of service |
| coal | 5,780 | 5,732 | +47 MW petroleum coke (profile counts it as petroleum) |
| wind | 3,774 | 3,774 | |
| nuclear | 3,318 | 4,086 | Palisades (768 MW) out of service all of 2024; restarted in 2025 |
| solar | 1,158 | 1,158 | |
| waste | 711 | 728 | 17 MW out of service |
| oil | 388 | 435 | -47 MW petroleum coke moved to coal |
| hydro | 262 | 264 | 2 MW out of service |
| pumped storage | excluded | 2,180 | storage, see above |

Every fuel reconciles to within known out-of-service units. Capacity-weighted heat rates by technology: gas CC 7.2, gas GT 11.8, coal 10.6, nuclear 10.4. These match national fleet averages.

## Known limitations

- Wind and solar are at nameplate (net summer) capacity. Actual hourly output is far lower (wind capacity factor ~35%). Phase 1 and 2 treat them as fully available, which pushes the supply curve right. Phase 4 needs hourly wind and solar output from MISO.
- Fuel prices are flat defaults in meritorder/fleet.py (gas $3.00, coal $2.10/MMBtu). Phase 4 should use monthly delivered prices from EIA-923.
- Michigan imports from and exports to the rest of MISO. Phase 3 handles this.
