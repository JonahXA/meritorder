# Phase 4: Hourly validation against Michigan Hub prices

## Question

Run the Phase 3 model for every hour of 2024 with real inputs. How close does a cost-minimizing dispatch built only from public plant data get to the actual MISO Michigan Hub day-ahead LMP, and where does it break?

## Setup

- **Model**: Phase 3 two-region LP (Michigan within MISO vs the rest of MISO), 4,000 MW line, solved by the KKT closed form (merit_two_region) for each of the 8,784 hours. About 6 seconds per full-year run.
- **Load**: MISO actual load by zone, with Michigan = 0.592 x LRZ2_7 (docs/phase3.md).
- **Wind and solar**: hourly MISO output from EIA-930, shared across units by installed capacity.
- **Fuel prices (new)**: monthly delivered cost to power plants by state from EIA (scripts/fetch_fuel_prices.py). Each plant pays its own state's price that month. EIA withholds many state-months for the electric power sector total, so the script takes the first available of: electric power, all sectors, utilities, independent producers. After that, no gas or coal capacity relies on a cross-state average. Nuclear keeps the $0.70/MMBtu default.
- **Target**: MICHIGAN.HUB day-ahead ex-post LMP. Also the MISO-wide energy component (MEC) for the rest-of-MISO price.

Michigan delivered gas was $3.93/MMBtu in January 2024 (Winter Storm Heather) and $1.87 to $2.41 from February to November, against the flat $3.00 default. At those prices a combined cycle runs at about $18/MWh and undercuts coal ($29 to $30) for most of the year.

## Headline numbers

results/phase4_metrics.csv. "Within-month corr" is the average correlation inside each month, which removes level shifts from fuel prices. "Std ratio" is the model's within-month standard deviation over the market's: below 1 means the model's swings are too small.

| Variant | corr | within-month corr | std ratio | MAE | bias | model mean |
|---|---|---|---|---|---|---|
| **main** (monthly fuel, 4 GW line) | 0.53 | 0.70 | 0.14 | $7.58 | -$4.43 | $25.50 |
| flat_fuel (Phase 1 defaults) | 0.58 | 0.72 | 0.08 | $8.25 | -$3.18 | $26.75 |
| pooled (no line limit) | 0.53 | 0.70 | 0.14 | $7.58 | -$4.43 | $25.49 |
| nameplate_renew (wind/solar always at full output) | 0.54 | 0.58 | 0.28 | $10.95 | -$10.70 | $19.23 |

Actual mean: $29.92. Against the MISO-wide energy price (MEC, mean $28.63) the main model does better: corr 0.58, within-month corr 0.79.

What the ablations say:

- **Monthly fuel prices fix the level, not the shape.** MAE drops 8%, January's mean goes from $27.08 to $34.89 (actual $46.38), and the model's swings double (std ratio 0.08 to 0.14). Overall correlation dips slightly (0.58 to 0.53). Within-month correlation is nearly unchanged, so the dip comes from how the months line up against each other, not from hourly behavior.
- **The line limit doesn't matter for prices.** Pooled and main are identical to the cent, consistent with Phase 3: Michigan congestion isn't an interface problem.
- **Hourly wind and solar matter most.** At nameplate the model is $10.70 too low on average and within-month correlation falls from 0.70 to 0.58. Getting renewables' real output right is the single biggest input.

## Where it breaks

The model gets the **timing** right and the **size** wrong. Within each month it correlates 0.70 with the actual price, but its hourly swings are about a seventh as large.

By Michigan load decile (results/phase4_breakdown.csv):

| Decile | actual | model | bias |
|---|---|---|---|
| 1 (lowest) | $19.00 | $22.31 | +$3.31 |
| 3 | $23.67 | $24.13 | +$0.46 |
| 5 | $28.50 | $25.24 | -$3.26 |
| 8 | $32.22 | $27.31 | -$4.91 |
| 9 | $38.67 | $29.17 | -$9.50 |
| 10 (highest) | $50.73 | $28.62 | -$22.10 |

By actual price level:

| Actual LMP | hours | actual | model | bias |
|---|---|---|---|---|
| $0 to 15 | 302 | $13.16 | $20.94 | +$7.78 |
| $15 to 25 | 3,610 | $21.13 | $23.48 | +$2.35 |
| $25 to 35 | 3,094 | $29.22 | $26.31 | -$2.91 |
| $35 to 50 | 1,254 | $40.80 | $28.22 | -$12.58 |
| $50 to 100 | 428 | $62.37 | $29.73 | -$32.64 |
| over $100 | 96 | $149.12 | $35.16 | -$113.96 |

- **Peaks are underpredicted, as expected.** The top 5% of price hours average $83.39 actual vs $31.12 model, and the top 5% of Michigan load hours $57.22 vs $29.06.
- **Troughs are overpredicted.** Overnight the market clears below the model's floor. The model can't go below the cheapest marginal fossil unit once wind and nuclear are used up. The likely reason is that real units offer below fuel cost at night to avoid shutting down, since minimum run levels and startup costs make staying on cheaper.
- **The daily shape is missing** (results/phase4_profiles.png). The actual price runs $21 overnight, $30 to $31 in the morning, and $43 at hours ending 18 and 19. The model runs $24 to $27 all day.

### Why: a flat stack, then a cliff

With 2024 fuel prices, MISO's dispatchable stack goes from $18/MWh at 40 GW to $35/MWh at 120 GW (April prices). That's 80 GW of supply within $17. Every realistic hour lands somewhere in that band. A pure merit order can only move the price by moving along this curve, and it barely rises.

The residual (actual minus model) tracks the **level** of MISO net load (load minus wind and solar): correlation 0.46. Regressing the residual on net load gives R² = 0.21, about $0.53/MWh of missing price per GW. Adding hour-to-hour up and down ramps only lifts R² to 0.23. So the evening peak is mostly that net load is highest right after solar drops off, and the real supply curve is much steeper than the model's. It isn't mainly a ramp-rate effect.

### Availability sweep

One reason the real curve is steeper is that not every plant is available every hour (planned maintenance, forced outages). The model assumes 100%. The runner scales every non-wind/solar unit by a factor alpha. It adds a shortage unit in each region at MISO's $1,000/MWh energy offer cap, so hours that run short get a scarcity price instead of making the LP infeasible. alpha is chosen on January to June by MAE and scored on July to December (results/phase4_availability.csv, phase4_availability.png).

| alpha | fit MAE (Jan to Jun) | test MAE (Jul to Dec) | test corr | test std ratio |
|---|---|---|---|---|
| 1.00 | $7.71 | $7.46 | 0.63 | 0.13 |
| **0.90** (best on fit) | $7.64 | $7.26 | 0.68 | 0.15 |
| 0.85 | $7.66 | $7.18 | 0.71 | 0.17 |
| 0.80 | $7.79 | $8.30 | 0.45 | 1.10 |
| 0.75 | $8.01 | $17.55 | 0.58 | 3.26 |

The fitted derate (90%) carries over to the second half: test MAE improves 3% and correlation goes from 0.63 to 0.68. But the swings stay at about 15% of the market's until alpha reaches 0.8, and then summer peaks fall off the cliff into $1,000 shortage pricing and the errors explode. A uniform derate can't produce the market's smooth rise from $35 to $150. It either does nothing or causes shortages. What fills that gap in reality, and isn't in an energy-only LP:

1. **Unit commitment.** Startup and no-load costs mean the marginal unit at peak recovers fixed costs over a few hours, which shows up as prices well above its fuel cost. The model treats every unit as already on.
2. **Operating reserves and scarcity pricing.** MISO co-optimizes energy and reserves and has a demand curve for reserves. As reserve margins tighten, the reserve price feeds into the LMP before any actual shortage.
3. **Offer behavior.** Offers include risk premiums and opportunity costs (gas bought day-ahead, limited-energy hydro).
4. **Seasonal outages.** Maintenance concentrates in spring and fall, so a single alpha is too tight in summer and too loose in shoulder months.
5. **Ludington pumped storage** (2,180 MW), excluded since Phase 1. It buys at night and sells at peak, so it would shave the model's peaks (wrong direction for closing this gap) and lift its troughs (right direction).

### What's left over

- Congestion plus losses add $1.29 on average at Michigan Hub. An energy-only price can't capture this (Phase 3).
- January: actual $46.38 vs model $34.89. Almost all of it is January 14 to 18 (Winter Storm Heather), when daily average Michigan Hub prices hit $62 to $177. Spot gas in a cold snap runs far above the monthly delivered average the model uses.
- DA Michigan Hub had no negative-price hours in 2024, and neither does the model, so negative prices aren't a source of error here.

## Answer to the research question

A cost-minimizing dispatch built from public data gets the **average level** within about $4/MWh and the **timing** of price movements right within months (correlation 0.70 with Michigan Hub, 0.79 with the MISO energy price). It reproduces the **mechanisms** in direction: fuel prices set the level, wind sets the overnight floor, and net load moves the price. It **breaks on amplitude**: it predicts a stable $22 to $30 band while the market spans $13 to $150. The underprediction at peaks is a direct consequence of convex dispatch on a flat 2024 supply stack. Everything that makes peaks expensive (commitment costs, reserve scarcity, outages, offer premiums) is either non-convex or missing from public data.

## Reproduce

    python scripts/fetch_fuel_prices.py --year 2024
    python scripts/run_phase4.py          # ~1 minute
