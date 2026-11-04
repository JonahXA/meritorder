"""A full year of hourly two-region dispatch (the Phase 4 setup), for scenarios.

Year(root) loads the MISO fleet, hourly load, wind/solar and monthly fuel
prices. Year.run(...) solves every hour with merit_two_region and returns
annual totals plus hourly prices. Year.cap(E) finds the carbon price that
meets an annual CO2 cap by dual decomposition across hours (see emissions.py).
"""
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .emissions import emission_rates
from .fleet import load_fleet, monthly_marginal_costs
from .network import merit_two_region

SHORT_PRICE = 1000.0          # MISO energy offer cap, $/MWh
SHORT_MW = 1e6


@dataclass
class Result:
    hourly: pd.DataFrame       # date, he, lam_mi, lam_rest, flow, short_mw
    gen: np.ndarray            # annual MWh per unit
    cost: float                # fuel + VOM cost, $ (no carbon charge, no shortage)
    co2: float                 # tCO2
    co2_mi: float              # tCO2 from plants located in Michigan
    short_mwh: float           # unserved energy
    mi_payment: float          # sum_t lam_mi,t d_mi,t, $ (what Michigan load pays)
    congestion_rent: float     # sum_t (lam_mi - lam_rest) f, $
    gas_fuel_spend: float = 0.0  # sum_t (c - vom) g over gas units, $

    def summary(self, fleet):
        h = self.hourly
        by_fuel = pd.Series(self.gen, index=fleet["fuel"].values).groupby(level=0).sum()
        out = {"mi_price_mean": h.lam_mi.mean(),
               "mi_price_loadweighted": self.mi_payment / h.MI_est.sum(),
               "rest_price_mean": h.lam_rest.mean(),
               "cost_busd": self.cost / 1e9, "co2_mt": self.co2 / 1e6,
               "co2_mi_mt": self.co2_mi / 1e6,
               "mi_payment_busd": self.mi_payment / 1e9,
               "congestion_rent_musd": self.congestion_rent / 1e6,
               "line_binding_share": np.mean(np.abs(h.lam_mi - h.lam_rest) > 1e-6),
               "mean_import_mw": h.flow.mean(), "short_mwh": self.short_mwh}
        tot = by_fuel.sum()
        for f in ["coal", "gas", "nuclear", "wind", "solar"]:
            out[f"share_{f}"] = by_fuel.get(f, 0.0) / tot
        return out


class Year:
    def __init__(self, root, year=2024, coal_must_run=0.0):
        """coal_must_run: fraction of each coal unit's capacity that self-commits,
        offered at $0 (always runs) but costed at its true cost."""
        root = Path(root)
        f = load_fleet(root / "data/miso_fleet.csv")
        f["must_run"] = False
        if coal_must_run > 0:
            coal = f[f["fuel"] == "coal"].copy()
            coal["capacity_mw"] *= coal_must_run
            coal["must_run"] = True
            coal["unit"] += " must-run"
            f.loc[f["fuel"] == "coal", "capacity_mw"] *= 1 - coal_must_run
            f = pd.concat([f, coal], ignore_index=True)
        self.fleet = f
        self.must_run = f["must_run"].values
        self.mi = (f["state"] == "MI").values
        self.cap0 = f["capacity_mw"].values
        self.wind = (f["fuel"] == "wind").values
        self.solar = (f["fuel"] == "solar").values
        self.gas = (f["fuel"] == "gas").values
        self.coal = (f["fuel"] == "coal").values
        self.vom = f["vom"].values
        self.e = emission_rates(f)
        self.mc = monthly_marginal_costs(f, pd.read_csv(root / f"data/fuel_prices_{year}.csv"))
        load = pd.read_csv(root / f"data/miso_load_{year}.csv")
        ren = pd.read_csv(root / f"data/miso_renew_{year}.csv")
        h = load[["date", "he", "MI_est", "MISO"]].merge(ren, on=["date", "he"])
        h["month"] = h["date"].str[:7]
        self.h = h

    def costs(self, month, gas_mult=1.0, tau=0.0, bid=False):
        """Unit costs for a month: fuel part of gas units scaled, plus carbon.
        bid=True gives the offer used for dispatch: must-run blocks offer $0."""
        c = self.mc[month].copy()
        if gas_mult != 1.0:
            c[self.gas] = self.vom[self.gas] + gas_mult * (c[self.gas] - self.vom[self.gas])
        c = c + tau * self.e
        if bid:
            c[self.must_run] = 0.0
        return c

    def run(self, F=4000.0, gas_mult=1.0, tau=0.0, retire=None):
        """Solve every hour. retire: boolean mask of units removed from the fleet."""
        mi, n = self.mi, len(self.cap0)
        cap_base = self.cap0.copy()
        if retire is not None:
            cap_base[retire] = 0.0
        cost_c = {m: self.costs(m, gas_mult, 0.0) for m in self.mc}
        bid_c = {m: self.costs(m, gas_mult, tau, bid=True) for m in self.mc}
        gen = np.zeros(n)
        T = len(self.h)
        lamM, lamR, flow, short = (np.empty(T) for _ in range(4))
        cost = co2 = co2_mi = gas_spend = 0.0
        wsum, ssum = self.cap0[self.wind].sum(), self.cap0[self.solar].sum()
        for k, r in enumerate(self.h.itertuples()):
            cap = cap_base.copy()
            cap[self.wind] *= r.wind_mw / wsum
            cap[self.solar] *= r.solar_mw / ssum
            b = bid_c[r.month]
            gM, gR, flow[k], _, lamM[k], lamR[k] = merit_two_region(
                np.r_[b[mi], SHORT_PRICE], np.r_[cap[mi], SHORT_MW], r.MI_est,
                np.r_[b[~mi], SHORT_PRICE], np.r_[cap[~mi], SHORT_MW], r.MISO - r.MI_est, F)
            short[k] = gM[-1] + gR[-1]
            g = np.empty(n)
            g[mi], g[~mi] = gM[:-1], gR[:-1]
            gen += g
            c = cost_c[r.month]
            cost += c @ g
            co2 += self.e @ g
            co2_mi += self.e[mi] @ g[mi]
            gas_spend += (c - self.vom)[self.gas] @ g[self.gas]
        hourly = self.h[["date", "he", "MI_est", "MISO"]].copy()
        hourly["lam_mi"], hourly["lam_rest"], hourly["flow"], hourly["short_mw"] = lamM, lamR, flow, short
        return Result(hourly, gen, cost, co2, co2_mi, short.sum(),
                      float(lamM @ self.h.MI_est.values), float((lamM - lamR) @ flow),
                      gas_spend)

    def cap(self, E, F=4000.0, tau_max=500.0, tol=0.01, **kw):
        """Carbon price meeting an annual CO2 cap E (tonnes), by bisection on tau.

        Returns (tau, result) where result is the blend of the dispatches just
        below and above tau that emits exactly E (primal recovery, emissions.py).
        Prices and flows reported are those at tau (just above).
        """
        base = self.run(F=F, tau=0.0, **kw)
        if base.co2 <= E:
            return 0.0, base
        lo, hi = 0.0, tau_max
        r_lo, r_hi = base, self.run(F=F, tau=hi, **kw)
        if r_hi.co2 > E:
            raise ValueError(f"cap not reachable at tau = {tau_max}")
        while hi - lo > tol:
            mid = 0.5 * (lo + hi)
            r = self.run(F=F, tau=mid, **kw)
            if r.co2 > E:
                lo, r_lo = mid, r
            else:
                hi, r_hi = mid, r
        th = 0.0 if r_lo.co2 <= r_hi.co2 else (E - r_hi.co2) / (r_lo.co2 - r_hi.co2)
        blend = Result(r_hi.hourly, th * r_lo.gen + (1 - th) * r_hi.gen,
                       th * r_lo.cost + (1 - th) * r_hi.cost, E,
                       th * r_lo.co2_mi + (1 - th) * r_hi.co2_mi,
                       th * r_lo.short_mwh + (1 - th) * r_hi.short_mwh,
                       r_hi.mi_payment, r_hi.congestion_rent)
        return hi, blend
