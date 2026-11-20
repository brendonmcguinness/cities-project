"""Out-of-sample test on a second disaster: the Camp Fire (Paradise, Butte County CA, 8 Nov 2018).

No parameter is refitted. The behavioural rates calibrated on Katrina (S_s, kappa, lam) are applied to Butte County.
Butte's capacity path is taken from Census housing units (Vintage 2020): a step down at the fire to the July 2019
level (relative to the 2014-18 trend), then linear to July 2020. Because the calibrated host-to-host coupling is zero
and host cities are large relative to the displaced population, the network model reduces exactly to the struck place
plus its displaced residents living elsewhere (hosts at phi = 1).
Predictions are compared with: Census population of Butte County (July 2019, 2020) relative to trend; Census net
domestic migration (years to June 2019, 2020); IRS excess out-migration timing (2018-19 vs 2019-20); and destinations
of 2018-19 excess out-migration vs pre-disaster flow shares (2016-17, 2017-18).
Writes output/campfire.json and output/campfire_traj.npz."""
import json
import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from common import HERE, OUT

RAW = HERE / "raw"
FIRE = 2018 + 311 / 365                    # 8 Nov 2018
BUTTE = ("06", "007")
C = json.load(open(OUT / "calibration.json")); B = C["B"]

p = pd.read_csv(RAW / "co-est2020-alldata.csv", encoding="latin1", dtype={"STATE": str, "COUNTY": str})
b = p[(p.STATE == BUTTE[0]) & (p.COUNTY == BUTTE[1])].iloc[0]
h = pd.read_csv(RAW / "HU-EST2020_6.csv", encoding="latin1", dtype={"STATE": str, "COUNTY": str})
hb = h[h.COUNTY == BUTTE[1]].iloc[0]
yrs = np.arange(2014, 2019)
def trend(vals):
    k = np.polyfit(yrs, vals, 1); return lambda t: np.polyval(k, t)
tp = trend([b[f"POPESTIMATE{y}"] for y in yrs]); th = trend([hb[f"HUESTIMATE{y}"] for y in yrs])
pop_rel = {y: b[f"POPESTIMATE{y}"] / tp(y) for y in (2019, 2020)}
hu_rel = {y: hb[f"HUESTIMATE{y}"] / th(y) for y in (2018, 2019, 2020)}
r_nat = np.mean([b[f"NATURALINC{y}"] / b[f"POPESTIMATE{y-1}"] for y in range(2015, 2019)])
dm_base = np.mean([b[f"DOMESTICMIG{y}"] for y in range(2015, 2019)])
dm_obs = {y: float(b[f"DOMESTICMIG{y}"] - dm_base) for y in (2019, 2020)}
P0 = float(tp(FIRE))

t19, t20 = 2019.5 - FIRE, 2020.5 - FIRE
def K_rel(t):                               # step at the fire, then linear between measured July values
    return np.interp(t, [0, t19, t20], [hu_rel[2019], hu_rel[2019], hu_rel[2020]], right=hu_rel[2020])


def simulate(S_s, kappa, lam, T=2.0):
    def rhs(t, y):
        N, D, co, ci = y; phi = N / K_rel(t)
        out = S_s * N * max(phi - 1, 0); ret = kappa * max(1 - phi, 0) * D
        return [r_nat * N * (1 - phi) - out + ret, out - ret - lam * D, out, ret]
    return solve_ivp(rhs, (0, T), [1.0, 0.0, 0.0, 0.0], t_eval=np.linspace(0, T, 401), max_step=0.01, rtol=1e-8, atol=1e-10)


sol = simulate(B["S_s"], B["kappa"], B["lam"])
t, N = sol.t, sol.y[0]
pred = {y: float(np.interp(ty, t, N)) for y, ty in ((2019, t19), (2020, t20))}
# Census-year (July-June) net migration in persons
def net_window(a, b):
    co = np.interp([a, b], t, sol.y[2]); ci = np.interp([a, b], t, sol.y[3]); return float(((ci[1] - ci[0]) - (co[1] - co[0])) * P0)
dm_mod = {2019: net_window(0, t19), 2020: net_window(t19, t20)}
# uncertainty: Katrina parameter covariance
rng = np.random.default_rng(8112018); mu, cov = np.array(B["x_log"]), np.array(B["cov_log"])
mc = np.array([[np.interp(ty, *(lambda s: (s.t, s.y[0]))(simulate(*np.exp(rng.multivariate_normal(mu, cov))))) for ty in (t19, t20)] for _ in range(100)])

# IRS: excess out-migration timing and destinations
def irs(name):
    d = pd.read_csv(RAW / name, encoding="latin1", dtype={"y1_statefips": str, "y1_countyfips": str, "y2_statefips": str, "y2_countyfips": str})
    return d
out = {y: irs(f"countyoutflow{y}.csv") for y in ["1617", "1718", "1819", "1920"]}
def butte_out_total(d):
    r = d[(d.y1_statefips == BUTTE[0]) & (d.y1_countyfips == BUTTE[1]) & (d.y2_statefips == "97") & (d.y2_countyfips == "000")]
    return float(r.n2.iloc[0])
tot = {y: butte_out_total(d) for y, d in out.items()}
base = (tot["1617"] + tot["1718"]) / 2
e1, e2 = tot["1819"] - base, tot["1920"] - base
share_obs = e1 / (e1 + e2)
co = np.interp([0.0, 2019.33 - FIRE, 2020.33 - FIRE], t, sol.y[2])      # IRS filing years end ~April
share_mod = (co[1] - co[0]) / (co[2] - co[0])
def dest(d):
    r = d[(d.y1_statefips == BUTTE[0]) & (d.y1_countyfips == BUTTE[1]) & (d.y2_statefips.astype(int) <= 56) & (d.y2_countyfips != "000")]
    r = r[~((r.y2_statefips == BUTTE[0]) & (r.y2_countyfips == BUTTE[1]))]
    return r.set_index(r.y2_statefips + r.y2_countyfips).n2.clip(lower=0)
D = pd.DataFrame({y: dest(d) for y, d in out.items()}).fillna(0)
D["base"] = (D["1617"] + D["1718"]) / 2; D["excess"] = (D["1819"] - D["base"]).clip(lower=0)
o, pr = D.excess / D.excess.sum(), D.base / D.base.sum()
dest_R2 = float(1 - ((o - pr) ** 2).sum() / ((o - o.mean()) ** 2).sum()); dest_tv = float(0.5 * np.abs(o - pr).sum())

res = dict(event="Camp Fire, Butte County CA, 8 Nov 2018", katrina_parameters={k: B[k] for k in B["names"]},
           housing_rel_trend=hu_rel, natural_increase=r_nat,
           population_rel_trend=dict(observed=pop_rel, predicted=pred, predicted_ci95={y: np.percentile(mc[:, i], [2.5, 97.5]).tolist() for i, y in enumerate((2019, 2020))}),
           net_domestic_migration=dict(observed=dm_obs, predicted=dm_mod),
           irs_first_year_share=dict(observed=share_obs, predicted=float(share_mod)), irs_excess_out=[e1, e2],
           destinations=dict(R2_pre_disaster_shares=dest_R2, misallocated=dest_tv, n_counties=int((D.excess > 0).sum()),
                             top=[(k, float(o[k]), float(pr[k])) for k in o.sort_values(ascending=False).index[:6]]))
np.savez(OUT / "campfire_traj.npz", t=t + FIRE, N=N, K=K_rel(t))
json.dump(res, open(OUT / "campfire.json", "w"), indent=1, default=float)
print(json.dumps(res, indent=1, default=lambda x: round(float(x), 4)))
