"""Is tolerated occupancy an artefact of how housing capacity is measured?  Plus profile-likelihood intervals.

(1) Capacity cross-check. Five counties of the ten-county panel have ACS 1-year coverage: Orleans, Jefferson,
Harrison, Jackson (Katrina) and Butte (Camp Fire). For the Katrina counties the ACS has no pre-event years, so
ACS usable housing (units minus other-vacant and seasonal) is taken relative to the Census 2000 usable share of
units times the 2000-05 trend of Census housing estimates. Butte uses the ACS capacity of 13_acs_panel.py.
Each county is fitted with Census capacity and with ACS capacity under the same model (kappa fixed at the
ten-county panel value; S_s, lambda and phi* per county; phi* in [0.6, 1.6]; profile grid 0.70-1.50 in steps of 0.02).
(2) Profile likelihood for phi* in all ten counties (Census capacity) and in the ACS check: phi* on a grid,
S_s and lambda re-optimised, 95% interval where chi2 rises by less than 3.84. kappa is held at the panel value,
so the intervals are conditional on it. Writes output/capacity_check.json."""
import json, os, pickle, urllib.request
import numpy as np
from scipy.optimize import least_squares
from scipy.stats import spearmanr
from common import OUT, counties
from struck import predictions, capacity_piecewise

SIG_POP, SIG_SHARE = 0.005, 0.03
PANEL = json.load(open(OUT / "events.json")); KAPPA = PANEL["models"]["B_tolerated_occupancy"]["kappa"]
EV = {e["name"]: e for e in pickle.load(open(OUT / "events_inputs.pkl", "rb"))}
for e in EV.values():
    e["K"] = (lambda kt, kv: (lambda s: np.interp(s, kt, kv)))(e["Kt"], e["Kv"])
    if e.get("fy_end") is None or np.isnan(e["share_obs"]): e["share_obs"] = np.nan
ACSIN = {(e["fips"], e["disaster"]): e for e in pickle.load(open(OUT / "acs_inputs.pkl", "rb"))}


def resid(e, K, S, lam, phis):
    pop, share, _ = predictions(e, K, S, KAPPA, lam, phis)
    r = (pop - e["pop_rel"]) / SIG_POP
    return r if np.isnan(e["share_obs"]) else np.r_[r, (share - e["share_obs"]) / SIG_SHARE]


def fit(e, K, phis=None, starts=(), light=False):
    """Least squares for one county; phis None = free. Multi-start, including any given (S, lam, phi*) starts."""
    defaults = [(np.exp(0.5), np.exp(-0.7), 1.0), (np.exp(1.5), np.exp(-1.5), 1.1), (np.exp(-0.5), 1.0, 0.9), (3.0, 0.1, 1.0), (0.3, 2.0, 1.0)]
    cand = list(starts) + (defaults[:1] if light else defaults)
    best = None
    for S0, l0, p0 in cand:
        if phis is None:
            f = lambda x: resid(e, K, np.exp(x[0]), np.exp(x[1]), x[2])
            x0, lo, hi = [np.log(S0), np.log(l0), min(max(p0, 0.61), 1.59)], [-5, -6, 0.6], [5, 3, 1.6]
        else:
            f = lambda x: resid(e, K, np.exp(x[0]), np.exp(x[1]), phis); x0, lo, hi = [np.log(S0), np.log(l0)], [-5, -6], [5, 3]
        x0 = np.clip(x0, np.array(lo) + 1e-6, np.array(hi) - 1e-6)
        r = least_squares(f, x0, bounds=(lo, hi), max_nfev=600, x_scale="jac")
        if best is None or r.cost < best.cost: best = r
    x = best.x
    return dict(chi2=float(2 * best.cost), S=float(np.exp(x[0])), lam=float(np.exp(x[1])), phis=float(x[2]) if phis is None else phis)


def profile(e, K, start=None, grid=np.round(np.arange(0.70, 1.501, 0.02), 2)):
    """Profile chi2 over phi*, by continuation outward from the best free fit (each grid point also multi-started)."""
    m = fit(e, K, starts=[start] if start else ())
    i0 = int(np.abs(grid - m["phis"]).argmin()); prof = np.full(len(grid), np.nan); sol = {}
    for direction in (range(i0, len(grid)), range(i0, -1, -1)):
        prev = (m["S"], m["lam"], m["phis"])
        for i in direction:
            r = fit(e, K, grid[i], starts=[prev] + ([sol[i]] if i in sol else []), light=True)
            if np.isnan(prof[i]) or r["chi2"] < prof[i]: prof[i] = r["chi2"]; sol[i] = (r["S"], r["lam"], grid[i])
            prev = sol[i]
    chi_min = min(m["chi2"], np.nanmin(prof)); ok = grid[prof - chi_min <= 3.84]
    phis = m["phis"] if m["chi2"] <= np.nanmin(prof) else float(grid[np.nanargmin(prof)])
    return dict(phis=phis, chi2=float(chi_min), ci=[float(ok.min()), float(ok.max())] if len(ok) else None,
                at_grid_edge=bool(len(ok) and (ok.min() <= grid[0] or ok.max() >= grid[-1])), grid=grid.tolist(), prof=prof.tolist())


# ---- (2) profile likelihood, all ten counties with Census capacity
B = {x["name"]: x["B"] for x in PANEL["events"]}
prof10 = {n: profile(e, e["K"], start=(B[n]["S_s"], B[n]["lam"], B[n]["phis"])) for n, e in EV.items()}
for n, p in prof10.items():
    print(f"{n:12s} phi* {p['phis']:.3f} (panel {B[n]['phis']:.3f})  chi2 {p['chi2']:.1f}  profile 95% {p['ci']}  edge={p['at_grid_edge']}", flush=True)

# ---- (1) ACS capacity for the Katrina counties
key = os.environ.get("CENSUS_API_KEY") or open(os.path.expanduser("~/.census_api_key")).read().strip()
acs = __import__("pandas").read_csv(OUT.parent / "data" / "acs_county.psv", sep="|", comment="#", dtype={"fips": str}).set_index(["fips", "year"])
C = counties().set_index("fips")
check = {}
for name, fips in [("Orleans", "22071"), ("Jefferson", "22051"), ("Harrison", "28047"), ("Jackson", "28059")]:
    st, co = fips[:2], fips[2:]
    v = json.load(urllib.request.urlopen(f"https://api.census.gov/data/2000/dec/sf1?get=H001001,H005005,H005007&for=county:{co}&in=state:{st}&key={key}"))[1]
    usable_share = (int(v[0]) - int(v[1]) - int(v[2])) / int(v[0])
    hu = [float(C.loc[fips, f"hu{y}"]) for y in range(2000, 2006)]; b = np.polyfit(range(2000, 2006), hu, 1)
    e = EV[name]; t0 = e["t0"]
    yrs = [y for y in range(2006, 2011) if y + 0.5 - t0 <= e["t_obs"][-1] + 0.5]
    cap = np.array([acs.loc[(fips, y), "usable_units"] / (usable_share * np.polyval(b, y + 0.5)) for y in yrs])
    tc = np.array(yrs) + 0.5 - t0
    Kacs = capacity_piecewise(tc, cap)
    check[name] = dict(acs_capacity=cap.tolist(), acs_years=yrs, census_capacity=[float(e["K"](t)) for t in tc],
                       census=prof10[name], acs=profile(e, Kacs, start=(B[name]["S_s"], B[name]["lam"], B[name]["phis"])), usable_share_2000=usable_share)
# Butte: ACS capacity from the ACS panel (Camp Fire)
a = ACSIN[("06007", "Camp Fire 2018")]; e = EV["Butte"]
Kb = (lambda kt, kv: (lambda s: np.interp(s, kt, kv)))(a["Kt"], a["Kv"])
check["Butte"] = dict(acs_capacity=a["cap_rel"].tolist(), acs_years=a["acs_years"], census_capacity=[float(e["K"](t)) for t in a["t_cap"]],
                      census=prof10["Butte"], acs=profile(e, Kb, start=(B["Butte"]["S_s"], B["Butte"]["lam"], B["Butte"]["phis"])))
for n, c in check.items():
    print(f"{n:10s} capacity census {np.round(c['census_capacity'], 3)} acs {np.round(c['acs_capacity'], 3)} | phi* census {c['census']['phis']:.3f} {c['census']['ci']}  acs {c['acs']['phis']:.3f} {c['acs']['ci']}", flush=True)
pc = [c["census"]["phis"] for c in check.values()]; pa = [c["acs"]["phis"] for c in check.values()]
res = dict(kappa=KAPPA, profile_ten=prof10, capacity_check=check,
           agreement=dict(spearman=float(spearmanr(pc, pa)[0]), pearson=float(np.corrcoef(pc, pa)[0, 1]), max_abs_diff=float(np.max(np.abs(np.array(pc) - np.array(pa))))))
print(res["agreement"])
json.dump(res, open(OUT / "capacity_check.json", "w"), indent=1)
