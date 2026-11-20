"""Expanded disaster panel with capacity measured from the American Community Survey (ACS).

Events are fixed in advance: major housing-destroying US disasters 2008-2024 striking counties with ACS 1-year
coverage (>= 65,000 residents). Every listed county-event is analysed whatever its measured loss (no selection on
the outcome). Puerto Rico is excluded (separate survey context; IRS coverage).

Per event:
  capacity   ACS usable housing units (units - 'other vacant' - seasonal) relative to the linear trend of the
             pre-event ACS years (>= 3 of the 6 before the event; the ACS year containing the event is dropped);
             piecewise-linear path through the post-event estimates (ACS year y taken at mid-year y + 0.5).
  population Census July estimates relative to the linear trend of the 5 estimates before the event
             (intercensal 2000-09, Vintage 2020 for 2010-19, Vintage 2025 for 2020-25 spliced to Vintage 2020 at 2020).
  timing     IRS share of two-year excess out-migration in the event filing year, when both years are released.
  doubling   ACS average household size relative to its pre-event mean (held out; never used in fitting).
Model (struck.py): shared return and settling rates; per-event evacuation rate and tolerated occupancy phi*.
Writes output/acs_panel.json."""
import json
import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from common import HERE, OUT, counties
from struck import predictions, capacity_piecewise

RAW, DATA = HERE / "raw", HERE / "data"
SIG_POP, SIG_SHARE = 0.005, 0.03
D = lambda y, m, d: y + (pd.Timestamp(y, m, d).dayofyear - 1) / 365.25
EVENTS = [  # (disaster, county, fips, date)
    ("Ike 2008", "Galveston TX", "48167", D(2008, 9, 13)), ("Ike 2008", "Harris TX", "48201", D(2008, 9, 13)),
    ("Ike 2008", "Jefferson TX", "48245", D(2008, 9, 13)), ("Ike 2008", "Orange TX", "48361", D(2008, 9, 13)),
    ("Tuscaloosa tornado 2011", "Tuscaloosa AL", "01125", D(2011, 4, 27)), ("Joplin tornado 2011", "Jasper MO", "29097", D(2011, 5, 22)),
    ("Sandy 2012", "Ocean NJ", "34029", D(2012, 10, 29)), ("Sandy 2012", "Monmouth NJ", "34025", D(2012, 10, 29)),
    ("Sandy 2012", "Atlantic NJ", "34001", D(2012, 10, 29)), ("Sandy 2012", "Richmond NY", "36085", D(2012, 10, 29)),
    ("Sandy 2012", "Nassau NY", "36059", D(2012, 10, 29)), ("Moore tornado 2013", "Cleveland OK", "40027", D(2013, 5, 20)),
    ("Louisiana floods 2016", "Livingston LA", "22063", D(2016, 8, 12)), ("Louisiana floods 2016", "East Baton Rouge LA", "22033", D(2016, 8, 12)),
    ("Louisiana floods 2016", "Ascension LA", "22005", D(2016, 8, 12)), ("Matthew 2016", "Robeson NC", "37155", D(2016, 10, 8)),
    ("Harvey 2017", "Harris TX", "48201", D(2017, 8, 25)), ("Harvey 2017", "Jefferson TX", "48245", D(2017, 8, 25)),
    ("Harvey 2017", "Orange TX", "48361", D(2017, 8, 25)), ("Harvey 2017", "Nueces TX", "48355", D(2017, 8, 25)),
    ("Harvey 2017", "Fort Bend TX", "48157", D(2017, 8, 25)), ("Harvey 2017", "Galveston TX", "48167", D(2017, 8, 25)),
    ("Irma 2017", "Monroe FL", "12087", D(2017, 9, 10)), ("Irma 2017", "Collier FL", "12021", D(2017, 9, 10)),
    ("Irma 2017", "Lee FL", "12071", D(2017, 9, 10)), ("Tubbs fire 2017", "Sonoma CA", "06097", D(2017, 10, 8)),
    ("Tubbs fire 2017", "Napa CA", "06055", D(2017, 10, 8)), ("Florence 2018", "New Hanover NC", "37129", D(2018, 9, 14)),
    ("Florence 2018", "Onslow NC", "37133", D(2018, 9, 14)), ("Florence 2018", "Craven NC", "37049", D(2018, 9, 14)),
    ("Michael 2018", "Bay FL", "12005", D(2018, 10, 10)), ("Camp Fire 2018", "Butte CA", "06007", D(2018, 11, 8)),
    ("Carr fire 2018", "Shasta CA", "06089", D(2018, 7, 23)), ("Laura 2020", "Calcasieu LA", "22019", D(2020, 8, 27)),
    ("Ida 2021", "Terrebonne LA", "22109", D(2021, 8, 29)), ("Ida 2021", "Lafourche LA", "22057", D(2021, 8, 29)),
    ("Ida 2021", "Jefferson LA", "22051", D(2021, 8, 29)), ("Ida 2021", "Orleans LA", "22071", D(2021, 8, 29)), ("Ian 2022", "Lee FL", "12071", D(2022, 9, 28)),
    ("Ian 2022", "Charlotte FL", "12015", D(2022, 9, 28)), ("Ian 2022", "Collier FL", "12021", D(2022, 9, 28)),
    ("Lahaina fire 2023", "Maui HI", "15009", D(2023, 8, 8)), ("Helene 2024", "Buncombe NC", "37021", D(2024, 9, 26)),
]
WINDOW = 4.6   # years of post-event data used (stops earlier if the same county is struck again)

# ------------------------------------------------------------------ data
ACS = pd.read_csv(DATA / "acs_county.psv", sep="|", comment="#", dtype={"fips": str}).set_index(["fips", "year"]).sort_index()
C00 = counties().set_index("fips")
P20 = pd.read_csv(RAW / "co-est2020-alldata.csv", encoding="latin1", dtype={"STATE": str, "COUNTY": str})
P20 = P20[P20.SUMLEV == 50].assign(fips=lambda d: d.STATE + d.COUNTY).set_index("fips")
P25 = pd.read_csv(RAW / "co-est2025-alldata.csv", encoding="latin1", dtype={"STATE": str, "COUNTY": str})
P25 = P25[P25.SUMLEV == 50].assign(fips=lambda d: d.STATE + d.COUNTY).set_index("fips")


def pop_series(fips):
    s = {}
    if fips in C00.index:
        s.update({y: float(C00.loc[fips, f"p{y}"]) for y in range(2000, 2010)})
    if fips in P20.index:
        s.update({y: float(P20.loc[fips, f"POPESTIMATE{y}"]) for y in range(2010, 2020)})
        if fips in P25.index:                     # splice Vintage 2025 to Vintage 2020 at July 2020
            f = float(P20.loc[fips, "POPESTIMATE2020"]) / float(P25.loc[fips, "POPESTIMATE2020"])
            s.update({y: f * float(P25.loc[fips, f"POPESTIMATE{y}"]) for y in range(2020, 2026)})
    return s


def irs_total(kind, fips, fy):
    yy = f"{str(fy)[2:]}{str(fy + 1)[2:]}"
    f = RAW / f"county{kind}flow{yy}.csv"
    if fy <= 2007:                                # 2003-04 .. 2007-08 from the .dat files (00_build_data.py)
        t = pd.read_csv(DATA / "irs_county_totals.psv", sep="|", comment="#", dtype={"fips": str, "year": str}).set_index(["fips", "year"])
        return float(t.loc[(fips, yy), f"{kind}_total"]) if (fips, yy) in t.index else np.nan
    if not f.exists():
        return np.nan
    d = pd.read_csv(f, encoding="latin1", dtype=str)
    d.columns = ["y1s", "y1c", "y2s", "y2c"] + list(d.columns[4:])
    n = d.columns[7]                               # exemptions (n2 / Exmpt_Num)
    if kind == "out":
        r = d[(d.y2s.str.zfill(2) == "97") & (d.y2c.str.zfill(3) == "000") & (d.y1s.str.zfill(2) + d.y1c.str.zfill(3) == fips)]
    else:
        r = d[(d.y1s.str.zfill(2) == "97") & (d.y1c.str.zfill(3) == "000") & (d.y2s.str.zfill(2) + d.y2c.str.zfill(3) == fips)]
    return float(pd.to_numeric(r[n], errors="coerce").iloc[0]) if len(r) else np.nan


def event_data(dis, name, fips, t0):
    later = [e[3] for e in EVENTS if e[2] == fips and e[3] > t0]
    t_end = min([t0 + WINDOW] + [t - 0.01 for t in later])
    # population
    ps = pop_series(fips)
    base = sorted([y for y in ps if y + 0.5 < t0])[-5:]
    post = [y for y in sorted(ps) if t0 < y + 0.5 <= t_end]
    kp = np.polyfit(base, [ps[y] for y in base], 1)
    pop_rel = np.array([ps[y] / np.polyval(kp, y) for y in post])
    # ACS capacity and household size (ACS year containing the event dropped)
    ay = [y for (f, y) in ACS.index if f == fips and not np.isnan(ACS.loc[(f, y), "usable_units"])]
    a_pre = [y for y in ay if y + 1 <= t0 and y >= int(t0) - 6]
    a_post = [y for y in ay if y >= int(np.floor(t0)) + 1 and y + 0.5 <= t_end + 0.5]
    if len(a_pre) < 3 or not a_post or len(post) == 0:
        return None
    ku = np.polyfit(a_pre, [ACS.loc[(fips, y), "usable_units"] for y in a_pre], 1)
    cap_rel = np.array([ACS.loc[(fips, y), "usable_units"] / np.polyval(ku, y) for y in a_post])
    hh0 = np.mean([ACS.loc[(fips, y), "hh_size"] for y in a_pre])
    hh_rel = np.array([ACS.loc[(fips, y), "hh_size"] / hh0 for y in a_post])
    oth0 = np.mean([ACS.loc[(fips, y), "v_other"] / ACS.loc[(fips, y), "hu"] for y in a_pre])
    oth1 = float(ACS.loc[(fips, a_post[0]), "v_other"] / ACS.loc[(fips, a_post[0]), "hu"])
    # IRS timing share
    fy = int(np.floor(t0 - 0.33))
    tot = {k: irs_total("out", fips, k) for k in (fy - 2, fy - 1, fy, fy + 1)}
    share, fy_end = np.nan, None
    if all(np.isfinite(list(tot.values()))):
        o_base = (tot[fy - 2] + tot[fy - 1]) / 2; e1, e2 = tot[fy] - o_base, tot[fy + 1] - o_base
        if e1 > 0:
            share = e1 / (e1 + max(e2, 0.0)); fy_end = (fy + 1 + 0.33 - t0, fy + 2 + 0.33 - t0)
    return dict(disaster=dis, name=name, fips=fips, t0=t0, years=post, t_obs=np.array(post) + 0.5 - t0, pop_rel=pop_rel,
                acs_years=a_post, t_cap=np.array(a_post) + 0.5 - t0, cap_rel=cap_rel, hh_rel=hh_rel, loss=float(1 - cap_rel[0]),
                other_jump=oth1 - oth0, share_obs=share, fy_end=fy_end, P0=float(np.polyval(kp, t0)))


EV, skipped = [], []
for e in EVENTS:
    d = event_data(*e)
    (EV if d else skipped).append(d if d else f"{e[0]} / {e[1]}")
KS = [capacity_piecewise(e["t_cap"], e["cap_rel"]) for e in EV]
NE = len(EV); print(NE, "events;", "skipped (insufficient ACS years):", skipped)


def resid(ev, K, S, kap, lam, phis):
    pop, share, _ = predictions(ev, K, S, kap, lam, phis)
    r = (pop - ev["pop_rel"]) / SIG_POP
    return r if (np.isnan(ev["share_obs"]) or ev["fy_end"] is None) else np.r_[r, (share - ev["share_obs"]) / SIG_SHARE]


NR = [len(resid(e, k, 1.0, 5.0, 0.5, 1.0)) for e, k in zip(EV, KS)]


def joint(free_phi, events=None):
    idx = list(range(NE)) if events is None else events
    n_e = len(idx)
    def unpack(x):
        kap, lam = np.exp(x[0]), np.exp(x[1]); S = np.exp(x[2:2 + n_e]); P = x[2 + n_e:2 + 2 * n_e] if free_phi else np.ones(n_e)
        return kap, lam, S, P
    def f(x):
        kap, lam, S, P = unpack(x)
        return np.concatenate([resid(EV[i], KS[i], S[j], kap, lam, P[j]) for j, i in enumerate(idx)])
    npar = 2 + n_e + (n_e if free_phi else 0)
    sp = np.zeros((sum(NR[i] for i in idx), npar), dtype=int); r0 = 0
    for j, i in enumerate(idx):
        sp[r0:r0 + NR[i], :2] = 1; sp[r0:r0 + NR[i], 2 + j] = 1
        if free_phi: sp[r0:r0 + NR[i], 2 + n_e + j] = 1
        r0 += NR[i]
    lo = np.r_[-3, -5, np.full(n_e, -4), np.full(n_e if free_phi else 0, 0.7)]
    hi = np.r_[6, 2, np.full(n_e, 4), np.full(n_e if free_phi else 0, 1.5)]
    best = None
    for x0k in [(2.0, -0.7, 0.5), (1.0, -1.5, 1.0)]:
        x0 = np.r_[x0k[0], x0k[1], np.full(n_e, x0k[2]), np.full(n_e if free_phi else 0, 1.0)]
        r = least_squares(f, x0, bounds=(lo, hi), jac_sparsity=sp, max_nfev=300)
        if best is None or r.cost < best.cost: best = r
    J = best.jac.toarray() if hasattr(best.jac, "toarray") else best.jac
    se = np.sqrt(np.clip(np.diag(np.linalg.pinv(J.T @ J)), 0, None))
    kap, lam, S, P = unpack(best.x)
    return dict(x=best.x, se=se, kap=float(kap), lam=float(lam), S=S, P=P, chi2=float(2 * best.cost), k=npar, n=len(best.fun), n_e=n_e)


A = joint(False); B = joint(True)
print("A chi2", round(A["chi2"], 1), "k", A["k"], "| B chi2", round(B["chi2"], 1), "k", B["k"], "n", B["n"])
rows = []
for j, (e, K) in enumerate(zip(EV, KS)):
    popA, shA, _ = predictions(e, K, A["S"][j], A["kap"], A["lam"], 1.0)
    popB, shB, _ = predictions(e, K, B["S"][j], B["kap"], B["lam"], B["P"][j])
    pse = B["se"][2 + NE + j]
    rm = lambda p: float(np.sqrt(np.mean((p - e["pop_rel"]) ** 2)))
    rows.append(dict(disaster=e["disaster"], name=e["name"], fips=e["fips"], t0=e["t0"], years=e["years"], pop_obs=e["pop_rel"].tolist(),
                     acs_years=e["acs_years"], cap_obs=e["cap_rel"].tolist(), hh_rel=e["hh_rel"].tolist(), loss=e["loss"],
                     other_jump=e["other_jump"], share_obs=None if np.isnan(e["share_obs"]) else float(e["share_obs"]),
                     occ1=float(e["pop_rel"][0] / np.interp(e["t_obs"][0], np.r_[0, e["t_cap"]], np.r_[e["cap_rel"][0], e["cap_rel"]])),
                     A=dict(S_s=float(A["S"][j]), pop=popA.tolist(), rmse=rm(popA)),
                     B=dict(S_s=float(B["S"][j]), phis=float(B["P"][j]), phis_se=float(pse), pop=popB.tolist(), rmse=rm(popB))))
res = dict(events=rows, skipped=skipped, models=dict(A=dict(chi2=A["chi2"], k=A["k"], n=A["n"], kappa=A["kap"], lam=A["lam"]),
                                                     B=dict(chi2=B["chi2"], k=B["k"], n=B["n"], kappa=B["kap"], lam=B["lam"])),
           settings=dict(sig_pop=SIG_POP, sig_share=SIG_SHARE, window=WINDOW))
json.dump(res, open(OUT / "acs_panel.json", "w"), indent=1)
d = pd.DataFrame([dict(ev=r["disaster"][:14], county=r["name"], loss=r["loss"], oth=r["other_jump"], occ1=r["occ1"], hh1=r["hh_rel"][0],
                       phis=r["B"]["phis"], se=r["B"]["phis_se"], rA=r["A"]["rmse"], rB=r["B"]["rmse"]) for r in rows])
pd.set_option("display.width", 200); print(d.round(3).to_string(index=False))

# event inputs for the held-out test (14_holdout.py): capacity path on a fine grid
import pickle
TG = np.linspace(0, 6, 601)
pickle.dump([dict(e, src="acs", Kt=TG, Kv=np.array([K(s) for s in TG])) for e, K in zip(EV, KS)], open(OUT / "acs_inputs.pkl", "wb"))
