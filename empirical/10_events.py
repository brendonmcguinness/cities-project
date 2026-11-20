"""Panel of disasters: the same struck-place model calibrated separately on every county whose Census housing
estimates record a disaster-related loss.

Events (county = struck place; landfall date):
  Katrina 2005 (29 Aug): Orleans, St Bernard, Plaquemines, Jefferson (LA); Hancock, Harrison, Jackson (MS)
  Rita 2005 (24 Sep): Cameron (LA)
  Harvey 2017 (25 Aug): Aransas (TX)
  Camp Fire 2018 (8 Nov): Butte (CA)
Hurricanes Ike (2008) and Michael (2018) were checked and excluded: Census housing estimates for the struck counties
show no loss, so capacity cannot be measured.

For each event: capacity K(t) from housing units relative to the 5-year pre-event linear trend (exponential recovery
curve, Eq. 2, when >= 4 post-event estimates; otherwise a step at landfall to the first estimate, then linear);
population relative to trend; IRS out-migration totals (all destinations, including suppressed flows).
Because the calibrated host coupling is zero and host areas are large, the network model reduces to the struck place
plus its displaced residents elsewhere (hosts at phi = 1), as in 08_campfire.py.
Fitted per event: S_s (evacuation), kappa (return), lam (settling), to population relative to trend and the IRS
first-year share of two-year excess out-migration. Held out: IRS excess in-migration in the second year (returns).
Also: pooled fit (common rates), leave-one-event-out prediction, and the relation between rates and shock size.
Writes output/events.json and output/events_traj.npz."""
import json
import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from struck import simulate, predictions
from common import HERE, OUT, counties

RAW, DATA = HERE / "raw", HERE / "data"
SIG_POP, SIG_SHARE = 0.005, 0.03          # measurement s.d.: county population ratio, IRS timing share

EVENTS = [  # name, fips, landfall (decimal year), vintage
    ("Orleans", "22071", 2005 + 240 / 365, "2000s"), ("St Bernard", "22087", 2005 + 240 / 365, "2000s"),
    ("Plaquemines", "22075", 2005 + 240 / 365, "2000s"), ("Jefferson", "22051", 2005 + 240 / 365, "2000s"),
    ("Hancock", "28045", 2005 + 240 / 365, "2000s"), ("Harrison", "28047", 2005 + 240 / 365, "2000s"),
    ("Jackson", "28059", 2005 + 240 / 365, "2000s"), ("Cameron", "22023", 2005 + 266 / 365, "2000s"),
    ("Aransas", "48007", 2017 + 236 / 365, "2010s"), ("Butte", "06007", 2018 + 311 / 365, "2010s"),
]
HAZARD = {"Orleans": "Katrina", "St Bernard": "Katrina", "Plaquemines": "Katrina", "Jefferson": "Katrina", "Hancock": "Katrina",
          "Harrison": "Katrina", "Jackson": "Katrina", "Cameron": "Rita", "Aransas": "Harvey", "Butte": "Camp Fire"}

# ------------------------------------------------------------------ data
C00 = counties().set_index("fips")
P10 = pd.read_csv(RAW / "co-est2020-alldata.csv", encoding="latin1", dtype={"STATE": str, "COUNTY": str}).copy()
P10["fips"] = P10.STATE + P10.COUNTY; P10 = P10.set_index("fips")
H10 = pd.concat([pd.read_csv(RAW / f, encoding="latin1", dtype={"STATE": str, "COUNTY": str}) for f in ("HU-EST2020_6.csv", "HU-EST2020_48.csv")])
H10["fips"] = H10.STATE + H10.COUNTY; H10 = H10.set_index("fips")
T00 = pd.read_csv(DATA / "irs_county_totals.psv", sep="|", comment="#", dtype={"fips": str, "year": str}).set_index(["fips", "year"])


def irs10(kind, y):
    d = pd.read_csv(RAW / f"county{kind}flow{y}.csv", encoding="latin1", dtype=str)
    if kind == "out":
        r = d[(d.y2_statefips == "97") & (d.y2_countyfips == "000")]; key = r.y1_statefips + r.y1_countyfips
    else:
        r = d[(d.y1_statefips == "97") & (d.y1_countyfips == "000")]; key = r.y2_statefips + r.y2_countyfips
    return pd.Series(r.n2.astype(float).values, index=key.values)


IRS10 = {(k, y): irs10(k, y) for k in ("out", "in") for y in ("1415", "1516", "1617", "1718", "1819", "1920", "2021")}


def event_data(name, fips, t0, vint):
    ey = int(np.floor(t0))                                   # calendar year of the event (after 1 July)
    base = np.arange(ey - 4, ey + 1)                         # 1 July estimates before the event
    if vint == "2000s":
        pop = {y: float(C00.loc[fips, f"p{y}"]) for y in range(2000, 2010)}; hu = {y: float(C00.loc[fips, f"hu{y}"]) for y in range(2000, 2010)}
        post = list(range(ey + 1, 2010))
        yy = lambda y: f"{str(y)[2:]}{str(y + 1)[2:]}"
        tot = lambda kind, y: float(T00.loc[(fips, yy(y)), f"{kind}_total"])
    else:
        pop = {y: float(P10.loc[fips, f"POPESTIMATE{y}"]) for y in range(2010, 2021)}; hu = {y: float(H10.loc[fips, f"HUESTIMATE{y}"]) for y in range(2010, 2021)}
        post = list(range(ey + 1, 2021))
        yy = lambda y: f"{str(y)[2:]}{str(y + 1)[2:]}"
        tot = lambda kind, y: float(IRS10[(kind, yy(y))].get(fips, np.nan))
    kp, kh = np.polyfit(base, [pop[y] for y in base], 1), np.polyfit(base, [hu[y] for y in base], 1)
    pop_rel = np.array([pop[y] / np.polyval(kp, y) for y in post]); hu_rel = np.array([hu[y] / np.polyval(kh, y) for y in post])
    # IRS filing years: year y covers ~April y to April y+1; event year = the filing year containing landfall
    fy = ey                                                   # e.g. 2005 -> '0506'
    o_base = np.mean([tot("out", fy - 2), tot("out", fy - 1)]); i_base = np.mean([tot("in", fy - 2), tot("in", fy - 1)])
    e1, e2 = tot("out", fy) - o_base, tot("out", fy + 1) - o_base
    ret2 = tot("in", fy + 1) - i_base                        # excess in-migration in the second filing year
    P0 = float(np.polyval(kp, t0))
    share = e1 / (e1 + max(e2, 0.0)) if e1 > 0 else np.nan        # second-year excess below zero counts as zero
    return dict(name=name, fips=fips, hazard=HAZARD[name], t0=t0, post=post, t_obs=np.array(post) + 0.5 - t0,
                pop_rel=pop_rel, hu_rel=hu_rel, share_obs=share, e1=e1, e2=e2, ret2=ret2, P0=P0,
                shock=1 - hu_rel[0], occ1=pop_rel[0] / hu_rel[0], fy_end=(fy + 1 + 0.33 - t0, fy + 2 + 0.33 - t0))


def capacity(ev):
    t, h = ev["t_obs"], ev["hu_rel"]
    if len(t) >= 4:
        f = lambda p: 1 - p[1] - (p[0] - p[1]) * np.exp(-p[2] * t) - h
        p = least_squares(f, [1 - h[0] + 0.05, 1 - h[-1], 0.5], bounds=([0, -0.2, 0.01], [0.99, 0.99, 10])).x
        return (lambda s: 1 - p[1] - (p[0] - p[1]) * np.exp(-p[2] * s)), dict(kind="exponential", d0=p[0], dinf=p[1], rho_s=p[2])
    return (lambda s: np.interp(s, np.r_[0, t], np.r_[h[0], h], right=h[-1])), dict(kind="piecewise", d0=1 - h[0], dinf=1 - h[-1], rho_s=None)


def resid(ev, K, S_s, kappa, lam, phis=1.0):
    pop, share, _ = predictions(ev, K, S_s, kappa, lam, phis)
    r = (pop - ev["pop_rel"]) / SIG_POP
    return r if np.isnan(ev["share_obs"]) else np.r_[r, (share - ev["share_obs"]) / SIG_SHARE]


def vacancy():
    """Pre-disaster housing occupancy of each struck county: Census 2000 (2005 events) or 2010 demographic profiles."""
    import csv, io, zipfile
    out = {}
    z = zipfile.ZipFile(RAW / "2khxx.zip"); rows = {r[3] + r[4]: r for r in csv.reader(io.TextIOWrapper(z.open("2khxx.csv"), encoding="latin1")) if r[1] == "050"}
    for name, fips, t0, vint in EVENTS:
        if vint == "2000s":
            tot, occ, vac, seas = map(float, rows[fips][-11:-7]); renter = float(rows[fips][-3])
        else:
            st = {"48": "tx", "06": "ca"}[fips[:2]]; z = zipfile.ZipFile(RAW / f"{st}2010.dp.zip")
            geo = [l for l in z.read(f"{st}geo2010.dp").decode("latin1").splitlines() if l[8:11] == "050" and l[29:32] == fips[2:]][0]
            rec = [l.split(",") for l in z.read(f"{st}000012010.dp").decode("latin1").splitlines() if l.split(",")[4] == geo[18:25]][0]
            v = [float(x) for x in rec[-18:-7]]          # total, occupied, vacant, for rent, rented, for sale, sold, seasonal, other, rates
            tot, occ, vac, seas = v[0], v[1], v[2], v[7]; renter = float(rec[-5])
        out[name] = dict(vacant=vac / tot, nonseasonal_vacant=(vac - seas) / tot, seasonal=seas / tot, renter_share=renter / occ)
    return out


EV = [event_data(*e) for e in EVENTS]
KS, KINFO = zip(*[capacity(e) for e in EV])
NE = len(EV); NR = [len(e["post"]) + (0 if np.isnan(e["share_obs"]) else 1) for e in EV]
KATRINA_METRO = json.load(open(OUT / "calibration.json"))["B"]


def joint(free_phi, pooled=False):
    """Parameters: log kappa (shared); per event log S_s, log lam (or shared if pooled); phi* per event if free_phi."""
    npe = 1 if pooled else NE
    def unpack(x):
        lk = x[0]; S = np.exp(x[1:1 + npe]); L = np.exp(x[1 + npe:1 + 2 * npe])
        P = x[1 + 2 * npe:1 + 3 * npe] if free_phi else np.ones(NE)
        if pooled: S, L = np.repeat(S, NE), np.repeat(L, NE)
        return np.exp(lk), S, L, P
    def f(x):
        k, S, L, P = unpack(x)
        return np.concatenate([resid(e, K, S[i], k, L[i], P[i]) for i, (e, K) in enumerate(zip(EV, KS))])
    n = 1 + 2 * npe + (NE if free_phi else 0)
    lo = np.r_[-3, np.full(npe, -4), np.full(npe, -5), np.full(NE if free_phi else 0, 0.8)]
    hi = np.r_[6, np.full(npe, 4), np.full(npe, 2), np.full(NE if free_phi else 0, 1.4)]
    sp = None
    if not pooled:                                   # block-sparse Jacobian: event i depends on kappa and its own parameters
        sp = np.zeros((sum(NR), n), dtype=int); r0 = 0
        for i in range(NE):
            sp[r0:r0 + NR[i], 0] = 1; sp[r0:r0 + NR[i], 1 + i] = 1; sp[r0:r0 + NR[i], 1 + NE + i] = 1
            if free_phi: sp[r0:r0 + NR[i], 1 + 2 * NE + i] = 1
            r0 += NR[i]
    best = None
    for lk0, ls0, ll0 in [(1.0, 0.5, -0.7), (2.0, 0.0, -1.5), (0.0, 1.0, 0.0)]:
        x0 = np.r_[lk0, np.full(npe, ls0), np.full(npe, ll0), np.full(NE if free_phi else 0, 1.0)]
        r = least_squares(f, x0, bounds=(lo, hi), jac_sparsity=sp, x_scale=1.0, max_nfev=400)
        if best is None or r.cost < best.cost: best = r
    J = best.jac.toarray() if hasattr(best.jac, "toarray") else best.jac
    cov = np.linalg.pinv(J.T @ J); se = np.sqrt(np.clip(np.diag(cov), 0, None))
    k, S, L, P = unpack(best.x)
    at_b = np.abs(best.active_mask) > 0
    return dict(x=best.x, se=se, kappa=k, S=S, L=L, P=P, chi2=float(2 * best.cost), k=n, n=len(best.fun), active=at_b, npe=npe)


A = joint(free_phi=False)                      # main model, per-event S_s and lam, shared kappa
B = joint(free_phi=True)                       # + per-event tolerated occupancy
Cp = joint(free_phi=False, pooled=True)        # one set of rates for all events
for nm, m in (("A", A), ("B", B), ("C", Cp)):
    print(nm, "chi2", round(m["chi2"], 1), "k", m["k"], "AIC", round(m["chi2"] + 2 * m["k"], 1), "kappa", round(m["kappa"], 2))

rows, traj = [], {}
for i, (e, K, ki) in enumerate(zip(EV, KS, KINFO)):
    def ci(m, j, log=True):
        v, sdev = m["x"][j], m["se"][j]
        return [float(np.exp(v - 1.96 * sdev)), float(np.exp(v + 1.96 * sdev))] if log else [float(v - 1.96 * sdev), float(v + 1.96 * sdev)]
    popA, shA, retA = predictions(e, K, A["S"][i], A["kappa"], A["L"][i])
    popB, shB, retB = predictions(e, K, B["S"][i], B["kappa"], B["L"][i], B["P"][i])
    popC, _, _ = predictions(e, K, Cp["S"][i], Cp["kappa"], Cp["L"][i])
    popK, _, _ = predictions(e, K, KATRINA_METRO["S_s"], KATRINA_METRO["kappa"], KATRINA_METRO["lam"])
    rm = lambda p: float(np.sqrt(np.mean((p - e["pop_rel"]) ** 2)))
    jB = [1 + i, 1 + NE + i, 1 + 2 * NE + i]
    rows.append(dict(name=e["name"], hazard=e["hazard"], fips=e["fips"], pop0=e["P0"], shock=float(e["shock"]), occ1=float(e["occ1"]),
                     capacity=ki, n_obs=len(e["post"]), years=e["post"], pop_obs=e["pop_rel"].tolist(), hu_obs=e["hu_rel"].tolist(),
                     share_obs=None if np.isnan(e["share_obs"]) else float(e["share_obs"]), e1=float(e["e1"]), e2=float(e["e2"]),
                     A=dict(S_s=float(A["S"][i]), lam=float(A["L"][i]), pop=popA.tolist(), share=float(shA), rmse=rm(popA)),
                     B=dict(S_s=float(B["S"][i]), lam=float(B["L"][i]), phis=float(B["P"][i]), S_s_ci=ci(B, jB[0]), lam_ci=ci(B, jB[1]),
                            phis_ci=ci(B, jB[2], log=False), at_bound=bool(B["active"][jB].any()), pop=popB.tolist(), share=float(shB),
                            rmse=rm(popB), ret2=float(retB)),
                     rmse_pooled=rm(popC), rmse_katrina_metro_rates=rm(popK), ret2_obs=float(e["ret2"])))
    tt = np.linspace(0, (2010 - e["t0"]) if e["t0"] < 2010 else e["t_obs"][-1] + 0.3, 300)
    sB = simulate(K, B["S"][i], B["kappa"], B["L"][i], tt[-1] + 0.01, B["P"][i]); sA = simulate(K, A["S"][i], A["kappa"], A["L"][i], tt[-1] + 0.01)
    traj[e["name"]] = np.vstack([tt + e["t0"], sB.sol(tt)[0], sA.sol(tt)[0], K(tt)])
    r = rows[-1]
    print(f"{e['name']:11s} shock={r['shock']:.3f} occ1={r['occ1']:.3f} | A: S={r['A']['S_s']:.2f} lam={r['A']['lam']:.2f} rmse={r['A']['rmse']:.3f}"
          f" | B: S={r['B']['S_s']:.2f} lam={r['B']['lam']:.2f} phi*={r['B']['phis']:.3f} {np.round(r['B']['phis_ci'],3)} rmse={r['B']['rmse']:.4f} bound={r['B']['at_bound']}")

# leave-one-event-out: pooled rates (main model) from the other events
lofo = []
for i, (e, K) in enumerate(zip(EV, KS)):
    keep = [j for j in range(NE) if j != i]
    def f(x):
        return np.concatenate([resid(EV[j], KS[j], np.exp(x[1]), np.exp(x[0]), np.exp(x[2])) for j in keep])
    x = least_squares(f, [1.0, 0.5, -0.7], bounds=([-3, -4, -5], [6, 4, 2])).x
    pop, _, _ = predictions(e, K, np.exp(x[1]), np.exp(x[0]), np.exp(x[2]))
    lofo.append(dict(name=e["name"], rmse=float(np.sqrt(np.mean((pop - e["pop_rel"]) ** 2)))))

d = pd.DataFrame([dict(name=r["name"], shock=r["shock"], occ1=r["occ1"], S=r["B"]["S_s"], lam=r["B"]["lam"], phis=r["B"]["phis"],
                       bound=r["B"]["at_bound"]) for r in rows])
k = ~d.bound
rel = dict(phis_vs_shock_r=float(np.corrcoef(d.shock, d.phis)[0, 1]),
           logS_vs_logshock=dict(r=float(np.corrcoef(np.log(d.shock[k]), np.log(d.S[k]))[0, 1]), n=int(k.sum())),
           S_median_interior=float(np.median(d.S[k])), S_range_interior=[float(d.S[k].min()), float(d.S[k].max())],
           phis_range=[float(d.phis.min()), float(d.phis.max())])
ident = [r["name"] for r in rows if r["B"]["S_s_ci"][1] / max(r["B"]["S_s_ci"][0], 1e-12) < 10]   # S_s identified (95% CI within 10x)
di = d[d.name.isin(ident)]
rel["S_identified"] = dict(events=ident, S=[float(v) for v in di.S], logS_vs_logshock_r=float(np.corrcoef(np.log(di.shock), np.log(di.S))[0, 1]),
                           slope=float(np.polyfit(np.log(di.shock), np.log(di.S), 1)[0]))
VAC = vacancy()
for cov in ("vacant", "nonseasonal_vacant", "renter_share"):
    rel[f"phis_vs_{cov}_r"] = float(np.corrcoef(d.phis, [VAC[n][cov] for n in d.name])[0, 1])
x = np.array([[r["ret2_obs"] / r["pop0"], r["B"]["ret2"] / r["pop0"]] for r in rows if not np.isnan(r["share_obs"] if r["share_obs"] is not None else np.nan)])
rel["ret2_corr"] = float(np.corrcoef(x.T)[0, 1]); rel["ret2_n"] = int(len(x))
from scipy.stats import spearmanr
rel["ret2_spearman"] = float(spearmanr(x[:, 0], x[:, 1])[0])
res = dict(events=rows,
           models={nm: dict(chi2=m["chi2"], k=int(m["k"]), n=int(m["n"]), aic=m["chi2"] + 2 * m["k"], kappa=float(m["kappa"]))
                   for nm, m in (("A_housing_only", A), ("B_tolerated_occupancy", B), ("C_pooled", Cp))},
           pooled_rates=dict(S_s=float(Cp["S"][0]), lam=float(Cp["L"][0]), kappa=float(Cp["kappa"])),
           leave_one_out=lofo, relations=rel,
           katrina_metro_rates=dict(S_s=KATRINA_METRO["S_s"], kappa=KATRINA_METRO["kappa"], lam=KATRINA_METRO["lam"]),
           excluded=dict(Ike_2008_Galveston="no housing loss in Census estimates", Michael_2018_Bay_Gulf="no housing loss in Census estimates"),
           notes=dict(Aransas_IRS="IRS excess out-migration in the event year (about 400) is far below the Census population loss (about 2,600), consistent with disaster filing postponements; the timing share (1.0, second-year excess negative) is kept but carries little information"),
           settings=dict(sig_pop=SIG_POP, sig_share=SIG_SHARE, phis_bounds=[0.8, 1.4]))
json.dump(res, open(OUT / "events.json", "w"), indent=1)
np.savez(OUT / "events_traj.npz", **traj)
print("LOFO", [(l["name"], round(l["rmse"], 3)) for l in lofo]); print(json.dumps(rel))

# event inputs for the held-out test (14_holdout.py): capacity path on a fine grid
import pickle
TG = np.linspace(0, 6, 601)
pickle.dump([dict(e, disaster=e["hazard"], src="census", loss=float(e["shock"]), Kt=TG, Kv=np.array([K(s) for s in TG])) for e, K in zip(EV, KS)],
            open(OUT / "events_inputs.pkl", "wb"))
