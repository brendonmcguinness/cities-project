"""Held-out disasters predicted from pre-disaster hazard geography (Plan B, part 3).

Panel: the counties of 14_holdout.py with a hazard-geography measure (17_hazard_geography.py; fires and the tornado
have none): 17 counties, 10 disasters.
Models, all with shared return and settling rates and per-county evacuation rates:
  G  phi*_j ~ N(a + b x_j, tau^2), x = share of pre-event housing outside the mapped hazard zone (primary measure)
  H  phi*_j ~ N(mu, tau^2)              (pooled, as in 14_holdout.py)
  A  phi* = 1
For each disaster in turn all three are refitted to the other disasters, and every county it struck is predicted from
its measured capacity and, for G, its hazard geography: phi* = a + b x (G), mu (H) or 1 (A); evacuation rate from the
training relation with housing loss. The elevation measure (share of housing above 3 m) is run as a secondary
variant on the coastal counties. Writes output/holdout_geo.json."""
import json, pickle
import numpy as np
from scipy.optimize import least_squares
from common import HERE, OUT
from struck import predictions

SIG_POP, SIG_SHARE, TAU = 0.005, 0.03, 0.10
geo = {g["fips"] + "|" + g["disaster"]: g for g in json.load(open(HERE / "data" / "hazard_geography.json"))["counties"]}
ev10 = pickle.load(open(OUT / "events_inputs.pkl", "rb")); ev13 = pickle.load(open(OUT / "acs_inputs.pkl", "rb"))
group = lambda d: d.rsplit(" ", 1)[0] if d[-4:].isdigit() else d
EV = []
for e in ev10 + [e for e in ev13 if e["loss"] >= 0.04]:
    k = e["fips"] + "|" + (e["hazard"] if e["src"] == "census" else e["disaster"])
    if k not in geo:
        continue
    e = dict(e); e["g"] = group(e["disaster"]); e["x"] = geo[k]["outside_primary"]; e["x3"] = geo[k]["above3m"]
    e["K"] = (lambda kt, kv: (lambda s: np.interp(s, kt, kv)))(e["Kt"], e["Kv"])
    if e.get("fy_end") is None or np.isnan(e["share_obs"]): e["share_obs"] = np.nan
    EV.append(e)
print(len(EV), "counties,", len(set(e["g"] for e in EV)), "disasters")


def resid(e, S, kap, lam, phis):
    pop, share, _ = predictions(e, e["K"], S, kap, lam, phis)
    r = (pop - e["pop_rel"]) / SIG_POP
    return r if np.isnan(e["share_obs"]) else np.r_[r, (share - e["share_obs"]) / SIG_SHARE]


def fit(evs, kind, xkey="x"):
    """kind: 'A' (phi*=1), 'H' (pooled mean), 'G' (mean linear in x)."""
    n = len(evs); nr = [len(resid(e, 1.0, 5.0, 0.5, 1.0)) for e in evs]; x = np.array([e[xkey] for e in evs])
    nh = {"A": 0, "H": 1, "G": 2}[kind]; npar = 2 + n + (n + nh if kind != "A" else 0)
    def unpack(p):
        kap, lam, S = np.exp(p[0]), np.exp(p[1]), np.exp(p[2:2 + n])
        if kind == "A": return kap, lam, S, np.ones(n), None
        P = p[2 + n:2 + 2 * n]; h = p[2 + 2 * n:]
        return kap, lam, S, P, (np.full(n, h[0]) if kind == "H" else h[0] + h[1] * x)
    def f(p):
        kap, lam, S, P, m = unpack(p)
        r = [resid(e, S[j], kap, lam, P[j]) for j, e in enumerate(evs)]
        return np.concatenate(r + ([(P - m) / TAU] if kind != "A" else []))
    sp = np.zeros((sum(nr) + (n if kind != "A" else 0), npar), dtype=int); r0 = 0
    for j in range(n):
        sp[r0:r0 + nr[j], :2] = 1; sp[r0:r0 + nr[j], 2 + j] = 1
        if kind != "A": sp[r0:r0 + nr[j], 2 + n + j] = 1
        r0 += nr[j]
    if kind != "A":
        for j in range(n): sp[r0 + j, 2 + n + j] = 1; sp[r0 + j, 2 + 2 * n:] = 1
    lo = np.r_[-3, -5, np.full(n, -4)]; hi = np.r_[6, 2, np.full(n, 4)]
    if kind != "A":
        lo = np.r_[lo, np.full(n, 0.7), [0.7, -1.0][:nh]]; hi = np.r_[hi, np.full(n, 1.5), [1.4, 1.0][:nh]]
    best = None
    for k0, l0, s0 in [(2.0, -0.7, 0.5), (1.0, -1.5, 1.0)]:
        p0 = np.r_[k0, l0, np.full(n, s0)]
        if kind != "A": p0 = np.r_[p0, np.ones(n), [1.0, 0.0][:nh]]
        r = least_squares(f, p0, bounds=(lo, hi), jac_sparsity=sp, max_nfev=300)
        if best is None or r.cost < best.cost: best = r
    kap, lam, S, P, m = unpack(best.x)
    b, a = np.polyfit(np.log(np.array([max(e["loss"], 0.01) for e in evs])), np.log(S), 1)
    h = best.x[2 + 2 * n:] if kind != "A" else []
    return dict(kap=float(kap), lam=float(lam), S=S, P=P, h=[float(v) for v in h], a=float(a), b=float(b), chi2=float(2 * best.cost))


def predict(m, kind, e, xkey="x"):
    phis = 1.0 if kind == "A" else (m["h"][0] if kind == "H" else m["h"][0] + m["h"][1] * e[xkey])
    S = np.exp(m["a"] + m["b"] * np.log(max(e["loss"], 0.01)))
    return predictions(e, e["K"], S, m["kap"], m["lam"], phis)[0], phis


rmse = lambda p, o: float(np.sqrt(np.mean((np.asarray(p) - o) ** 2)))
out = {}
for label, xkey, evs in (("primary", "x", EV), ("elevation", "x3", [e for e in EV if e["x3"] is not None])):
    full = fit(evs, "G", xkey)
    print(label, "full-panel G: phi* =", np.round(full["h"], 3), "(intercept, slope)")
    rows = []
    for g in sorted(set(e["g"] for e in evs)):
        tr = [e for e in evs if e["g"] != g]
        M = {k: fit(tr, k, xkey) for k in ("G", "H", "A")}
        for e in (e for e in evs if e["g"] == g):
            o = e["pop_rel"]; pr = {k: predict(M[k], k, e, xkey) for k in M}
            rows.append(dict(g=g, name=e["name"], x=e[xkey], loss=float(e["loss"]), obs=o.tolist(), t=e["t_obs"].tolist(),
                             phis={k: float(pr[k][1]) for k in pr}, pred={k: pr[k][0].tolist() for k in pr},
                             rmse={**{k: rmse(pr[k][0], o) for k in pr}, "K": rmse(e["K"](e["t_obs"]), o), "none": rmse(np.ones_like(o), o)},
                             slope_train=M["G"]["h"][1]))
            r = rows[-1]["rmse"]
            print(f"  {g:16s} {e['name']:20s} x={e[xkey]:.2f} phi*G={pr['G'][1]:.3f}  G {r['G']:.3f}  H {r['H']:.3f}  A {r['A']:.3f}  N=K {r['K']:.3f}", flush=True)
    summ = {k: dict(median=float(np.median([r["rmse"][k] for r in rows])), mean=float(np.mean([r["rmse"][k] for r in rows])))
            for k in ("G", "H", "A", "K", "none")}
    summ["G_beats_H"] = int(sum(r["rmse"]["G"] < r["rmse"]["H"] for r in rows)); summ["n"] = len(rows)
    summ["slope_train_range"] = [min(r["slope_train"] for r in rows), max(r["slope_train"] for r in rows)]
    print(label, json.dumps(summ))
    out[label] = dict(full=dict(intercept=full["h"][0], slope=full["h"][1], kappa=full["kap"], lam=full["lam"]), rows=rows, summary=summ)
json.dump(out, open(OUT / "holdout_geo.json", "w"), indent=1)
