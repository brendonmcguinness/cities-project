"""Leave-one-disaster-out prediction on the combined panel.

Panel: the ten Census-capacity events (10_events.py) plus the ACS-capacity events (13_acs_panel.py) whose measured
first-year loss of usable housing is at least 4% (below that the ACS sampling error, about 1-2% of county units,
leaves tolerated occupancy unidentified). The Camp Fire appears in both; the Census version is kept.

Partial pooling (model H): shared return and settling rates, per-event evacuation rate, per-event tolerated
occupancy phi*_j with a normal prior N(mu, tau^2), mu estimated, tau fixed at TAU.
For each disaster (all counties struck by it held out together):
  fit H and the phi*=1 model A on the other disasters;
  predict the held-out population path from its measured capacity alone, with
    evacuation rate from the training relation log S = a + b log(loss) and phi* = mu (H) or 1 (A).
Baselines: population tracks capacity at once (N = K), and no change from the pre-event trend (N = 1).
Writes output/holdout.json."""
import json, pickle, sys
import numpy as np
from scipy.optimize import least_squares
from common import OUT
from struck import predictions

import os
SIG_POP, SIG_SHARE = 0.005, 0.03
TAU = float(os.environ.get("HOLDOUT_TAU", 0.10))          # sd of the prior on phi* (sensitivity: 0.05, 0.2)
MIN_LOSS = float(os.environ.get("HOLDOUT_MIN_LOSS", 0.04))  # ACS counties enter if first-year loss >= this
TAG = os.environ.get("HOLDOUT_TAG", "")                     # output suffix for sensitivity runs
ev10 = pickle.load(open(OUT / "events_inputs.pkl", "rb")); ev13 = pickle.load(open(OUT / "acs_inputs.pkl", "rb"))
group = lambda d: d.rsplit(" ", 1)[0] if d[-4:].isdigit() else d
EV = ev10 + [e for e in ev13 if e["loss"] >= MIN_LOSS and not (group(e["disaster"]) == "Camp Fire" and e["fips"] == "06007")]
if os.environ.get("HOLDOUT_ACSCAP"):   # sensitivity: survey-measured capacity for the Census-panel counties the ACS covers
    from struck import capacity_piecewise
    CC = json.load(open(OUT / "capacity_check.json"))["capacity_check"]
    for e in EV:
        if e["src"] == "census" and e["name"] in CC:
            c = CC[e["name"]]; tc = np.array(c["acs_years"]) + 0.5 - e["t0"]; Kf = capacity_piecewise(tc, np.array(c["acs_capacity"]))
            e["Kv"] = np.array([Kf(t) for t in e["Kt"]]); e["loss"] = float(1 - c["acs_capacity"][0]); e["capacity"] = "acs"
for e in EV:
    e["g"] = group(e["disaster"]); e["K"] = (lambda kt, kv: (lambda s: np.interp(s, kt, kv)))(e["Kt"], e["Kv"])
    if e.get("fy_end") is None or np.isnan(e["share_obs"]): e["share_obs"] = np.nan
print(len(EV), "events:", [(e["g"], e["name"]) for e in EV])


def resid(e, S, kap, lam, phis):
    pop, share, _ = predictions(e, e["K"], S, kap, lam, phis)
    r = (pop - e["pop_rel"]) / SIG_POP
    return r if np.isnan(e["share_obs"]) else np.r_[r, (share - e["share_obs"]) / SIG_SHARE]


def fit(evs, free_phi):
    n = len(evs); nr = [len(resid(e, 1.0, 5.0, 0.5, 1.0)) for e in evs]
    def unpack(x):
        return np.exp(x[0]), np.exp(x[1]), np.exp(x[2:2 + n]), (x[2 + n:2 + 2 * n] if free_phi else np.ones(n)), (x[-1] if free_phi else 1.0)
    def f(x):
        kap, lam, S, P, mu = unpack(x)
        r = [resid(e, S[j], kap, lam, P[j]) for j, e in enumerate(evs)]
        return np.concatenate(r + ([(P - mu) / TAU] if free_phi else []))
    npar = 2 + n + (n + 1 if free_phi else 0)
    sp = np.zeros((sum(nr) + (n if free_phi else 0), npar), dtype=int); r0 = 0
    for j in range(n):
        sp[r0:r0 + nr[j], :2] = 1; sp[r0:r0 + nr[j], 2 + j] = 1
        if free_phi: sp[r0:r0 + nr[j], 2 + n + j] = 1
        r0 += nr[j]
    if free_phi:
        for j in range(n): sp[r0 + j, 2 + n + j] = 1; sp[r0 + j, -1] = 1
    lo = np.r_[-3, -5, np.full(n, -4), np.full(n, 0.7), 0.8] if free_phi else np.r_[-3, -5, np.full(n, -4)]
    hi = np.r_[6, 2, np.full(n, 4), np.full(n, 1.5), 1.3] if free_phi else np.r_[6, 2, np.full(n, 4)]
    best = None
    for k0, l0, s0 in [(2.0, -0.7, 0.5), (1.0, -1.5, 1.0)]:
        x0 = np.r_[k0, l0, np.full(n, s0)] if not free_phi else np.r_[k0, l0, np.full(n, s0), np.ones(n), 1.0]
        r = least_squares(f, x0, bounds=(lo, hi), jac_sparsity=sp, max_nfev=300)
        if best is None or r.cost < best.cost: best = r
    kap, lam, S, P, mu = unpack(best.x)
    loss = np.array([max(e["loss"], 0.01) for e in evs])
    b, a = np.polyfit(np.log(loss), np.log(S), 1)
    sdS = float(np.std(np.log(S) - (a + b * np.log(loss)), ddof=2))
    return dict(kap=float(kap), lam=float(lam), S=S, P=P, mu=float(mu), a=float(a), b=float(b), sdS=sdS, chi2=float(2 * best.cost))


rmse = lambda p, o: float(np.sqrt(np.mean((np.asarray(p) - o) ** 2)))
H_all, A_all = fit(EV, True), fit(EV, False)
print("all: H mu", round(H_all["mu"], 3), "kappa", round(H_all["kap"], 2), "lam", round(H_all["lam"], 3), "| A kappa", round(A_all["kap"], 2))
groups = sorted(set(e["g"] for e in EV)); rows = []
for g in groups:
    tr = [e for e in EV if e["g"] != g]; te = [e for e in EV if e["g"] == g]
    H, A = fit(tr, True), fit(tr, False)
    for e in te:
        l = np.log(max(e["loss"], 0.01))
        pH = predictions(e, e["K"], np.exp(H["a"] + H["b"] * l), H["kap"], H["lam"], H["mu"])[0]
        pA = predictions(e, e["K"], np.exp(A["a"] + A["b"] * l), A["kap"], A["lam"], 1.0)[0]
        pK = e["K"](e["t_obs"]); o = e["pop_rel"]
        rows.append(dict(g=g, name=e["name"], src=e["src"], loss=float(e["loss"]), mu_train=H["mu"],
                         train_H=dict(kap=H["kap"], lam=H["lam"], a=H["a"], b=H["b"], sdS=H["sdS"]), t=e["t_obs"].tolist(), obs=o.tolist(),
                         H=pH.tolist(), A=pA.tolist(), Kpath=pK.tolist(),
                         rmse=dict(H=rmse(pH, o), A=rmse(pA, o), K=rmse(pK, o), none=rmse(np.ones_like(o), o))))
        print(f"{g:12s} {e['name']:20s} loss {e['loss']:.3f}  rmse H {rows[-1]['rmse']['H']:.3f} A {rows[-1]['rmse']['A']:.3f} N=K {rows[-1]['rmse']['K']:.3f} none {rows[-1]['rmse']['none']:.3f}", flush=True)
summ = {m: dict(median=float(np.median([r["rmse"][m] for r in rows])), mean=float(np.mean([r["rmse"][m] for r in rows])),
                wins_vs_K=int(sum(r["rmse"][m] < r["rmse"]["K"] for r in rows))) for m in ("H", "A", "K", "none")}
print(json.dumps(summ, indent=1))
json.dump(dict(rows=rows, summary=summ, n_events=len(EV), n_disasters=len(groups), settings=dict(tau=TAU, min_loss=MIN_LOSS),
               all=dict(H={k: (v.tolist() if hasattr(v, "tolist") else v) for k, v in H_all.items()},
                        A={k: (v.tolist() if hasattr(v, "tolist") else v) for k, v in A_all.items()},
                        events=[(e["g"], e["name"], e["src"], float(e["loss"])) for e in EV])),
          open(OUT / f"holdout{TAG}.json", "w"), indent=1)
