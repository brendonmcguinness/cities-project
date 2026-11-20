"""Prediction intervals for the held-out test and their coverage.

For each held-out county, the pooled model trained on the other disasters (14_holdout.py, rows[].train_H) is
simulated 400 times with tolerated occupancy drawn from N(mu_train, tau^2), tau = 0.1, truncated to [0.7, 1.5],
and the evacuation rate drawn from the training regression of log S_s on log housing loss with its residual sd.
Reported: 50% and 90% central intervals, and the share of observed county-years that fall inside them
(well-calibrated intervals cover about 50% and 90%). Writes output/pred_intervals.json."""
import json, pickle
import numpy as np
from common import OUT
from struck import predictions

TAU, NDRAW = 0.10, 400
rng = np.random.default_rng(7)
ev10 = pickle.load(open(OUT / "events_inputs.pkl", "rb")); ev13 = pickle.load(open(OUT / "acs_inputs.pkl", "rb"))
group = lambda d: d.rsplit(" ", 1)[0] if d[-4:].isdigit() else d
EV = {}
for e in ev10 + ev13:
    EV.setdefault((group(e["disaster"]), e["name"]), e)
H = json.load(open(OUT / "holdout.json"))
rows, inside50, inside90, n = [], 0, 0, 0
for r in H["rows"]:
    e = EV[(r["g"], r["name"])]; K = (lambda kt, kv: (lambda s: np.interp(s, kt, kv)))(e["Kt"], e["Kv"])
    if e.get("fy_end") is None: e = dict(e, fy_end=None)
    p = r["train_H"]; sims = []
    for _ in range(NDRAW):
        phis = float(np.clip(rng.normal(r["mu_train"], TAU), 0.7, 1.5))
        S = float(np.exp(p["a"] + p["b"] * np.log(max(r["loss"], 0.01)) + rng.normal(0, p["sdS"])))
        sims.append(predictions(e, K, S, p["kap"], p["lam"], phis)[0])
    sims = np.array(sims); q = np.percentile(sims, [5, 25, 75, 95], axis=0); o = np.array(r["obs"])
    in50 = (o >= q[1]) & (o <= q[2]); in90 = (o >= q[0]) & (o <= q[3])
    inside50 += in50.sum(); inside90 += in90.sum(); n += len(o)
    rows.append(dict(g=r["g"], name=r["name"], loss=r["loss"], t=r["t"], obs=r["obs"], q05=q[0].tolist(), q25=q[1].tolist(), q75=q[2].tolist(), q95=q[3].tolist(),
                     cover50=float(in50.mean()), cover90=float(in90.mean()), width90=float(np.mean(q[3] - q[0]))))
    print(f"{r['g']:12s} {r['name']:20s} 90% width {100*rows[-1]['width90']:.1f} pp  in90 {in90.sum()}/{len(o)}  in50 {in50.sum()}/{len(o)}", flush=True)
res = dict(rows=rows, coverage50=float(inside50 / n), coverage90=float(inside90 / n), n_obs=int(n),
           severe=dict(coverage90=float(np.mean([c for r in rows if r["loss"] >= 0.1 for c in [r["cover90"]]]))))
print({k: v for k, v in res.items() if k != "rows"})
json.dump(res, open(OUT / "pred_intervals.json", "w"), indent=1)
