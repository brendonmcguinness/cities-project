"""Does the dynamic model beat a simple statistical forecast in the held-out test?

Same 21 counties and leave-one-disaster-out folds as 14_holdout.py (output/holdout.json supplies, for each county,
the observation times, population relative to trend and measured capacity at those times). Baselines fitted by
least squares on the other disasters' county-years and applied to the held-out county:
  R1  N - 1 = b (K - 1)                               population gap proportional to capacity gap
  R2  N - 1 = b1 (K - 1) + b2 (K - 1) t                ... with a gap that changes with time since the event
  R3  N - 1 = b1 (K - 1) + b2 (K - 1) t + b3 (K0 - 1)  ... plus the first-year capacity loss K0 (persistent effect)
Compared with the pooled-phi* model (H) by paired Wilcoxon tests on county errors and a cluster bootstrap over
disasters. Writes output/baseline.json."""
import json
import numpy as np
from scipy.stats import wilcoxon
from common import OUT

rows = json.load(open(OUT / "holdout.json"))["rows"]
rng = np.random.default_rng(1)


def design(r, kind):
    t, K = np.array(r["t"]), np.array(r["Kpath"]); g = K - 1; g0 = np.full_like(g, K[0] - 1)
    cols = {"R1": [g], "R2": [g, g * t], "R3": [g, g * t, g0]}[kind]
    return np.column_stack(cols), np.array(r["obs"]) - 1


res = {"rows": []}
for r in rows:
    out = dict(name=r["name"], g=r["g"], loss=r["loss"], H=r["rmse"]["H"], K=r["rmse"]["K"])
    for kind in ("R1", "R2", "R3"):
        X, y = zip(*[design(q, kind) for q in rows if q["g"] != r["g"]]); X, y = np.vstack(X), np.concatenate(y)
        b = np.linalg.lstsq(X, y, rcond=None)[0]
        Xh, yh = design(r, kind); out[kind] = float(np.sqrt(np.mean((Xh @ b - yh) ** 2)))
    res["rows"].append(out)
E = {k: np.array([r[k] for r in res["rows"]]) for k in ("H", "K", "R1", "R2", "R3")}
g = sorted(set(r["g"] for r in res["rows"]))
for k in ("R1", "R2", "R3"):
    d = E[k] - E["H"]; boot = []
    for _ in range(10000):
        idx = [i for gg in rng.choice(g, len(g)) for i, r in enumerate(res["rows"]) if r["g"] == gg]
        boot.append(np.median(E[k][idx]) - np.median(E["H"][idx]))
    res[k] = dict(median=float(np.median(E[k])), wins_H=int(np.sum(E["H"] < E[k])), wilcoxon_p=float(wilcoxon(E[k], E["H"]).pvalue),
                  diff_median_ci95=[float(v) for v in np.percentile(boot, [2.5, 97.5])],
                  severe_median=float(np.median([r[k] for r in res["rows"] if r["loss"] >= 0.10])))
    print(f"{k}: median {100*res[k]['median']:.1f} pp (H {100*np.median(E['H']):.1f}); H better in {res[k]['wins_H']}/21; Wilcoxon p {res[k]['wilcoxon_p']:.3f}; "
          f"diff of medians 95% [{100*res[k]['diff_median_ci95'][0]:.1f}, {100*res[k]['diff_median_ci95'][1]:.1f}]; severe median {100*res[k]['severe_median']:.1f} (H {100*np.median([r['H'] for r in res['rows'] if r['loss'] >= 0.1]):.1f})")
json.dump(res, open(OUT / "baseline.json", "w"), indent=1)
