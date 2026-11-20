"""Uncertainty and robustness of the held-out comparison (output/holdout*.json from 14_holdout.py).

Cluster bootstrap over held-out disasters (all counties of a disaster resampled together, 10,000 draws): intervals
for each predictor's median error and for the median paired difference with the pooled model, and the share of draws
in which the pooled model has the lower median error. Wilcoxon signed-rank tests on the paired county errors.
Sensitivity runs (prior sd tau 0.05 / 0.2; ACS loss threshold 3% / 5%) are summarised when present.
Writes output/holdout_uncertainty.json."""
import json
import numpy as np
from scipy.stats import wilcoxon
from common import OUT

rng = np.random.default_rng(20261118)
M = ("H", "A", "K", "none")


def boot(rows, B=10000):
    g = sorted(set(r["g"] for r in rows)); by = {k: [r for r in rows if r["g"] == k] for k in g}
    E = {m: np.array([r["rmse"][m] for r in rows]) for m in M}
    out = dict(n_counties=len(rows), n_disasters=len(g))
    draws = {m: [] for m in M}; diff = {m: [] for m in M[1:]}
    for _ in range(B):
        s = [r for k in rng.choice(g, len(g)) for r in by[k]]
        e = {m: np.array([r["rmse"][m] for r in s]) for m in M}
        for m in M: draws[m].append(np.median(e[m]))
        for m in M[1:]: diff[m].append(np.median(e[m] - e["H"]))
    for m in M:
        out[m] = dict(median=float(np.median(E[m])), ci95=[float(v) for v in np.percentile(draws[m], [2.5, 97.5])])
    for m in M[1:]:
        d = np.array(diff[m]); w = wilcoxon(E[m], E["H"])
        out["diff_" + m] = dict(median_paired=float(np.median(E[m] - E["H"])), ci95=[float(v) for v in np.percentile(d, [2.5, 97.5])],
                                p_pooled_better=float(np.mean(np.array(draws["H"]) < np.array(draws[m]))),
                                wins=int(np.sum(E["H"] < E[m])), wilcoxon_p=float(w.pvalue))
    return out


res = {"main": boot(json.load(open(OUT / "holdout.json"))["rows"])}
big = [r for r in json.load(open(OUT / "holdout.json"))["rows"] if r["loss"] >= 0.10]
res["loss_ge_10pct"] = boot(big)
for tag in ("_tau005", "_tau02", "_loss3", "_loss5", "_acscap"):
    f = OUT / f"holdout{tag}.json"
    if f.exists():
        d = json.load(open(f)); res[tag[1:]] = dict(summary=d["summary"], n_events=d["n_events"], mu=d["all"]["H"]["mu"],
                                                    boot=boot(d["rows"], B=2000))
for k, v in res.items():
    b = v if "H" in v else v["boot"]
    print(f"{k:14s} n={b['n_counties']:2d}/{b['n_disasters']:2d}  H {100*b['H']['median']:.1f} [{100*b['H']['ci95'][0]:.1f},{100*b['H']['ci95'][1]:.1f}]"
          f"  K-H {100*b['diff_K']['median_paired']:.1f} [{100*b['diff_K']['ci95'][0]:.1f},{100*b['diff_K']['ci95'][1]:.1f}] p_better {b['diff_K']['p_pooled_better']:.2f} wilcox {b['diff_K']['wilcoxon_p']:.3f}"
          f"  A-H {100*b['diff_A']['median_paired']:.1f} [{100*b['diff_A']['ci95'][0]:.1f},{100*b['diff_A']['ci95'][1]:.1f}] wilcox {b['diff_A']['wilcoxon_p']:.3f}"
          f"  none-H {100*b['diff_none']['median_paired']:.1f} wilcox {b['diff_none']['wilcoxon_p']:.3f}")
json.dump(res, open(OUT / "holdout_uncertainty.json", "w"), indent=1)
