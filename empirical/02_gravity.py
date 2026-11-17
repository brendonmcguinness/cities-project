"""Gravity estimates (PPML) at county and metro level, and the joint pre-disaster fit that sets alpha, beta, gamma.
Writes output/gravity.json."""
import json
import numpy as np
import statsmodels.api as sm
from common import counties, metros, OUT, NOLA


def ppml(y, pop, dist):
    X = sm.add_constant(np.column_stack([np.log(pop), np.log(dist)]))
    r = sm.GLM(y, X, family=sm.families.Poisson()).fit(cov_type="HC1")
    p, se = np.asarray(r.params), np.asarray(r.bse)
    return dict(beta=p[1], beta_se=se[1], gamma=-p[2], gamma_se=se[2], N=float(np.sum(y)), pseudoR2=1 - r.deviance / r.null_deviance)


res = {"county": {}, "metro": {}}
d = counties(); d = d[d.dist > 0]
for y in ["o0405", "o0506", "excess", "o0607", "i0405", "i0607"]:
    res["county"][y] = ppml(d[y].values, d.pop2005.values, d.dist.values)
h = metros(); h = h[h.code != NOLA]
for y in ["o0405", "o0506", "excess", "i0405", "i0607"]:
    res["metro"][y] = ppml(h[y].values, h.p2005.values, h.dist.values)

# joint pre-disaster fit: out-flows NOLA->j (destination exponent beta), in-flows j->NOLA (origin exponent alpha), common gamma
lp, ld, n = np.log(h.p2005.values), np.log(h.dist.values), len(h)
y = np.concatenate([h.o0405.values, h.i0405.values])
X = np.column_stack([np.r_[np.ones(n), np.zeros(n)], np.r_[np.zeros(n), np.ones(n)],
                     np.r_[lp, np.zeros(n)], np.r_[np.zeros(n), lp], -np.r_[ld, ld]])
r = sm.GLM(y, X, family=sm.families.Poisson()).fit(cov_type="HC1")
p, se = np.asarray(r.params), np.asarray(r.bse)
res["joint_pre_disaster"] = dict(beta=p[2], beta_se=se[2], alpha=p[3], alpha_se=se[3], gamma=p[4], gamma_se=se[4],
                                 pseudoR2=1 - r.deviance / r.null_deviance)
json.dump(res, open(OUT / "gravity.json", "w"), indent=1)
for lvl in ["county", "metro"]:
    for k, v in res[lvl].items():
        print(f"{lvl:6s} {k:7s} beta={v['beta']:.2f}±{v['beta_se']:.2f} gamma={v['gamma']:.2f}±{v['gamma_se']:.2f} N={v['N']:.0f} R2={v['pseudoR2']:.2f}")
print("joint", {k: round(v, 3) for k, v in res["joint_pre_disaster"].items()})
