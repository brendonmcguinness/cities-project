"""Gravity estimates (PPML, robust SE).
  (1) network-wide: all ordered pairs of the model's metros, pre-disaster flows (mean of 2003-04 and 2004-05)
      -> alpha (origin size), beta (destination size), gamma (distance); these parameterise the model network
  (2) New Orleans only, county level and metro level, by year and flow type (descriptive; tests whether
      displacement followed the same law as normal migration)
  (3) cluster bootstrap over metros for the network-wide estimates
Writes output/gravity.json."""
import json
import numpy as np
import statsmodels.api as sm
from common import OUT, NOLA, BASE_YEARS, metros, metro_flows, counties, county_flows, hav, NOLA_PARISHES

rng = np.random.default_rng(20051829)
m = metros(); codes = m.index.tolist(); idx = {c: i for i, c in enumerate(codes)}
MF = metro_flows().reset_index()
MF = MF[MF.o.isin(codes) & MF.d.isin(codes)]
n = len(codes)
Y = np.zeros((n, n))
for _, r in MF.iterrows():
    Y[idx[r.o], idx[r.d]] = np.mean([r[y] for y in BASE_YEARS])
lp = np.log(m.p2005.values); lat, lon = m.lat.values, m.lon.values
D = hav(lat[:, None], lon[:, None], lat[None, :], lon[None, :])
I, J = np.where(~np.eye(n, dtype=bool))


def fit_pairs(I, J):
    X = sm.add_constant(np.column_stack([lp[I], lp[J], -np.log(D[I, J])]))
    r = sm.GLM(Y[I, J], X, family=sm.families.Poisson()).fit(cov_type="HC1")
    return np.asarray(r.params), np.asarray(r.bse), 1 - r.deviance / r.null_deviance


p, se, r2 = fit_pairs(I, J)
res = {"network": dict(alpha=p[1], alpha_se=se[1], beta=p[2], beta_se=se[2], gamma=p[3], gamma_se=se[3],
                       pseudoR2=r2, n_pairs=int(len(I)), n_metros=n, share_pairs_positive=float((Y[I, J] > 0).mean()))}
# cluster bootstrap over metros (resample metros, keep all pairs among the resampled set)
boots = []
for b in range(200):
    s = rng.choice(n, n, replace=True)
    ii, jj = np.meshgrid(s, s, indexing="ij"); ii, jj = ii.ravel(), jj.ravel(); k = ii != jj
    try: boots.append(fit_pairs(ii[k], jj[k])[0][1:])
    except Exception: pass
boots = np.array(boots)
res["network"]["bootstrap"] = dict(n=int(len(boots)), alpha_ci=np.percentile(boots[:, 0], [2.5, 97.5]).tolist(),
                                   beta_ci=np.percentile(boots[:, 1], [2.5, 97.5]).tolist(), gamma_ci=np.percentile(boots[:, 2], [2.5, 97.5]).tolist(),
                                   alpha_sd=float(boots[:, 0].std()), beta_sd=float(boots[:, 1].std()), gamma_sd=float(boots[:, 2].std()))


def ppml(y, pop, dist):
    X = sm.add_constant(np.column_stack([np.log(pop), np.log(dist)]))
    r = sm.GLM(y, X, family=sm.families.Poisson()).fit(cov_type="HC1")
    q, s = np.asarray(r.params), np.asarray(r.bse)
    return dict(beta=q[1], beta_se=s[1], gamma=-q[2], gamma_se=s[2], N=float(np.sum(y)), pseudoR2=1 - r.deviance / r.null_deviance)


# New Orleans, metro level
h = m.drop(index=NOLA)
res["nola_metro"] = {y: ppml(h[y].values, h.p2005.values, h.dist.values)
                     for y in ["o_base", "o0506", "excess", "o0607", "i_base", "i0607"]}
# New Orleans, county level: all counties with a flow plus all counties >= 100k
c = counties(); F = county_flows()
par = c[c.fips.isin(NOLA_PARISHES)]
lat0, lon0 = np.average(par.lat, weights=par.p2005), np.average(par.lon, weights=par.p2005)
cc = c[~c.fips.isin(NOLA_PARISHES)].copy(); cc["dist"] = hav(lat0, lon0, cc.lat, cc.lon)
for y in ["0304", "0405", "0506", "0607"]:
    f = F[F.year == y]
    cc[f"o{y}"] = f[f.orig.isin(NOLA_PARISHES)].groupby("dest").exemptions.sum().reindex(cc.fips).fillna(0).values
    cc[f"i{y}"] = f[f.dest.isin(NOLA_PARISHES)].groupby("orig").exemptions.sum().reindex(cc.fips).fillna(0).values
cc["o_base"], cc["i_base"] = (cc.o0304 + cc.o0405) / 2, (cc.i0304 + cc.i0405) / 2
cc["excess"] = (cc.o0506 - cc.o_base).clip(lower=0)
samp = cc[(cc[["o0304", "o0405", "o0506", "o0607", "i0304", "i0405", "i0607"]].sum(1) > 0) | (cc.p2005 >= 100_000)]
res["nola_county"] = {y: ppml(samp[y].values, samp.p2005.values, samp.dist.values)
                      for y in ["o_base", "o0506", "excess", "o0607", "i_base", "i0607"]}
res["nola_county"]["n_counties"] = int(len(samp))
json.dump(res, open(OUT / "gravity.json", "w"), indent=1, default=float)
g = res["network"]
print(f"network ({g['n_pairs']} pairs, {g['share_pairs_positive']:.0%} non-zero): alpha={g['alpha']:.3f}±{g['alpha_se']:.3f} beta={g['beta']:.3f}±{g['beta_se']:.3f} "
      f"gamma={g['gamma']:.3f}±{g['gamma_se']:.3f} R2={g['pseudoR2']:.2f}; bootstrap CIs {g['bootstrap']}")
for lvl in ["nola_metro", "nola_county"]:
    for k, v in res[lvl].items():
        if isinstance(v, dict): print(f"{lvl:11s} {k:7s} beta={v['beta']:.2f}±{v['beta_se']:.2f} gamma={v['gamma']:.2f}±{v['gamma_se']:.2f} N={v['N']:.0f} R2={v['pseudoR2']:.2f}")
