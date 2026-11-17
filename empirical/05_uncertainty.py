"""Propagate parameter uncertainty to the model's predictions and validation metrics.
  (1) stage-A uncertainty: refit stages B and C for each leave-one-year-out (jackknife) fit of the housing curve
  (2) stage-B/C uncertainty: 200 Monte Carlo draws of (S_s, kappa, lam) from their Gauss-Newton covariance (log scale)
      and of rho_h from its lognormal CI, simulated through 2009
Writes output/uncertainty.json and output/uncertainty_bands.npz."""
import json
import numpy as np
from common import OUT, JULY, APRIL, LANDFALL, nola_series
from model import Network, unpack
from calib import calibrate, sim_from, fit_A

rng = np.random.default_rng(29082005)
C = json.load(open(OUT / "calibration.json")); A, B, Cc = C["A"], C["B"], C["C"]
net = Network(); s = net.s; te = np.linspace(0, 4.0, 401)

# (1) jackknife over housing years -> refit B, C
jk = []
for k, x in enumerate(A["jackknife"]):
    Ak = dict(A, d0=x[0], dinf=x[1], rho_s=x[2])
    r = calibrate(net, A=Ak, verbose=False)
    jk.append(dict(drop_year=2006 + k, d0=x[0], dinf=x[1], rho_s=x[2], **{n: r["B"][n] for n in r["B"]["names"]}, rho_h=r["C"]["rho_h"]))
    print("jackknife", jk[-1])

# (2) Monte Carlo over B and C
mu, cov = np.array(B["x_log"]), np.array(B["cov_log"])
draws, pops, dms = [], [], []
for i in range(200):
    p = np.exp(rng.multivariate_normal(mu, cov))
    rh = float(np.exp(np.log(Cc["rho_h"]) + rng.normal() * Cc["se_log"]))
    Bi = dict(B, **dict(zip(B["names"], p)))
    u = unpack(sim_from(net, A, Bi, rh, C["S_med"], "empirical", t_eval=te), net.n)
    pops.append(u["N"][s] / net.K0[s])
    dm = []
    for y in (2006, 2007, 2008, 2009):
        a, b = max(y - 0.5 - LANDFALL, 0), y + 0.5 - LANDFALL
        ci, co = np.interp([a, b], u["t"], u["cum_in"]), np.interp([a, b], u["t"], u["cum_out"])
        dm.append(((ci[1] - ci[0]) - (co[1] - co[0])) * 1e6)
    dms.append(dm); draws.append(list(p) + [rh])
pops, dms = np.array(pops), np.array(dms)
band = np.percentile(pops, [2.5, 50, 97.5], axis=0)
np.savez(OUT / "uncertainty_bands.npz", t=te, pop_lo=band[0], pop_med=band[1], pop_hi=band[2],
         dm_lo=np.percentile(dms, 2.5, 0), dm_med=np.median(dms, 0), dm_hi=np.percentile(dms, 97.5, 0))
ns = nola_series()
cover = [bool(band[0][np.searchsorted(te, t)] <= y <= band[2][np.searchsorted(te, t)]) for t, y in zip(ns["t"], ns["pop_rel"])]
out = dict(jackknife=jk, mc=dict(n=len(draws), param_names=B["names"] + ["rho_h"],
           param_ci95=np.percentile(np.array(draws), [2.5, 97.5], axis=0).T.tolist(),
           nola_dm_ci95=np.percentile(dms, [2.5, 97.5], axis=0).T.tolist(), pop_band_covers_census=cover))
json.dump(out, open(OUT / "uncertainty.json", "w"), indent=1, default=float)
print("MC parameter 95% intervals:", dict(zip(out["mc"]["param_names"], np.round(out["mc"]["param_ci95"], 3).tolist())))
print("NOLA net domestic migration 95% band (k):", np.round(np.array(out["mc"]["nola_dm_ci95"]) / 1e3, 1).tolist())
print("population band covers Census:", cover)
