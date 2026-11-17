"""Calibration and validation figure (output/fig_calibration.{png,pdf}) and county gravity figure (output/fig_gravity.{png,pdf})."""
import json
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from common import OUT, counties, LANDFALL
from model import M, S_IDX

mpl.rcdefaults()
plt.rcParams.update({"font.size": 10, "legend.fontsize": 8.5, "xtick.direction": "in", "ytick.direction": "in",
                     "xtick.top": True, "ytick.right": True, "savefig.dpi": 300})
C = json.load(open(OUT / "calibration.json")); V = json.load(open(OUT / "validation.json"))
tr = np.load(OUT / "traj_empirical.npz")
ve = pd.read_csv(OUT / "validation_empirical.csv", dtype={"code": str}); vg = pd.read_csv(OUT / "validation_gravity.csv", dtype={"code": str})
ve, vg = ve[ve.code != "35380"], vg[vg.code != "35380"]

fig, ax = plt.subplots(2, 3, figsize=(13, 7.6)); ax = ax.ravel()
# a: NOLA population relative to trend
t = tr["t"]; K0s = tr["N_s"][0]
ax[0].plot(LANDFALL + t, tr["N_s"] / K0s, "k-", lw=1.4, label="model population")
ax[0].plot(LANDFALL + t, tr["K_s"] / K0s, "k:", lw=1.0, label="model capacity")
ax[0].plot([2006.5, 2007.5, 2008.5, 2009.5], C["n_obs"], "o", color="tab:red", ms=5, label="Census (1 July)")
ax[0].set_xlim(2005, 2010); ax[0].set_ylim(0.6, 1.05); ax[0].set_xlabel("year"); ax[0].set_ylabel("NOLA metro population / 2001–05 trend")
ax[0].legend(frameon=False, loc="lower right")
# b: NOLA net domestic migration per Census year vs model net flow
yrs = np.arange(2006, 2010)
cen = [M[f"dm{y}"][S_IDX] - M.dm_base[S_IDX] for y in yrs]
mod = []
for y in yrs:
    a, b = max((y - 1) + 0.5 - LANDFALL, 0), y + 0.5 - LANDFALL
    ci, co = np.interp([a, b], t, tr["cum_in"]), np.interp([a, b], t, tr["cum_out"])
    mod.append(((ci[1] - ci[0]) - (co[1] - co[0])) * 1e6)
ax[1].plot(yrs, np.array(cen) / 1e3, "o-", color="tab:red", ms=5, lw=1.2, label="Census, above 2002–05 mean")
ax[1].plot(yrs, np.array(mod) / 1e3, "s--", color="k", ms=4, lw=1.2, label="model net flow")
ax[1].axhline(0, color="0.6", lw=0.6); ax[1].set_xticks(yrs, [f"{y-1}–{str(y)[2:]}" for y in yrs])
ax[1].set_xlabel("year (July–June)"); ax[1].set_ylabel("NOLA net domestic migration (thousands)"); ax[1].legend(frameon=False, loc="lower right")
ins = ax[1].inset_axes([0.42, 0.33, 0.53, 0.38])
ins.plot(yrs[1:], np.array(cen[1:]) / 1e3, "o-", color="tab:red", ms=3, lw=1); ins.plot(yrs[1:], np.array(mod[1:]) / 1e3, "s--", color="k", ms=3, lw=1)
ins.set_xticks(yrs[1:], [f"{str(y)[2:]}" for y in yrs[1:]], fontsize=7); ins.tick_params(labelsize=7); ins.set_ylim(0, 50); ins.set_title("return years", fontsize=7.5)
def loglog(a, o, m, mg, xl, yl):
    k = o > 0
    a.scatter(mg[k], o[k], s=10, facecolor="none", edgecolor="tab:gray", lw=0.6, label="gravity network")
    a.scatter(m[k], o[k], s=10, color="tab:blue", lw=0, label="empirical network")
    a.plot([1e-5, 1], [1e-5, 1], "k--", lw=0.8); a.set_xscale("log"); a.set_yscale("log")
    a.set_xlim(1e-5, 0.6); a.set_ylim(1e-4, 0.6); a.set_xlabel(xl); a.set_ylabel(yl); a.legend(frameon=False, loc="lower right")
loglog(ax[2], ve.obs_dest.values, ve.mod_dest.values, vg.mod_dest.values, "model share of host gains (April 2006)", "observed share, IRS excess out 2005–06")
loglog(ax[3], ve.obs_ret.values, ve.mod_ret.values, vg.mod_ret.values, "model share of return flow", "observed share, IRS excess in 2006–07")
# e: host net domestic migration excess vs model gain
nr = ve[ve.near]
ax[4].scatter(nr.mod_gain06 / 1e3, nr.dm_excess2006 / 1e3, s=12, color="tab:blue", lw=0)
for nm, off, ha in [("Houston TX", (4, -2), "left"), ("Baton Rouge LA", (4, -2), "left"), ("Dallas TX", (5, -6), "left"), ("Atlanta GA", (-5, 3), "right")]:
    r = nr[nr.name == nm].iloc[0]
    ax[4].annotate(nm.rsplit(" ", 1)[0], (r.mod_gain06 / 1e3, r.dm_excess2006 / 1e3), xytext=off, textcoords="offset points", fontsize=8,
                   va="top" if off[1] < 0 else "bottom", ha=ha)
lim = [-15, 90]; ax[4].plot(lim, lim, "k--", lw=0.8); ax[4].set_xlim(lim); ax[4].set_ylim(lim)
ax[4].set_xlabel("model population gain, July 2006 (thousands)"); ax[4].set_ylabel("Census net domestic migration\n2005–06 above 2002–05 mean (thousands)")
# f: stage-2 grid
g = pd.DataFrame(C["grid"]); piv = g.pivot(index="rho_h", columns="S_med", values="misalloc")
im = ax[5].imshow(piv.values, origin="lower", cmap="viridis_r", aspect="auto")
ax[5].set_xticks(range(len(piv.columns)), [f"{x:g}" for x in piv.columns]); ax[5].set_yticks(range(len(piv.index)), [f"{x:g}" for x in piv.index])
ax[5].set_xlabel(r"host coupling $S_{\mathrm{med}}$ (yr$^{-1}$)"); ax[5].set_ylabel(r"host capacity response $\rho_h$ (yr$^{-1}$)")
cb = fig.colorbar(im, ax=ax[5], fraction=0.046, pad=0.03); cb.set_label("misallocated share of destinations")
for a, l in zip(ax, "abcdef"): a.text(-0.2, 1.02, l, transform=a.transAxes, fontsize=12, fontweight="bold", va="bottom")
fig.tight_layout(w_pad=2.0, h_pad=2.0)
fig.savefig(OUT / "fig_calibration.png"); fig.savefig(OUT / "fig_calibration.pdf")

# county gravity figure (destinations of excess out-migration vs gravity kernels)
d = counties(); d = d[d.dist > 0]; G = json.load(open(OUT / "gravity.json"))["county"]["excess"]
obs = d.excess / d.excess.sum(); m = obs > 0
def sh(b, gm): s = d.pop2005 ** b * d.dist ** -gm; return s / s.sum()
fig, a = plt.subplots(figsize=(4.4, 3.9))
a.scatter(sh(1, 1.5)[m], obs[m], s=9, facecolor="none", edgecolor="tab:gray", lw=0.6, label=r"$\beta=1,\ \gamma=1.5$ (report)")
a.scatter(sh(G["beta"], G["gamma"])[m], obs[m], s=9, color="tab:blue", lw=0, alpha=0.8, label=rf"$\beta={G['beta']:.2f},\ \gamma={G['gamma']:.2f}$ (fitted)")
a.plot([1e-6, 1], [1e-6, 1], "k--", lw=0.8); a.set_xscale("log"); a.set_yscale("log"); a.set_xlim(1e-6, 0.5); a.set_ylim(1e-5, 0.5)
a.set_xlabel("predicted share"); a.set_ylabel("observed share, excess out 2005–06"); a.legend(frameon=False, loc="lower right", fontsize=8)
fig.tight_layout(); fig.savefig(OUT / "fig_gravity.png"); fig.savefig(OUT / "fig_gravity.pdf")
print("figures written")
