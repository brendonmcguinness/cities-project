"""Figures for the paper (run after 01-08): fig_calibration, fig_simulations, fig_campfire (fig_data is made by 01)."""
import json
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from common import OUT, LANDFALL, NOLA, nola_series
from model import capacity_struck

mpl.rcdefaults()
plt.rcParams.update({"font.size": 10, "legend.fontsize": 8.5, "xtick.direction": "in", "ytick.direction": "in",
                     "xtick.top": True, "ytick.right": True, "savefig.dpi": 300})
def letters(ax, x=-0.2):
    for a, l in zip(ax, "abcdefgh"): a.text(x, 1.02, l, transform=a.transAxes, fontsize=12, fontweight="bold", va="bottom")

C = json.load(open(OUT / "calibration.json")); V = json.load(open(OUT / "validation.json")); A = C["A"]
U = np.load(OUT / "uncertainty_bands.npz"); ns = nola_series()
vm = pd.read_csv(OUT / "validation_metros.csv", dtype={"code": str}); vm = vm[vm.code != NOLA]

# ---------------- calibration and validation
fig, ax = plt.subplots(2, 3, figsize=(13.5, 8)); ax = ax.ravel()
t = U["t"]; yr = LANDFALL + t; obs_t = np.array(ns["years"]) + 0.5
ax[0].fill_between(yr, U["pop_lo"], U["pop_hi"], color="0.8", lw=0, label="model 95% band")
ax[0].plot(yr, U["pop_med"], "k-", lw=1.4, label="model population")
ax[0].plot(yr, capacity_struck(t, 1.0, A["d0"], A["dinf"], A["rho_s"]), "k:", lw=1.1, label="capacity (fit to housing)")
ax[0].plot(obs_t, ns["hu_rel"], "s", mfc="none", mec="tab:gray", ms=6, label="Census housing units")
ax[0].plot(obs_t, ns["pop_rel"], "o", color="tab:red", ms=5, label="Census population")
ax[0].set_xlim(2005, 2010); ax[0].set_ylim(0.5, 1.05); ax[0].set_xlabel("year"); ax[0].set_ylabel("New Orleans metro / 2001–05 trend")
ax[0].legend(frameon=False, loc="lower right", fontsize=7.5)
yrs = np.arange(2006, 2010); lab = [f"{y-1}–{str(y)[2:]}" for y in yrs]
ax[1].fill_between(yrs[1:], U["dm_lo"][1:] / 1e3, U["dm_hi"][1:] / 1e3, color="0.8", lw=0)
ax[1].plot(yrs[1:], np.array(V["tracked/empirical"]["nola_dm_obs"][1:]) / 1e3, "o-", color="tab:red", ms=5, lw=1.2, label="Census (above 2002–05 mean)")
ax[1].plot(yrs[1:], np.array(V["tracked/empirical"]["nola_dm_model"][1:]) / 1e3, "s-", color="k", ms=4, lw=1.2, label="model, displaced tracked")
ax[1].plot(yrs[1:], np.array(V["gradient/empirical"]["nola_dm_model"][1:]) / 1e3, "^--", color="0.5", ms=4, lw=1.0, label="model, gradient return")
ax[1].set_xticks(yrs[1:], lab[1:]); ax[1].set_ylim(0, 60); ax[1].set_xlabel("year (July–June)"); ax[1].set_ylabel("New Orleans net domestic in-migration (thousands)")
ax[1].legend(frameon=False, loc="upper right", fontsize=7.5)
def loglog(a, o, m, xl, yl):
    k = o > 0
    a.scatter(m[k], o[k], s=11, color="tab:blue", lw=0)
    a.plot([1e-5, 1], [1e-5, 1], "k--", lw=0.8); a.set_xscale("log"); a.set_yscale("log")
    a.set_xlim(1e-5, 0.6); a.set_ylim(1e-4, 0.6); a.set_xlabel(xl); a.set_ylabel(yl)
    h = vm.name.str.startswith("Houston").values & k
    a.annotate("Houston", (m[h][0], o[h][0]), xytext=(-6, 0), textcoords="offset points", ha="right", va="center", fontsize=8)
loglog(ax[2], vm.obs_dest.values, vm.mod_dest.values, "model share of host gains (April 2006)", "observed share, IRS excess out 2005–06")
loglog(ax[3], vm.obs_ret.values, vm.mod_ret.values, "model share of returns (2006–07)", "observed share, IRS excess in 2006–07")
nr = vm[vm.near]
ax[4].scatter(nr.mod_gain06 / 1e3, nr.dm_excess2006 / 1e3, s=12, color="tab:blue", lw=0)
for nm, off, ha in [("Houston", (4, -2), "left"), ("Baton Rouge", (4, -2), "left"), ("Dallas", (5, -6), "left"), ("Atlanta", (-5, 3), "right")]:
    r = nr[nr.name.str.startswith(nm)].iloc[0]
    ax[4].annotate(nm, (r.mod_gain06 / 1e3, r.dm_excess2006 / 1e3), xytext=off, textcoords="offset points", fontsize=8, va="top" if off[1] < 0 else "bottom", ha=ha)
lim = [-15, 90]; ax[4].plot(lim, lim, "k--", lw=0.8); ax[4].set_xlim(lim); ax[4].set_ylim(lim)
ax[4].set_xlabel("model population gain, July 2006 (thousands)"); ax[4].set_ylabel("Census net domestic migration 2005–06\nabove 2002–05 mean (thousands)")
o5 = V["tracked/empirical"]["V5_occ"]
for y in (2006, 2007):
    ax[5].scatter([100 * o[f"mod{y}"] for o in o5], [100 * o[f"obs{y}"] for o in o5], marker="o" if y == 2006 else "s", s=18,
                  label=f"July {y}", color="tab:blue" if y == 2006 else "tab:gray")
for o in o5:
    if o["name"].startswith(("Baton", "Houston", "Atlanta")):
        ax[5].annotate(o["name"].split("-")[0].split(",")[0], (100 * o["mod2006"], 100 * o["obs2006"]), xytext=(4, 0), textcoords="offset points", fontsize=8, va="center")
ax[5].plot([-1, 4], [-1, 4], "k--", lw=0.8); ax[5].set_xlim(-1, 4); ax[5].set_ylim(-1, 4)
ax[5].set_xlabel("model host occupancy above capacity (%)"); ax[5].set_ylabel("Census persons per housing unit\nabove trend (%)"); ax[5].legend(frameon=False, loc="upper left")
letters(ax); fig.tight_layout(w_pad=2.2, h_pad=2.2)
fig.savefig(OUT / "fig_calibration.png"); fig.savefig(OUT / "fig_calibration.pdf")

# ---------------- simulations
S = json.load(open(OUT / "simulations.json")); sm = pd.read_csv(OUT / "simulations_metros.csv", dtype={"code": str})
fig, ax = plt.subplots(1, 3, figsize=(13.5, 3.9))
for scen, col in [("uniform", "0.5"), ("intensity", "tab:blue")]:
    d = sm[sm.scenario == scen]
    ax[0].scatter(d.S_i, d.total, s=10, color=col, lw=0, alpha=0.8, label=f"evacuation rate: {scen}")
n = sm[(sm.scenario == "uniform") & (sm.code == NOLA)].iloc[0]
ax[0].scatter([n.S_i], [n.total], s=40, facecolor="none", edgecolor="k", lw=1.0)
ax[0].annotate("New Orleans", (n.S_i, n.total), xytext=(6, 6), textcoords="offset points", fontsize=8)
ax[0].set_xscale("log"); ax[0].set_yscale("log"); ax[0].set_xlabel(r"evacuation rate of struck city $S_i$ (yr$^{-1}$)")
for axis, ticks in [(ax[0].xaxis, [0.3, 0.5, 1, 2, 3, 5]), (ax[0].yaxis, [0.2, 0.3, 0.4])]:
    axis.set_major_locator(mpl.ticker.FixedLocator(ticks)); axis.set_minor_formatter(mpl.ticker.NullFormatter())
    axis.set_major_formatter(mpl.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
ax[0].set_ylabel("severity (excess-occupancy\nperson-years per resident)"); ax[0].legend(frameon=False, loc="lower left")
dd = S["sweeps_nola"]["delta"]
ax[1].plot([x["d0"] for x in dd], [x["total"] for x in dd], "o-", color="k", ms=4, lw=1.2, label="total")
ax[1].plot([x["d0"] for x in dd], [x["hosts"] for x in dd], "s--", color="tab:blue", ms=4, lw=1.0, label="borne by hosts")
ax[1].axvline(A["d0"], color="0.6", lw=0.8, ls=":"); ax[1].set_xlabel(r"capacity lost at landfall $d_0$"); ax[1].set_ylabel("severity (person-years per resident)")
ax[1].legend(frameon=False, loc="upper left")
rr = S["sweeps_nola"]["rho_h"]
ax[2].axvspan(*C["C"]["ci95"], color="0.9", lw=0, label=r"95% CI of calibrated $\rho_h$")
ax[2].plot([x["rho_h"] for x in rr], [x["total"] for x in rr], "o-", color="k", ms=4, lw=1.2, label="total")
ax[2].plot([x["rho_h"] for x in rr], [x["hosts"] for x in rr], "s--", color="tab:blue", ms=4, lw=1.0, label="borne by hosts")
ax[2].set_xscale("log"); ax[2].set_xlabel(r"host capacity response $\rho_h$ (yr$^{-1}$)"); ax[2].set_ylabel("severity (person-years per resident)")
ax[2].legend(frameon=False, loc="upper right")
letters(ax); fig.tight_layout(w_pad=2.2)
fig.savefig(OUT / "fig_simulations.png"); fig.savefig(OUT / "fig_simulations.pdf")

# ---------------- Camp Fire
CF = json.load(open(OUT / "campfire.json")); ct = np.load(OUT / "campfire_traj.npz")
fig, ax = plt.subplots(1, 2, figsize=(9.5, 3.8))
ax[0].plot(ct["t"], ct["N"], "k-", lw=1.4, label="model (Katrina parameters)")
ax[0].plot(ct["t"], ct["K"], "k:", lw=1.0, label="capacity (housing units)")
po = CF["population_rel_trend"]["observed"]
ax[0].plot([2019.5, 2020.5], [po["2019"], po["2020"]], "o", color="tab:red", ms=5, label="Census population")
ax[0].plot([2018.5, 2019.5, 2020.5], [CF["housing_rel_trend"][k] for k in ("2018", "2019", "2020")], "s", mfc="none", mec="tab:gray", ms=6, label="Census housing units")
ax[0].set_ylim(0.8, 1.02); ax[0].set_xlabel("year"); ax[0].set_ylabel("Butte County / 2014–18 trend"); ax[0].legend(frameon=False, loc="upper right", fontsize=8)
top = CF["destinations"]["top"]
ax[1].scatter([x[2] for x in top], [x[1] for x in top], s=18, color="tab:blue")
NAMES = {"06103": "Tehama", "06101": "Sutter", "06061": "Placer", "06089": "Shasta", "06115": "Yuba", "06067": "Sacramento",
         "06007": "Butte", "06021": "Glenn", "06113": "Yolo", "32031": "Washoe NV", "41029": "Jackson OR"}
for x in top: ax[1].annotate(NAMES.get(x[0], x[0]), (x[2], x[1]), xytext=(4, 2), textcoords="offset points", fontsize=8)
ax[1].plot([0, 0.15], [0, 0.15], "k--", lw=0.8); ax[1].set_xlabel("pre-disaster flow share (2016–18)"); ax[1].set_ylabel("share of excess out-migration 2018–19")
ax[1].set_title(f"six largest destinations (R² over all counties = {CF['destinations']['R2_pre_disaster_shares']:.2f})", fontsize=9)
letters(ax, -0.18); fig.tight_layout(w_pad=2.2)
fig.savefig(OUT / "fig_campfire.png"); fig.savefig(OUT / "fig_campfire.pdf")
print("figures written")
