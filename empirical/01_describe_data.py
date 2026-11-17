"""Describe the data: coverage, key numbers and a descriptive figure. Writes output/data_summary.json and
output/fig_data.{png,pdf}. Nothing here is fitted."""
import json
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from common import counties, metros, parishes, flows_county, OUT, IRS_TOTAL_OUT, IRS_TOTAL_IN, NOLA

f, d, m, p = flows_county(), counties(), metros(), parishes()
yrs = ["0405", "0506", "0607", "0708"]
S = {}
# --- IRS coverage
S["irs_n_dest_counties"] = {y: int((f["o" + y] > 0).sum()) for y in yrs}
S["irs_n_orig_counties"] = {y: int((f["i" + y] > 0).sum()) for y in yrs}
S["irs_total_out"], S["irs_total_in"] = IRS_TOTAL_OUT, IRS_TOTAL_IN
S["irs_identified_out"] = {y: int(f["o" + y].sum()) for y in yrs}
S["irs_identified_share_out"] = {y: round(f["o" + y].sum() / IRS_TOTAL_OUT[y], 3) for y in yrs}
host = m[m.code != NOLA]
S["metro_share_of_identified_out"] = {y: round(host["o" + y].sum() / f["o" + y].sum(), 3) for y in yrs}
S["excess_out_0506"] = int(IRS_TOTAL_OUT["0506"] - IRS_TOTAL_OUT["0405"])
S["excess_in"] = {y: int(IRS_TOTAL_IN[y] - IRS_TOTAL_IN["0405"]) for y in yrs[1:]}
# --- distance profile of flows (identified county flows)
for lab, col in [("baseline_0405", "o0405"), ("katrina_0506", "o0506"), ("excess_0506", "excess"), ("return_0607", "i0607")]:
    w = d[col]
    S[f"median_dist_km_{lab}"] = float(np.round(np.interp(0.5, np.cumsum(w[np.argsort(d.dist)]) / w.sum(), np.sort(d.dist)), 0))
    S[f"share_within_300km_{lab}"] = round(w[d.dist < 300].sum() / w.sum(), 3)
# --- concentration: share of identified flow in top destination metro, and Herfindahl index over metros
for lab, col in [("baseline_0405", "o0405"), ("excess_0506", "excess"), ("return_0607", "excess_ret")]:
    sh = host[col] / host[col].sum()
    S[f"top_metro_{lab}"] = [host.name[sh.idxmax()], round(sh.max(), 3)]
    S[f"herfindahl_{lab}"] = round((sh ** 2).sum(), 3)
# --- Census
core = p[p.name.isin(["Jefferson", "Orleans", "Plaquemines", "St Bernard"])]
for lab, q in [("metro", p), ("flooded4", core), ("orleans", p[p.name == "Orleans"])]:
    a, b, c = q.p2005.sum(), q.p2006.sum(), q.p2009.sum()
    S[f"census_{lab}"] = dict(p2005=int(a), p2006=int(b), p2009=int(c), drop=round(1 - b / a, 3), recovered_by_2009=round((c - b) / (a - b), 3))
nola = m[m.code == NOLA].iloc[0]
S["census_nola_domestic_migration"] = {y: int(nola[f"dm{y}"]) for y in range(2002, 2010)}
top = host.nlargest(6, "dm_excess2006")
S["census_top_hosts_dm_excess2006"] = {r["name"]: int(r.dm_excess2006) for _, r in top.iterrows()}
S["metros"] = dict(n=int(len(m)), pop2005=int(m.p2005.sum()))
json.dump(S, open(OUT / "data_summary.json", "w"), indent=1)
print(json.dumps(S, indent=1))

# --- descriptive figure
mpl.rcdefaults()
plt.rcParams.update({"font.size": 10, "legend.fontsize": 8.5, "xtick.direction": "in", "ytick.direction": "in",
                     "xtick.top": True, "ytick.right": True, "savefig.dpi": 300})
fig, ax = plt.subplots(1, 3, figsize=(13, 3.7))
yy = np.arange(2002, 2010)
ax[0].plot(yy, [nola[f"dm{y}"] / 1e3 for y in yy], "o-", color="k", ms=4, lw=1.2, label="New Orleans")
for nm, c, mk in [("Houston TX", "tab:red", "s"), ("Baton Rouge LA", "tab:blue", "^"), ("Atlanta GA", "tab:green", "v"), ("Dallas TX", "tab:orange", "D")]:
    r = m[m.name == nm].iloc[0]
    ax[0].plot(yy, [r[f"dm{y}"] / 1e3 for y in yy], marker=mk, color=c, ms=4, lw=1.2, label=nm.rsplit(" ", 1)[0])
ax[0].axhline(0, color="0.6", lw=0.6)
ax[0].set_xlabel("year ending 30 June"); ax[0].set_ylabel("net domestic migration (thousands)")
ax[0].set_ylim(-350, 120); ax[0].legend(frameon=False, loc="lower left", ncol=1)
# b: cumulative distance profile
order = np.argsort(d.dist.values); dist = d.dist.values[order]
for lab, col, c, ls in [("out 2004–05 (baseline)", "o0405", "0.4", "--"), ("excess out 2005–06", "excess", "tab:red", "-"), ("in 2006–07 (return)", "i0607", "tab:blue", "-")]:
    w = d[col].values[order]; ax[1].plot(dist, np.cumsum(w) / w.sum(), color=c, ls=ls, lw=1.4, label=lab)
ax[1].set_xscale("log"); ax[1].set_xlim(50, 5000); ax[1].set_ylim(0, 1)
ax[1].set_xlabel("distance from New Orleans (km)"); ax[1].set_ylabel("cumulative share of migrants"); ax[1].legend(frameon=False, loc="upper left")
# c: top destination metros, baseline vs Katrina shares
sb, se = host.o0405 / host.o0405.sum(), host.excess / host.excess.sum()
idx = se.sort_values(ascending=False).index[:10][::-1]
yp = np.arange(len(idx))
ax[2].plot(sb[idx], yp, "o", mfc="none", mec="0.3", ms=6, label="out 2004–05 (baseline)")
ax[2].plot(se[idx], yp, "o", color="tab:red", ms=5, label="excess out 2005–06")
for k, i in enumerate(idx): ax[2].plot([sb[i], se[i]], [k, k], color="0.75", lw=0.8, zorder=0)
ax[2].set_yticks(yp, [host.name[i].rsplit(" ", 1)[0] for i in idx]); ax[2].set_xlabel("share of metro-bound migrants")
ax[2].legend(frameon=False, loc="lower right")
for a, l in zip(ax, "abc"): a.text(-0.18 if l != "c" else -0.32, 1.02, l, transform=a.transAxes, fontsize=12, fontweight="bold", va="bottom")
fig.tight_layout(w_pad=2.0)
fig.savefig(OUT / "fig_data.png"); fig.savefig(OUT / "fig_data.pdf")
