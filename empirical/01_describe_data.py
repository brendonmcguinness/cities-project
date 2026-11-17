"""Describe the data (nothing is fitted). Writes output/data_summary.json and output/fig_data.{png,pdf}."""
import json
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from common import (OUT, NOLA, NOLA_PARISHES, BASE_YEARS, OTHER_STRUCK, counties, county_flows, metros, metros_all,
                    nola_totals, nola_series, hav)

c, F, m, T, ns = counties(), county_flows(), metros(), nola_totals(), nola_series()
S = {}
# IRS coverage and totals
S["irs_totals"] = T.reset_index().to_dict("records")
for y in ["0304", "0405", "0506", "0607", "0708"]:
    f = F[F.year == y]
    o = f[f.orig.isin(NOLA_PARISHES) & ~f.dest.isin(NOLA_PARISHES)]; i = f[f.dest.isin(NOLA_PARISHES) & ~f.orig.isin(NOLA_PARISHES)]
    S[f"irs_{y}"] = dict(out_identified=int(o.exemptions.sum()), out_counties=int(o.dest.nunique()), in_identified=int(i.exemptions.sum()),
                         in_counties=int(i.orig.nunique()), share_out_identified=float(o.exemptions.sum() / T.loc[y, "nola_out_total"]))
S["irs_county_pairs_per_year"] = F.groupby("year").size().to_dict()
base_out = T.loc[BASE_YEARS, "nola_out_total"].mean(); base_in = T.loc[BASE_YEARS, "nola_in_total"].mean()
S["irs_excess_out_0506"] = float(T.loc["0506", "nola_out_total"] - base_out)
S["irs_excess_in"] = {y: float(T.loc[y, "nola_in_total"] - base_in) for y in ["0506", "0607", "0708"]}
# geography of flows (county level)
par = c[c.fips.isin(NOLA_PARISHES)]
lat0, lon0 = np.average(par.lat, weights=par.p2005), np.average(par.lon, weights=par.p2005)
cc = c[~c.fips.isin(NOLA_PARISHES)].set_index("fips"); cc["dist"] = hav(lat0, lon0, cc.lat, cc.lon)
def cflow(y, direction):
    f = F[F.year == y]
    if direction == "out": return f[f.orig.isin(NOLA_PARISHES)].groupby("dest").exemptions.sum().reindex(cc.index).fillna(0)
    return f[f.dest.isin(NOLA_PARISHES)].groupby("orig").exemptions.sum().reindex(cc.index).fillna(0)
prof = {"baseline out": (cflow("0304", "out") + cflow("0405", "out")) / 2}
prof["excess out 2005-06"] = (cflow("0506", "out") - prof["baseline out"]).clip(lower=0)
prof["return in 2006-07"] = (cflow("0607", "in") - (cflow("0304", "in") + cflow("0405", "in")) / 2).clip(lower=0)
for k, w in prof.items():
    S[f"share_within_300km::{k}"] = float(w[cc.dist < 300].sum() / w.sum())
h = m.drop(index=[NOLA])
for k, col in [("baseline out", "o_base"), ("excess out 2005-06", "excess"), ("return in 2006-07", "excess_ret")]:
    sh = h[col] / h[col].sum()
    S[f"concentration::{k}"] = dict(top=h.name[sh.idxmax()], top_share=float(sh.max()), herfindahl=float((sh ** 2).sum()))
# Census
a = metros_all().loc[NOLA]
S["census_nola"] = dict(pop={y: int(a[f"p{y}"]) for y in range(2003, 2010)}, hu={y: int(a[f"hu{y}"]) for y in range(2003, 2010)},
                        dm={y: int(a[f"dm{y}"]) for y in range(2002, 2010)}, rel_trend=ns)
oc = c.set_index("fips").loc["22071"]
S["census_orleans_parish"] = dict(pop={y: int(oc[f"p{y}"]) for y in (2005, 2006, 2009)}, hu={y: int(oc[f"hu{y}"]) for y in (2005, 2006, 2009)})
top = h.drop(index=[x for x in OTHER_STRUCK if x in h.index]).nlargest(8, "excess")
S["hosts"] = [dict(code=k, name=r["name"], irs_excess_arrivals=float(r.excess), dm_excess2006=float(r.dm_excess2006),
                   hu_excess2006=float(r.hu_excess2006), hu_excess2007=float(r.hu_excess2007), occ2006=float(r.occ2006), occ2007=float(r.occ2007))
              for k, r in top.iterrows()]
S["network"] = dict(n_metros=int(len(m)), pop2005=int(m.p2005.sum()),
                    share_identified_out_0506_to_network=float(h.o0506.sum() / S["irs_0506"]["out_identified"]))
json.dump(S, open(OUT / "data_summary.json", "w"), indent=1, default=float)
print(json.dumps({k: v for k, v in S.items() if k not in ("census_nola", "hosts", "irs_totals")}, indent=1, default=float))

# ---- figure
import figstyle
figstyle.apply()
fig, ax = plt.subplots(2, 2, figsize=(10.5, 8)); ax = ax.ravel()
yy = np.arange(2002, 2010); A = metros_all()
ax[0].plot(yy, [A.loc[NOLA, f"dm{y}"] / 1e3 for y in yy], "o-", color="k", ms=4, lw=1.2, label="New Orleans")
for code, col, mk in [("26420", "tab:red", "s"), ("12940", "tab:blue", "^"), ("12060", "tab:green", "v"), ("19100", "tab:orange", "D")]:
    ax[0].plot(yy, [A.loc[code, f"dm{y}"] / 1e3 for y in yy], marker=mk, color=col, ms=4, lw=1.2, label=A.loc[code, "name"].split("-")[0].split(",")[0])
ax[0].axhline(0, color="0.6", lw=0.6); ax[0].set_ylim(-350, 120)
ax[0].set_xlabel("year ending 30 June"); ax[0].set_ylabel("net domestic migration (thousands)"); ax[0].legend(frameon=False, loc="lower left")
yrs = np.array(ns["years"]) + 0.5
ax[1].plot(np.r_[2005.5, yrs], np.r_[1, ns["hu_rel"]], "s-", color="tab:gray", ms=4, lw=1.2, label="housing units")
ax[1].plot(np.r_[2005.5, yrs], np.r_[1, ns["pop_rel"]], "o-", color="k", ms=4, lw=1.2, label="population")
ax[1].axvline(2005.66, color="0.6", lw=0.8, ls="--")
ax[1].set_xlim(2005, 2010); ax[1].set_xticks(range(2005, 2011)); ax[1].set_ylim(0.6, 1.05); ax[1].set_xlabel("year"); ax[1].set_ylabel("New Orleans metro / 2001–05 trend"); ax[1].legend(frameon=False, loc="lower right")
order = np.argsort(cc.dist.values); dist = cc.dist.values[order]
for (k, w), col, ls in zip(prof.items(), ["0.4", "tab:red", "tab:blue"], ["--", "-", "-"]):
    v = w.values[order]; ax[2].plot(dist, np.cumsum(v) / v.sum(), color=col, ls=ls, lw=1.4, label=k)
ax[2].set_xscale("log"); ax[2].set_xlim(50, 5000); ax[2].set_ylim(0, 1)
ax[2].set_xlabel("distance from New Orleans (km)"); ax[2].set_ylabel("cumulative share of migrants"); ax[2].legend(frameon=False, loc="upper left")
sb, se = h.o_base / h.o_base.sum(), h.excess / h.excess.sum()
idx = se.sort_values(ascending=False).index[:10][::-1]; yp = np.arange(len(idx))
ax[3].plot(sb[idx], yp, "o", mfc="none", mec="0.3", ms=6, label="baseline out (2003–05)")
ax[3].plot(se[idx], yp, "o", color="tab:red", ms=5, label="excess out 2005–06")
for k, i in enumerate(idx): ax[3].plot([sb[i], se[i]], [k, k], color="0.75", lw=0.8, zorder=0)
ax[3].set_yticks(yp, [h.name[i].split("-")[0].split(",")[0] for i in idx]); ax[3].set_xlabel("share of metro-bound migrants")
ax[3].legend(frameon=False, loc="lower right")
for a_, l in zip(ax, "abcd"): a_.text(-0.16 if l != "d" else -0.3, 1.02, l, transform=a_.transAxes, fontsize=12, fontweight="bold", va="bottom")
fig.tight_layout(w_pad=2.0, h_pad=2.0)
fig.savefig(OUT / "fig_data.png"); fig.savefig(OUT / "fig_data.pdf")
