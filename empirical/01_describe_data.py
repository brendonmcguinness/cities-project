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
import figstyle as fs
fs.apply()
fig, ax = plt.subplots(2, 2, figsize=(fs.W, 4.6)); ax = ax.ravel()
yy = np.arange(2002, 2010); A = metros_all()
# a: net domestic migration, New Orleans and the four largest hosts
for code in ["26420", "19100", "12060", "12940"]:
    v = [A.loc[code, f"dm{y}"] / 1e3 for y in yy]
    ax[0].plot(yy, v, color=fs.ALT, lw=0.8)
    fs.label(ax[0], 2009.25, {"26420": 72, "19100": 48, "12060": 24, "12940": 0}[code], A.loc[code, "name"].split("-")[0].split(",")[0], color="0.35")
v = [A.loc[NOLA, f"dm{y}"] / 1e3 for y in yy]
ax[0].plot(yy, v, color=fs.OBS, lw=1.1, marker="o", ms=2.5)
fs.label(ax[0], 2006.15, -332, "New Orleans")
ax[0].axhline(0, color="0.7", lw=0.5, zorder=0); ax[0].set_ylim(-350, 120); ax[0].set_xlim(2001.6, 2010.6)
ax[0].set_xticks(range(2002, 2010, 2)); ax[0].set_xlabel("year ending 30 June"); ax[0].set_ylabel("net domestic migration (thousands)")
# b: population and housing relative to trend
yrs = np.array(ns["years"]) + 0.5
ax[1].plot(np.r_[2005.5, yrs], np.r_[1, ns["hu_rel"]], color=fs.CAP, lw=0.8, marker="s", ms=2.8, mfc="white", mec=fs.CAP)
ax[1].plot(np.r_[2005.5, yrs], np.r_[1, ns["pop_rel"]], color=fs.OBS, lw=1.1, marker="o", ms=2.8)
fs.label(ax[1], 2009.6, ns["hu_rel"][-1], "housing units", color=fs.CAP); fs.label(ax[1], 2009.6, ns["pop_rel"][-1], "population")
fs.label(ax[1], 2005.74, 0.62, "landfall", color="0.4", rotation=90, va="bottom")
ax[1].axvline(2005.66, color="0.7", lw=0.5, ls=":", zorder=0)
ax[1].set_xlim(2005, 2010.9); ax[1].set_xticks(range(2005, 2011)); ax[1].set_ylim(0.6, 1.05)
ax[1].set_xlabel("year (1 July)"); ax[1].set_ylabel("New Orleans metro / 2001–05 trend")
# c: distance profile
order = np.argsort(cc.dist.values); dist = cc.dist.values[order]
sty = {"baseline out": dict(color=fs.ALT, ls="--", lw=0.9), "excess out 2005-06": dict(color=fs.OBS, lw=1.1), "return in 2006-07": dict(color=fs.MOD, lw=1.0)}
names = {"baseline out": "normal out-migration, 2003–05", "excess out 2005-06": "excess out-migration, 2005–06", "return in 2006-07": "excess in-migration, 2006–07"}
for k, w in prof.items():
    v = w.values[order]; ax[2].plot(dist, np.cumsum(v) / v.sum(), label=names[k], **sty[k])
ax[2].set_xscale("log"); ax[2].set_xlim(50, 5000); ax[2].set_ylim(0, 1)
ax[2].xaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
ax[2].set_xlabel("distance from New Orleans (km)"); ax[2].set_ylabel("cumulative share of migrants"); ax[2].legend(loc="upper left")
# d: destination shares
sb, se = h.o_base / h.o_base.sum(), h.excess / h.excess.sum()
idx = se.sort_values(ascending=False).index[:10][::-1]; yp = np.arange(len(idx))
for k, i in enumerate(idx): ax[3].plot([sb[i], se[i]], [k, k], color="0.8", lw=0.8, zorder=0)
ax[3].plot(sb[idx], yp, "o", mfc="white", mec=fs.OBS, mew=0.7, ms=3.5, label="normal out-migration, 2003–05")
ax[3].plot(se[idx], yp, "o", color=fs.MOD, ms=3.5, label="excess out-migration, 2005–06")
ax[3].set_yticks(yp, [h.name[i].split("-")[0].split(",")[0] for i in idx]); ax[3].tick_params(axis="y", length=0)
ax[3].spines["left"].set_visible(False); ax[3].set_xlim(0, 0.3); ax[3].set_xlabel("share of metro-bound migrants")
ax[3].legend(loc="lower right")
fig.tight_layout(w_pad=2.5, h_pad=1.8); fs.letters(ax[:3], "abc", dx=-0.13); fs.letters([ax[3]], "d", dx=-0.28)
fig.savefig(OUT / "fig_data.png"); fig.savefig(OUT / "fig_data.pdf")
