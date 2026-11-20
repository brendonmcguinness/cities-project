"""Fig. 9: tolerated occupancy against pre-disaster hazard geography (output/hazard_geography.json,
output/holdout_geo.json). (a) share of housing outside the mapped flood, surge and levee hazard; (b) share of housing
on ground at least 3 m above sea level (coastal storms); (c) held-out error with geography-predicted vs pooled phi*."""
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from common import OUT
import figstyle

figstyle.apply()
HALO = [pe.withStroke(linewidth=1.6, foreground="white")]
R = json.load(open(OUT / "hazard_geography.json")); HG = json.load(open(OUT / "holdout_geo.json"))
cen = {e["name"]: e for e in json.load(open(OUT / "events.json"))["events"]}
acs = json.load(open(OUT / "acs_panel.json"))["events"]


def ci(r):
    if r["panel"] == "census":
        lo, hi = cen[r["name"]]["B"]["phis_ci"]
    else:
        e = next(x for x in acs if x["fips"] == r["fips"] and x["disaster"] == r["disaster"]); s = e["B"]["phis_se"]
        lo, hi = e["B"]["phis"] - 1.96 * s, e["B"]["phis"] + 1.96 * s
    return max(lo, 0.65), min(hi, 1.45)


fig, axs = plt.subplots(1, 3, figsize=(figstyle.W, 2.35), gridspec_kw=dict(width_ratios=[1, 1, 1.05], wspace=0.42))
short = lambda r: r["name"].replace(" TX", "").replace(" FL", "").replace(" LA", "").replace(" NC", "").replace("East Baton Rouge", "E. Baton Rouge")
for ax, m, xl in ((axs[0], "outside_primary", "housing outside flood, surge\nand levee hazard (share)"),
                  (axs[1], "above3m", "housing on ground above 3 m (share)")):
    for r in R["events"]:
        if r.get(m) is None: continue
        kat = r.get("disaster") == "Katrina"; c = figstyle.MOD if kat else figstyle.OBS
        lo, hi = ci(r)
        if hi - lo < 0.6:                     # Galveston (Ike) is unidentified (interval 0.65-1.45); no bar drawn
            ax.plot([r[m], r[m]], [lo, hi], color=c, lw=0.6, alpha=0.45)
        ax.plot(r[m], r["phis"], "o", color=c, ms=3.2 if kat else 2.8, mfc=c if kat else "white", mew=0.8)
        nm = short(r)
        if nm in ("Orleans", "Jefferson", "Plaquemines"):
            continue
        dy = {("above3m", "Harrison"): 0.035, ("above3m", "Calcasieu"): -0.005, ("above3m", "Jackson"): -0.03, ("outside_primary", "Galveston"): -0.035}.get((m, nm), 0)
        dx = {("above3m", "Calcasieu"): -0.32, ("above3m", "Aransas"): -0.3}.get((m, nm), 0.02)
        if kat or nm in ("Cameron", "Monroe", "Galveston", "Robeson", "Calcasieu", "Aransas"):
            ax.text(r[m] + dx, r["phis"] + dy, nm, fontsize=5.6, va="center", path_effects=HALO, color="0.25")
    ax.text(0.06, 0.985, "Orleans, Jefferson,\nPlaquemines", fontsize=5.6, va="center", path_effects=HALO, color="0.25")
    t = R["tests"][m]
    ax.axhline(1, color="0.85", lw=0.5, zorder=0)
    ax.set_xlim(-0.04, 1.08); ax.set_ylim(0.65, 1.45); ax.set_xlabel(xl)
    pf = f"{t['p_spearman']:.3f}" if t['p_spearman'] < 0.01 else f"{t['p_spearman']:.2f}"
    ax.text(0.02, 0.97, f"Spearman {t['spearman']:.2f} (p = {pf}), n = {t['n']}", transform=ax.transAxes, fontsize=5.8, va="top", color="0.3")
axs[0].set_ylabel(r"tolerated occupancy $\phi^*$")
axs[0].plot([], [], "o", color=figstyle.MOD, ms=3.2, label="Katrina counties"); axs[0].plot([], [], "o", mfc="white", mec=figstyle.OBS, ms=2.8, label="other disasters")
axs[0].legend(loc="upper right", fontsize=5.8, bbox_to_anchor=(1.0, 0.9))

ax = axs[2]; rows = HG["primary"]["rows"]
for r in rows:
    ax.plot(100 * r["rmse"]["H"], 100 * r["rmse"]["G"], "o", ms=3, mfc=figstyle.MOD if r["g"] == "Katrina" else "white", mec=figstyle.MOD, mew=0.8)
ax.plot([0.5, 20], [0.5, 20], color="0.6", lw=0.6, ls="--")
ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(0.5, 20); ax.set_ylim(0.5, 20)
for a in (ax.xaxis, ax.yaxis): a.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:g}"))
s = HG["primary"]["summary"]
ax.set_xlabel(r"held-out error, pooled $\phi^*$ (pp)"); ax.set_ylabel(r"held-out error, $\phi^*$ from geography (pp)")
ax.text(0.04, 0.97, f"geography better in {s['G_beats_H']} of {s['n']}\nmedian {100 * s['G']['median']:.1f} vs {100 * s['H']['median']:.1f} pp",
        transform=ax.transAxes, fontsize=5.8, va="top", color="0.3")
figstyle.letters(axs, labels="abc")
fig.savefig(OUT / "fig_hazard.png"); fig.savefig(OUT / "fig_hazard.pdf")
