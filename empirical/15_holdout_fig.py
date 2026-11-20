"""Fig. 6: prediction for disasters not used in fitting (output/holdout.json from 14_holdout.py).
(a-h) held-out population paths for the eight counties with the largest housing losses;
(i) error of each predictor for all 21 held-out counties."""
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from common import OUT
import figstyle

figstyle.apply()
d = json.load(open(OUT / "holdout.json")); rows = d["rows"]
big = sorted(rows, key=lambda r: -r["loss"])[:8]
fig = plt.figure(figsize=(figstyle.W, 6.6))
gs = GridSpec(4, 4, figure=fig, width_ratios=[1, 1, 0.3, 1.55], hspace=0.62, wspace=0.42, top=0.93, bottom=0.07, left=0.08, right=0.98)
axs = [fig.add_subplot(gs[i // 2, i % 2]) for i in range(8)]
for a, r in zip(axs, big):
    t = np.array(r["t"])
    a.plot(np.r_[0, t], np.r_[r["Kpath"][0], r["Kpath"]], ":", color=figstyle.CAP, lw=1.0)
    a.plot(t, r["A"], "--", color=figstyle.ALT, lw=0.9)
    a.plot(t, r["H"], "-", color=figstyle.MOD, lw=1.1)
    a.plot(t, r["obs"], "o", color=figstyle.OBS, ms=2.6, zorder=5)
    a.axhline(1, color="0.85", lw=0.5, zorder=0)
    nm = r["name"].replace(" TX", "").replace(" FL", "").replace(" LA", "").replace(" CA", "")
    a.set_title(f"{nm}, {r['g']} ({100 * r['loss']:.0f}% lost)", fontsize=6.3, loc="left", pad=2)
    a.set_xlim(0, 4.6); a.set_xticks([0, 2, 4]); a.yaxis.set_major_locator(plt.MaxNLocator(4))
    v = np.r_[r["obs"], r["H"], r["A"], r["Kpath"], 1.0]; pad = 0.08 * (v.max() - v.min()); a.set_ylim(v.min() - 2 * pad, v.max() + pad)
for a in axs[6:]: a.set_xlabel("years after event")
for a in axs[0::2]: a.set_ylabel("population / trend")
axs[0].plot([], [], "o", color=figstyle.OBS, ms=2.6, label="Census estimate")
axs[0].plot([], [], "-", color=figstyle.MOD, label=r"predicted, pooled $\phi^*$")
axs[0].plot([], [], "--", color=figstyle.ALT, label=r"predicted, $\phi^*=1$")
axs[0].plot([], [], ":", color=figstyle.CAP, label="measured capacity")
fig.legend(*axs[0].get_legend_handles_labels(), loc="upper center", ncol=4, bbox_to_anchor=(0.5, 1.0))

ax = fig.add_subplot(gs[:, 3])
order = sorted(rows, key=lambda r: r["loss"])
y = np.arange(len(order))
for yi, r in zip(y, order):
    e = r["rmse"]; xs = [100 * e[m] for m in ("H", "A", "K", "none")]
    ax.plot([min(xs[:3]), max(xs[:3])], [yi, yi], color="0.85", lw=0.6, zorder=0)
    ax.plot(100 * e["none"], yi, "|", color="0.6", ms=4)
    ax.plot(100 * e["K"], yi, "o", mfc="white", mec=figstyle.CAP, ms=3, mew=0.8)
    ax.plot(100 * e["A"], yi, "s", mfc="white", mec=figstyle.ALT, ms=2.8, mew=0.8)
    ax.plot(100 * e["H"], yi, "o", color=figstyle.MOD, ms=3)
short = {"Tuscaloosa tornado": "tornado", "Louisiana floods": "floods", "Lahaina fire": "fire", "Tubbs fire": "fire"}
lab = [f"{r['name'].rsplit(' ', 1)[0] if r['src'] == 'acs' else r['name']} ({short.get(r['g'], r['g'])})" for r in order]
ax.set_yticks(y); ax.set_yticklabels(lab, fontsize=6); ax.set_xscale("log")
ax.set_xticks([1, 2, 5, 10, 20, 50]); ax.set_xticklabels(["1", "2", "5", "10", "20", "50"]); ax.set_xlim(0.6, 80)
ax.set_xlabel("held-out error (pp of population)", labelpad=1)
ax.plot([], [], "o", color=figstyle.MOD, ms=3, label=r"pooled $\phi^*$")
ax.plot([], [], "s", mfc="white", mec=figstyle.ALT, ms=2.8, label=r"$\phi^*=1$")
ax.plot([], [], "o", mfc="white", mec=figstyle.CAP, ms=3, label="population = capacity")
ax.plot([], [], "|", color="0.6", ms=4, label="no change")
ax.legend(loc="upper center", bbox_to_anchor=(0.45, -0.09), ncol=2, fontsize=5.8, handletextpad=0.3, columnspacing=0.8)
s = d["summary"]
ax.set_title(f"median {100 * s['H']['median']:.1f} / {100 * s['A']['median']:.1f} / {100 * s['K']['median']:.1f} / {100 * s['none']['median']:.1f} pp",
             fontsize=6, loc="left", pad=2, color="0.3")
figstyle.letters(axs + [ax], labels="abcdefghi")
fig.savefig(OUT / "fig_holdout.png"); fig.savefig(OUT / "fig_holdout.pdf")
