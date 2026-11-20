"""Fig. 1: model illustration drawn from the model itself (calibrated Katrina rates).
(a) struck city and hosts: capacity loss, evacuation down the occupancy gradient, return of the displaced;
(b) struck place over time: capacity, population and excess occupancy (severity);
(c) same shock with different tolerated occupancy phi* (local absorption vs abandonment)."""
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch
from scipy.integrate import solve_ivp
from common import OUT
import figstyle as fs

fs.apply()
B = json.load(open(OUT / "calibration.json"))["B"]
S, KAP, LAM = B["S_s"], B["kappa"], B["lam"]
D0, DINF, RHO = 0.42, 0.11, 0.76                     # stage-A capacity path (rounded)
K = lambda t: np.where(t < 0, 1.0, 1 - DINF - (D0 - DINF) * np.exp(-RHO * np.clip(t, 0, None)))


def run(phis, T=6.0):
    def rhs(t, y):
        N, D = y; phi = N / float(K(t))
        out = S * N * max(phi - phis, 0); ret = KAP * max(phis - phi, 0) * D
        return [-out + ret, out - ret - LAM * D]
    t = np.linspace(0, T, 600)
    s = solve_ivp(rhs, (0, T), [1.0, 0.0], t_eval=t, method="LSODA", rtol=1e-8, atol=1e-10, max_step=0.01)
    return t, s.y[0], s.y[1]


fig = plt.figure(figsize=(fs.W, 2.45))
gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 1, 1], wspace=0.38)
ax0, ax1, ax2 = fig.add_subplot(gs[0]), fig.add_subplot(gs[1]), fig.add_subplot(gs[2])

# ---- (a) network sketch
ax0.set_xlim(-1.3, 1.3); ax0.set_ylim(-1.1, 1.1); ax0.set_aspect("equal"); ax0.axis("off")
sx, sy = -0.78, 0.0
ax0.add_patch(Circle((sx, sy), 0.38, fill=False, ls=(0, (2, 1.5)), lw=0.8, ec=fs.CAP))          # capacity before
ax0.add_patch(Circle((sx, sy), 0.38 * np.sqrt(1 - D0), color=fs.CAP, alpha=0.25, lw=0))        # capacity after
ax0.add_patch(Circle((sx, sy), 0.38 * np.sqrt(0.8), fill=False, lw=1.0, ec=fs.MOD))            # population
hosts = [(0.80, 0.80, 0.17), (0.95, 0.22, 0.24), (0.88, -0.42, 0.19), (0.70, -0.92, 0.12)]
w = [0.8, 1.6, 1.0, 0.4]
for (hx, hy, r), wi in zip(hosts, w):
    ax0.add_patch(Circle((hx, hy), r, color=fs.CAP, alpha=0.25, lw=0))
    ax0.add_patch(Circle((hx, hy), r, fill=False, lw=0.8, ec=fs.MOD))
    v = np.array([hx - sx, hy - sy]); u = v / np.linalg.norm(v)
    a = (sx, sy) + u * 0.42; b = (hx, hy) - u * (r + 0.05)
    ax0.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=5, lw=0.4 + 0.8 * wi, color=fs.MOD,
                                  connectionstyle="arc3,rad=0.15", shrinkA=0, shrinkB=0))
    ax0.add_patch(FancyArrowPatch(b, a, arrowstyle="-|>", mutation_scale=4, lw=0.5, color=fs.MOD, ls=(0, (2, 1.5)),
                                  connectionstyle="arc3,rad=0.15", shrinkA=0, shrinkB=0))
ax0.text(sx, sy + 0.47, "struck city", ha="center", va="bottom", fontsize=6.5)
ax0.text(sx, sy - 0.47, r"$N_s>K_s$", ha="center", va="top", fontsize=6.5)
ax0.text(0.95, 1.02, "hosts", ha="center", va="bottom", fontsize=6.5)
ax0.text(-0.05, 0.62, "evacuation", fontsize=6, ha="center", color=fs.MOD, rotation=28)
ax0.text(-0.02, -0.60, "return", fontsize=6, ha="center", color=fs.MOD, rotation=-30)

# ---- (b) one struck place over time
t, N, Dd = run(1.0)
tt = np.r_[-0.6, 0, t]; KK = np.r_[1, 1, K(t)]; NN = np.r_[1, 1, N]
ax1.fill_between(t, K(t), N, where=N > K(t), color=fs.CAP, alpha=0.18, lw=0)
ax1.plot(tt, KK, color=fs.CAP, lw=1.0, ls="--")
ax1.plot(tt, NN, color=fs.MOD, lw=1.2)
fs.label(ax1, 3.0, float(K(3.0)) + 0.035, r"capacity $K_s$", color=fs.CAP, ha="left")
fs.label(ax1, 3.0, float(np.interp(3.0, t, N)) - 0.04, r"population $N_s$", color=fs.MOD, ha="left")
ax1.annotate("crowding", (0.32, 0.705), xytext=(0.9, 0.6), fontsize=6, ha="left", va="center",
             arrowprops=dict(arrowstyle="-", lw=0.5, color="0.45", shrinkA=1, shrinkB=0))
ax1.set_xlim(-0.6, 6); ax1.set_ylim(0.5, 1.05); ax1.set_xticks([0, 2, 4, 6])
ax1.set_xlabel("years after the disaster"); ax1.set_ylabel("relative to pre-disaster size")

# ---- (c) tolerated occupancy
for phis, ls, lab, dy in [(1.2, (0, (5, 1.5)), r"absorption, $\phi^*=1.2$", 0.03), (1.0, "-", r"$\phi^*=1$", 0.03),
                         (0.9, (0, (1, 1.2)), r"abandonment, $\phi^*=0.9$", -0.035)]:
    t, N, _ = run(phis)
    ax2.plot(np.r_[-0.6, 0, t], np.r_[1, 1, N], color=fs.MOD, lw=1.1, ls=ls)
    fs.label(ax2, 2.2, float(np.interp(2.2, t, N)) + dy, lab, color=fs.MOD, ha="left")
ax2.plot(np.r_[-0.6, 0, t], np.r_[1, 1, K(t)], color=fs.CAP, lw=1.0, ls="--")
ax2.set_xlim(-0.6, 6); ax2.set_ylim(0.5, 1.05); ax2.set_xticks([0, 2, 4, 6])
ax2.set_xlabel("years after the disaster")

fs.letters([ax0], "a", dx=0.04, dy=1.0); fs.letters([ax1, ax2], "bc", dx=-0.2)
fig.savefig(OUT / "fig_schematic.png"); fig.savefig(OUT / "fig_schematic.pdf")
print("schematic written")
