"""
Network metapopulation model: many cities on a graph respond to a shock.

This generalizes the two-patch New Orleans <-> Atlanta model to n cities.
Each city i grows logistically toward its own (time-varying) carrying capacity
K_i. People migrate along the edges of a weighted graph, flowing down the
occupancy gradient phi = N/K (from more-crowded to less-crowded cities). A shock
is a sharp drop in one city's capacity; chosen cities can rebuild capacity with
effort u_i.

For city i:

    dN_i/dt = r_i N_i (1 - N_i/K_i)
              + sum_j W_ij [ N_j (phi_j - phi_i)_+  -  N_i (phi_i - phi_j)_+ ]
    dK_i/dt = u_i (1 - K_i / K_i_max)   for t >= t_s + tau   (else 0)

The migration term conserves people on every edge (out of the crowded end,
into the other), so only the logistic term changes the total population --
the same accounting as the two-city model (see ../src). With n = 2 and a single
edge of weight m, this reduces exactly to the gradient-mode two-patch model.

Toy network: six Gulf / Southeast US metros, fully connected with gravity-model
weights W_ij ~ 1 / distance^gamma. The shock hits New Orleans.
"""

import os
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import networkx as nx

FIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
os.makedirs(FIG_DIR, exist_ok=True)


# ----------------------------------------------------------------------
# Toy network: six metros (approx lon/lat; capacities ~ metro pop / 100k)
# ----------------------------------------------------------------------
CITIES = {
    "New Orleans": dict(pos=(-90.07, 29.95), K=12.0),
    "Baton Rouge": dict(pos=(-91.19, 30.45), K=8.0),
    "Houston":     dict(pos=(-95.37, 29.76), K=70.0),
    "Atlanta":     dict(pos=(-84.39, 33.75), K=60.0),
    "Dallas":      dict(pos=(-96.80, 32.78), K=75.0),
    "Memphis":     dict(pos=(-90.05, 35.15), K=13.0),
}
NAMES = list(CITIES)
N_CITIES = len(NAMES)


def gravity_weights(gamma=2.0, scale=8.0):
    """Symmetric migration-ease matrix W_ij ~ scale / distance^gamma."""
    pos = np.array([CITIES[c]["pos"] for c in NAMES])
    W = np.zeros((N_CITIES, N_CITIES))
    for i in range(N_CITIES):
        for j in range(N_CITIES):
            if i != j:
                d = np.linalg.norm(pos[i] - pos[j])
                W[i, j] = scale / d ** gamma
    return W


def default_params():
    K0 = np.array([CITIES[c]["K"] for c in NAMES])
    u = np.zeros(N_CITIES)
    u[NAMES.index("Atlanta")] = 4.0          # Atlanta rebuilds/absorbs
    u[NAMES.index("Houston")] = 4.0          # Houston too (the other big hub)
    return dict(
        r=np.full(N_CITIES, 0.5),            # growth rates
        W=gravity_weights(),                 # migration-ease graph
        K0=K0,                               # baseline capacities
        K_max=K0 * 1.4,                      # ceiling when building
        u=u,                                 # per-city build effort
        shock_city="New Orleans",
        delta=0.6,                           # 60% capacity loss
        t_s=20.0, tau=0.0,                   # shock time, response delay
        N0=K0.copy(),                        # start at carrying capacity
    )


# ----------------------------------------------------------------------
# Model
# ----------------------------------------------------------------------
def rhs(t, y, p):
    N = y[:N_CITIES]
    K = y[N_CITIES:]
    phi = N / K

    D = phi[:, None] - phi[None, :]          # D_ij = phi_i - phi_j
    out = p["W"] * N[:, None] * np.maximum(D, 0.0)   # flow i -> j (i crowded)
    dN_migr = out.sum(axis=0) - out.sum(axis=1)      # inflow - outflow

    dN = p["r"] * N * (1 - phi) + dN_migr

    building = t >= p["t_s"] + p["tau"]
    dK = p["u"] * (1 - K / p["K_max"]) if building else np.zeros(N_CITIES)
    return np.concatenate([dN, dK])


def simulate(p, T=120.0, n=2000):
    y0 = np.concatenate([p["N0"], p["K0"]])
    t1 = np.linspace(0.0, p["t_s"], int(n * p["t_s"] / T) + 2)
    s1 = solve_ivp(rhs, (0.0, p["t_s"]), y0, t_eval=t1, args=(p,),
                   method="LSODA", rtol=1e-8, atol=1e-10)

    y_shock = s1.y[:, -1].copy()
    si = NAMES.index(p["shock_city"])
    y_shock[N_CITIES + si] = (1 - p["delta"]) * p["K0"][si]   # drop shocked city's K

    t2 = np.linspace(p["t_s"], T, int(n * (T - p["t_s"]) / T) + 2)
    s2 = solve_ivp(rhs, (p["t_s"], T), y_shock, t_eval=t2, args=(p,),
                   method="LSODA", rtol=1e-8, atol=1e-10)

    t = np.concatenate([s1.t, s2.t])
    Y = np.concatenate([s1.y, s2.y], axis=1)
    return t, Y


def split(Y):
    return Y[:N_CITIES], Y[N_CITIES:]        # N(t), K(t)  each (n_cities, n_t)


# ----------------------------------------------------------------------
# Plots
# ----------------------------------------------------------------------
def plot_timeseries(p):
    t, Y = simulate(p)
    N, K = split(Y)
    phi = N / K
    ts = p["t_s"]
    colors = plt.cm.tab10(np.arange(N_CITIES))

    fig, ax = plt.subplots(1, 3, figsize=(15, 4.6))
    for i, name in enumerate(NAMES):
        ax[0].plot(t, N[i], color=colors[i], label=name)
        ax[1].plot(t, phi[i], color=colors[i], label=name)
    ax[2].plot(t, N.sum(axis=0), color="0.3")

    ax[0].set_title("Populations"); ax[0].set_ylabel("population $N$")
    ax[1].set_title(r"Occupancy $\phi = N/K$"); ax[1].set_ylabel(r"$\phi$")
    ax[1].axhline(1.0, color="0.7", lw=0.8)
    ax[2].set_title("Total population (deaths = the dip)"); ax[2].set_ylabel("total $N$")
    for a in ax:
        a.axvline(ts, color="k", ls=":", lw=1); a.set_xlabel("time")
    ax[0].legend(fontsize=8)
    fig.suptitle("Network response to a shock at New Orleans")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "network_timeseries.png"), dpi=130)
    print("saved network_timeseries.png")


def plot_network_snapshots(p):
    """Node size = population, color = occupancy, at three times."""
    t, Y = simulate(p)
    N, K = split(Y)
    phi = N / K

    G = nx.Graph()
    for name in NAMES:
        G.add_node(name, pos=CITIES[name]["pos"])
    for i in range(N_CITIES):
        for j in range(i + 1, N_CITIES):
            if p["W"][i, j] > 0:
                G.add_edge(NAMES[i], NAMES[j], w=p["W"][i, j])
    pos = nx.get_node_attributes(G, "pos")
    wmax = max(d["w"] for *_, d in G.edges(data=True))

    # three snapshots: just before shock, peak overshoot, final
    pre = np.searchsorted(t, p["t_s"]) - 1
    peak = pre + int(np.argmax(phi[NAMES.index(p["shock_city"]), pre:]))
    idxs = [pre, peak, len(t) - 1]
    titles = ["pre-shock", "peak overshoot", "final"]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5.2))
    for ax, k, title in zip(axes, idxs, titles):
        sizes = 30 + 60 * N[:, k]
        node_phi = phi[:, k]
        for (a, b, d) in G.edges(data=True):
            xa, ya = pos[a]; xb, yb = pos[b]
            ax.plot([xa, xb], [ya, yb], "-", color="0.8",
                    lw=0.5 + 2.5 * d["w"] / wmax, zorder=1)
        sc = ax.scatter([pos[c][0] for c in NAMES], [pos[c][1] for c in NAMES],
                        s=sizes, c=node_phi, cmap="coolwarm", vmin=0.7, vmax=1.6,
                        edgecolors="k", zorder=2)
        for c in NAMES:
            ax.annotate(c, pos[c], fontsize=7, ha="center", va="bottom")
        ax.set_title(f"{title}  (t = {t[k]:.0f})")
        ax.set_xticks([]); ax.set_yticks([])
    fig.colorbar(sc, ax=axes, label=r"occupancy $\phi = N/K$",
                 fraction=0.025, pad=0.02)
    fig.suptitle("Shock propagation across the network (node size = population)")
    fig.savefig(os.path.join(FIG_DIR, "network_snapshots.png"), dpi=130)
    print("saved network_snapshots.png")


def report(p):
    t, Y = simulate(p)
    N, K = split(Y)
    total = N.sum(axis=0)
    pre = total[t < p["t_s"]][-1]
    print(f"pre-shock total = {pre:.2f}   final total = {total[-1]:.2f}   "
          f"min total = {total.min():.2f}")
    print("final populations:")
    for i, name in enumerate(NAMES):
        print(f"  {name:13s} N0={p['N0'][i]:6.2f}  N_final={N[i,-1]:6.2f}  "
              f"phi_final={N[i,-1]/K[i,-1]:.3f}")


def main():
    p = default_params()
    plot_timeseries(p)
    plot_network_snapshots(p)
    report(p)


if __name__ == "__main__":
    main()
