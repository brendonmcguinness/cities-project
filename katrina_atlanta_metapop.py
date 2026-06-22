"""
Metapopulation model of displacement after a natural disaster:
New Orleans (Katrina, patch K) <-> Atlanta (receiving city, patch A).

Each patch grows logistically toward its own (time-varying) carrying capacity.
A shock is a sharp drop in carrying capacity. New Orleans is permanently
diminished; Atlanta is undamaged and can *build* capacity through effort u
after the shock. Migration is bidirectional, driven by occupancy phi = N/K:
the per-capita emigration rate is proportional to occupancy, so the emigration
flux out of patch i is

        E_i = m * phi_i * N_i = m * N_i^2 / K_i.

Emigrants from K arrive in A and vice versa (mass-conserving).

Full system for t >= t_s (after the shock):

    dN_K/dt = r_K N_K (1 - N_K/K_K) - m N_K^2/K_K + m N_A^2/K_A
    dN_A/dt = r_A N_A (1 - N_A/K_A) - m N_A^2/K_A + m N_K^2/K_K
    dK_K/dt = 0                       (K_K held at (1-delta_K) K_K0 after shock)
    dK_A/dt = u (1 - K_A/K_A_max)     (effort-driven capacity building)

The shock at t_s is applied as an instantaneous reset of K_K.
"""

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt


# ----------------------------------------------------------------------
# Model
# ----------------------------------------------------------------------
def net_flux_KA(N_K, N_A, K_K, K_A, p):
    """Net migration flux from K to A (people / time). Positive => K -> A.

    Two switchable forms (p['migration_mode']):

      'gradient'  : direction set by the occupancy GAP. People leave whichever
                    city is more crowded, at a per-capita rate ~ |phi_K - phi_A|.
                    => if NOLA is over capacity and Atlanta is at it, flow goes
                       to Atlanta regardless of city sizes. (Recommended.)

      'occupancy' : Reading A. Each city emits flux m*N_i^2/K_i; net is the
                    difference. Direction can be dominated by the larger city.
    """
    phi_K = N_K / K_K
    phi_A = N_A / K_A
    if p["migration_mode"] == "gradient":
        gap = phi_K - phi_A
        src = N_K if gap > 0 else N_A      # source = the more-crowded city
        return p["m"] * src * gap
    else:  # 'occupancy'
        return p["m"] * (N_K * phi_K - N_A * phi_A)


def rhs(t, y, p):
    """Right-hand side. State y = [N_K, N_A, K_K, K_A]."""
    N_K, N_A, K_K, K_A = y

    phi_K = N_K / K_K
    phi_A = N_A / K_A

    F = net_flux_KA(N_K, N_A, K_K, K_A, p)   # net flow K -> A

    dN_K = p["r_K"] * N_K * (1 - phi_K) - F
    dN_A = p["r_A"] * N_A * (1 - phi_A) + F

    dK_K = 0.0  # New Orleans permanently lower: no recovery
    # Atlanta builds capacity only after the shock
    if t >= p["t_s"]:
        dK_A = p["u"] * (1 - K_A / p["K_A_max"])
    else:
        dK_A = 0.0

    return [dN_K, dN_A, dK_K, dK_A]


def simulate(p, T=120.0, n=2000):
    """Two-phase integration so the shock is a clean discontinuity at t_s."""
    # --- Phase 1: pre-shock, let the system settle to its joint equilibrium ---
    y0 = [p["N_K0"], p["N_A0"], p["K_K0"], p["K_A0"]]
    t1 = np.linspace(0.0, p["t_s"], int(n * p["t_s"] / T) + 2)
    s1 = solve_ivp(rhs, (0.0, p["t_s"]), y0, t_eval=t1,
                   args=(p,), method="LSODA", rtol=1e-8, atol=1e-10)

    # --- Apply the shock: sharp drop in K's carrying capacity ---
    y_shock = s1.y[:, -1].copy()
    y_shock[2] = (1 - p["delta_K"]) * p["K_K0"]   # K_K reset

    # --- Phase 2: post-shock dynamics ---
    t2 = np.linspace(p["t_s"], T, int(n * (T - p["t_s"]) / T) + 2)
    s2 = solve_ivp(rhs, (p["t_s"], T), y_shock, t_eval=t2,
                   args=(p,), method="LSODA", rtol=1e-8, atol=1e-10)

    t = np.concatenate([s1.t, s2.t])
    Y = np.concatenate([s1.y, s2.y], axis=1)
    return t, Y


# ----------------------------------------------------------------------
# Parameters (illustrative; in "hundreds of thousands of people" units)
# ----------------------------------------------------------------------
params = dict(
    migration_mode="gradient",  # 'gradient' (occupancy-gap) or 'occupancy' (Reading A)
    r_K=0.5, r_A=0.5,      # intrinsic growth rates (per year)
    m=0.8,                 # migration rate constant
    delta_K=0.6,           # 60% of New Orleans capacity destroyed
    u=4.0,                 # Atlanta build effort (capacity units / year)
    K_K0=12.0,             # New Orleans baseline capacity (~1.2M people)
    K_A0=40.0,             # Atlanta baseline capacity (~4.0M people)
    K_A_max=55.0,          # Atlanta capacity ceiling with effort
    t_s=20.0,              # shock time (Katrina)
    N_K0=12.0,             # initial populations ~ baseline capacities
    N_A0=40.0,
)


def metrics(t, Y, p):
    """Summary scalars used by the sweeps.

    Death accounting: migration conserves people, so the only thing that
    changes the total population is the logistic term G_i = r_i N_i(1-phi_i).
    A negative G_i (city over capacity) is net loss of people => 'deaths'.
    """
    N_K, N_A, K_K, K_A = Y
    total = N_K + N_A
    post = t >= p["t_s"]
    tp = t[post]

    flux = np.array([net_flux_KA(*Y[:, i], p) for i in range(Y.shape[1])])
    G_K = p["r_K"] * N_K * (1 - N_K / K_K)      # net growth (births - crowding deaths)
    G_A = p["r_A"] * N_A * (1 - N_A / K_A)
    deaths_K = -np.trapezoid(np.clip(G_K[post], None, 0), tp)   # over-capacity loss in K
    deaths_A = -np.trapezoid(np.clip(G_A[post], None, 0), tp)   # over-capacity loss in A

    T0 = total[~post][-1]                        # pre-shock total (reference)
    deficit = np.clip(T0 - total, 0, None)

    return dict(
        final_N_K=N_K[-1], final_N_A=N_A[-1],
        final_phi_K=N_K[-1] / K_K[-1], final_phi_A=N_A[-1] / K_A[-1],
        peak_flux=flux[post].max(),                 # size of the exodus wave
        min_total=total[post].min(),                # depth of the population dip
        dip_depth=T0 - total[post].min(),           # net lives lost at worst moment
        lost_person_time=np.trapezoid(deficit[post], tp),  # cumulative shortfall
        deaths_K=deaths_K, deaths_A=deaths_A,
        deaths_total=deaths_K + deaths_A,           # gross crowding mortality
        cum_displaced=np.trapezoid(np.clip(flux[post], 0, None), tp),
    )


# ----------------------------------------------------------------------
# Plot 1+2: single-run dynamics and time-varying parameters
# ----------------------------------------------------------------------
def plot_single(p):
    t, Y = simulate(p)
    N_K, N_A, K_K, K_A = Y
    total = N_K + N_A
    flux = np.array([net_flux_KA(*Y[:, i], p) for i in range(Y.shape[1])])
    ts = p["t_s"]

    fig, ax = plt.subplots(2, 2, figsize=(11, 7.5))
    ax[0, 0].plot(t, N_K, label="$N_K$ New Orleans", color="C3")
    ax[0, 0].plot(t, N_A, label="$N_A$ Atlanta", color="C0")
    ax[0, 0].plot(t, total, "--", label="total", color="0.5")
    ax[0, 0].set_title("Populations"); ax[0, 0].set_ylabel("population  $N$")

    ax[0, 1].plot(t, K_K, label="$K_K$ New Orleans", color="C3")
    ax[0, 1].plot(t, K_A, label="$K_A$ Atlanta", color="C0")
    ax[0, 1].set_title("Carrying capacities (shock + effort)")
    ax[0, 1].set_ylabel("carrying capacity  $K$")

    ax[1, 0].plot(t, N_K / K_K, label=r"$\phi_K$", color="C3")
    ax[1, 0].plot(t, N_A / K_A, label=r"$\phi_A$", color="C0")
    ax[1, 0].axhline(1.0, color="0.7", lw=0.8)
    ax[1, 0].set_title(r"Occupancy $\phi = N/K$")
    ax[1, 0].set_ylabel(r"occupancy  $\phi = N/K$")

    ax[1, 1].plot(t, flux, color="C2")
    ax[1, 1].axhline(0.0, color="0.7", lw=0.8)
    ax[1, 1].set_title("Net migration flux  K → A")
    ax[1, 1].set_ylabel("net flux  K → A  (people / time)")

    for a in ax.flat:
        a.axvline(ts, color="k", ls=":", lw=1)
        a.set_xlabel("time")
        if a.get_legend_handles_labels()[1]:
            a.legend()
    fig.suptitle(f"Katrina → Atlanta  (migration_mode='{p['migration_mode']}')")
    fig.tight_layout()
    fig.savefig("katrina_atlanta_dynamics.png", dpi=130)
    print("saved katrina_atlanta_dynamics.png")

    # --- Plot 2: the time-dependent parameters (capacities) + build effort ---
    fig2, ax2 = plt.subplots(1, 2, figsize=(11, 4))
    ax2[0].plot(t, K_K, label="$K_K(t)$ New Orleans", color="C3")
    ax2[0].plot(t, K_A, label="$K_A(t)$ Atlanta", color="C0")
    ax2[0].axhline(p["K_A_max"], color="C0", ls="--", lw=0.8, label="$K_A^{max}$")
    ax2[0].axvline(ts, color="k", ls=":", lw=1)
    ax2[0].set_title("Time-dependent carrying capacities")
    ax2[0].set_xlabel("time"); ax2[0].set_ylabel("carrying capacity  $K$"); ax2[0].legend()

    build = np.where(t >= ts, p["u"] * (1 - K_A / p["K_A_max"]), 0.0)
    ax2[1].plot(t, build, color="C4")
    ax2[1].axvline(ts, color="k", ls=":", lw=1)
    ax2[1].set_title(r"Atlanta build rate $\dot K_A = u(1-K_A/K_A^{max})$")
    ax2[1].set_xlabel("time"); ax2[1].set_ylabel(r"build rate  $\dot K_A$")
    fig2.tight_layout()
    fig2.savefig("katrina_atlanta_timeparams.png", dpi=130)
    print("saved katrina_atlanta_timeparams.png")

    print(f"pre-shock total = {total[t < ts][-1]:.3f}   final total = {total[-1]:.3f}")
    print(f"final N_K,N_A = {N_K[-1]:.3f},{N_A[-1]:.3f}  final phi_K = {N_K[-1]/K_K[-1]:.3f}")


# ----------------------------------------------------------------------
# Plot 3: sweep effort u  and  Plot 4: sweep shock severity delta_K
# ----------------------------------------------------------------------
def sweep_1d(key, values, base, xlabel, fname):
    rows = []
    for v in values:
        p = dict(base); p[key] = v
        t, Y = simulate(p)
        rows.append(metrics(t, Y, p))
    g = {k: np.array([r[k] for r in rows]) for k in rows[0]}

    fig, ax = plt.subplots(2, 2, figsize=(10, 7))
    ax[0, 0].plot(values, g["peak_flux"], "o-", color="C3")
    ax[0, 0].set_title("Peak exodus flux  K → A")
    ax[0, 0].set_ylabel("peak net flux  K → A")

    ax[0, 1].plot(values, g["final_N_A"], "o-", color="C0")
    ax[0, 1].set_title("Final Atlanta population $N_A$ (absorptive capacity)")
    ax[0, 1].set_ylabel("final  $N_A$")

    ax[1, 0].plot(values, g["min_total"], "o-", color="0.4")
    ax[1, 0].set_title("Min total population (depth of dip)")
    ax[1, 0].set_ylabel("min total population")

    ax[1, 1].plot(values, g["cum_displaced"], "o-", color="C2")
    ax[1, 1].set_title("Cumulative displaced  K → A")
    ax[1, 1].set_ylabel("cumulative displaced")

    for a in ax.flat:
        a.set_xlabel(xlabel)
    fig.tight_layout()
    fig.savefig(fname, dpi=130)
    print(f"saved {fname}")


# ----------------------------------------------------------------------
# Plot 5: 2D interaction — does more effort offset a worse shock?
# ----------------------------------------------------------------------
def sweep_2d(base):
    us = np.linspace(0.0, 10.0, 25)
    deltas = np.linspace(0.1, 0.9, 25)
    Z = np.zeros((len(deltas), len(us)))
    for i, d in enumerate(deltas):
        for j, u in enumerate(us):
            p = dict(base); p["delta_K"] = d; p["u"] = u
            t, Y = simulate(p, n=800)
            Z[i, j] = metrics(t, Y, p)["final_phi_K"]

    fig, ax = plt.subplots(figsize=(7, 5.5))
    im = ax.pcolormesh(us, deltas, Z, shading="auto", cmap="magma")
    cs = ax.contour(us, deltas, Z, levels=[1.0], colors="cyan")
    ax.clabel(cs, fmt=r"$\phi_K=1$")
    fig.colorbar(im, label=r"final NOLA occupancy $\phi_K$")
    ax.set_xlabel("Atlanta build effort  $u$")
    ax.set_ylabel(r"shock severity  $\delta_K$")
    ax.set_title("Chronic NOLA overcrowding vs. effort × shock")
    fig.tight_layout()
    fig.savefig("katrina_atlanta_sweep2d.png", dpi=130)
    print("saved katrina_atlanta_sweep2d.png")


def main():
    plot_single(params)
    sweep_1d("u", np.linspace(0.0, 10.0, 21), params,
             "Atlanta build effort  $u$", "katrina_atlanta_sweep_u.png")
    sweep_1d("delta_K", np.linspace(0.1, 0.9, 21), params,
             r"shock severity  $\delta_K$", "katrina_atlanta_sweep_delta.png")
    sweep_2d(params)


if __name__ == "__main__":
    main()
