"""Struck-place model shared by the event panels (10_events.py, 13_acs_panel.py).

With zero host-host coupling and large host areas (Section 'Ten disasters'), the network model reduces exactly to
the struck place (population N, capacity K(t)) and its displaced residents D living elsewhere:
    dN/dt = -S N (N/K - phi*)_+ + kappa (phi* - N/K)_+ D
    dD/dt =  S N (N/K - phi*)_+ - kappa (phi* - N/K)_+ D - lam D
phi* = 1 is the main model; phi* != 1 is the tolerated occupancy. N, K are relative to the pre-event trend (N0 = 1).
State also accumulates cumulative out- and in-migration for comparison with IRS filing years."""
import numpy as np
from scipy.integrate import solve_ivp


def simulate(K, S_s, kappa, lam, T, phis=1.0):
    def rhs(t, y):
        N, D, co, ci = y; phi = N / K(t)
        out = S_s * N * max(phi - phis, 0); ret = kappa * max(phis - phi, 0) * D
        return [-out + ret, out - ret - lam * D, out, ret]
    return solve_ivp(rhs, (0, T), [1.0, 0.0, 0.0, 0.0], method="LSODA", dense_output=True, rtol=1e-8, atol=1e-10, max_step=0.02)


def predictions(ev, K, S_s, kappa, lam, phis=1.0):
    """Population at observation times, IRS event-year share of two-year excess out-migration, second-year returns."""
    fy = ev.get("fy_end")
    T = max(ev["t_obs"][-1], fy[1] if fy is not None else 0) + 0.05
    s = simulate(K, S_s, kappa, lam, T, phis)
    pop = s.sol(ev["t_obs"])[0]
    share = ret = np.nan
    if fy is not None:
        co = s.sol(np.array([0.0, *fy]))[2]; ci = s.sol(np.array([fy[0], fy[1]]))[3]
        share = (co[1] - co[0]) / max(co[2] - co[0], 1e-12); ret = (ci[1] - ci[0]) * ev["P0"]
    return pop, share, ret


def capacity_piecewise(t_obs, k_rel):
    """Step at the event to the first post-event capacity estimate, linear between estimates, flat afterwards."""
    t_obs, k_rel = np.asarray(t_obs, float), np.asarray(k_rel, float)
    return lambda s: np.interp(s, np.r_[0, t_obs], np.r_[k_rel[0], k_rel], right=k_rel[-1])
