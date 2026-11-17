"""Metapopulation urban model on the 196-metro network, parameterised from data.

Equations (time in years since landfall, populations in millions):
    dN_i/dt = r_i N_i (1 - phi_i) + sum_j [F_ji - F_ij],   F_ij = W_ij N_i (phi_i - phi_j)_+,   phi = N/K
    dK_s/dt = rho_s (K_s(0) - K_s)            struck city rebuilds toward its pre-disaster capacity
    dK_i/dt = rho_h (N_i - K_i)_+             hosts expand capacity when crowded
    K_s(0+) = (1 - delta) K_s(0)              the disaster

Gravity network: w_ij = K_i^alpha K_j^beta / d_ij^gamma (great-circle km). Edges that touch the struck city are
scaled so its out-degree is S_s; all other edges so that the median host out-degree is S_med.
Compared with the CSSS report model (src/, network/): separate evacuation and host couplings, rebuilding of the
struck city (so return migration emerges), crowding-driven host capacity, and data-driven parameters.
"""
import numpy as np
from scipy.integrate import solve_ivp
from common import metros, hav, NOLA

M = metros()
S_IDX = int(np.where(M.code == NOLA)[0][0])
K0 = M.p2005.values / 1e6
R_NAT = M.natinc_rate_0205.clip(lower=0).values          # natural increase, births minus deaths per year
DIST = hav(M.lat.values[:, None], M.lon.values[:, None], M.lat.values[None, :], M.lon.values[None, :])
np.fill_diagonal(DIST, np.inf)
N_CITIES = len(K0)


def build_W(S_s, S_med, alpha, beta, gamma, struck_edges=None):
    """struck_edges: optional weights for the struck city's edges (e.g. pre-disaster IRS flows); gravity if None."""
    w = K0[:, None] ** alpha * K0[None, :] ** beta / DIST ** gamma
    hosts = np.ones(N_CITIES, bool); hosts[S_IDX] = False
    wh = w.copy(); wh[S_IDX, :] = 0; wh[:, S_IDX] = 0
    W = wh * (S_med / np.median(wh[hosts].sum(1)))
    e = w[S_IDX].copy() if struck_edges is None else np.asarray(struck_edges, float).copy()
    e[S_IDX] = 0
    e = e / e.sum() * S_s
    W[S_IDX, :] = e; W[:, S_IDX] = e
    return W


def make_rhs(W, K0_, r, s_idx, rho_s, rho_h):
    """Right-hand side for any network (used by simulate and by the tests).
    State y = [N (n), K (n), cumulative out of s, cumulative into s, cumulative crowding loss]."""
    n = len(K0_)
    def rhs(t, y):
        N, K = y[:n], y[n:2 * n]
        phi = N / K
        F = W * N[:, None] * np.maximum(phi[:, None] - phi[None, :], 0)   # F[i, j]: flow i -> j
        growth = r * N * (1 - phi)
        dK = rho_h * np.maximum(N - K, 0)
        dK[s_idx] = rho_s * (K0_[s_idx] - K[s_idx])
        return np.concatenate([growth + F.sum(0) - F.sum(1), dK,
                               [F[s_idx].sum(), F[:, s_idx].sum(), np.maximum(-growth, 0).sum()]])
    return rhs


def simulate(W, delta, rho_s, rho_h, r=None, T=4.2, t_eval=None, K0_=None, s_idx=None, rtol=1e-6, atol=1e-9):
    K0_ = K0 if K0_ is None else np.asarray(K0_, float)
    s_idx = S_IDX if s_idx is None else s_idx
    r = (R_NAT if K0_ is K0 else np.zeros(len(K0_))) if r is None else np.asarray(r, float)
    K_init = K0_.copy(); K_init[s_idx] *= (1 - delta)
    y0 = np.concatenate([K0_, K_init, [0, 0, 0]])
    return solve_ivp(make_rhs(W, K0_, r, s_idx, rho_s, rho_h), (0, T), y0, t_eval=t_eval,
                     method="LSODA", rtol=rtol, atol=atol)


def unpack(sol):
    n = N_CITIES
    return dict(t=sol.t, N=sol.y[:n], K=sol.y[n:2 * n], cum_out=sol.y[2 * n], cum_in=sol.y[2 * n + 1], loss=sol.y[2 * n + 2])
