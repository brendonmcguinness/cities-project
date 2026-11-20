"""Metapopulation urban model on the metro network, parameterised from data.

State: populations N_i, capacities K_i (time in years since landfall; people in millions).
    dN_i/dt = r_i N_i (1 - phi_i) + sum_j [F_ji - F_ij],     F_ij = W_ij N_i (phi_i - phi_j)_+,   phi = N/K
    K_s(t)  = K_s(0) [1 - d_inf - (d_0 - d_inf) exp(-rho_s t)]     struck city's capacity, fitted to Census housing units
    dK_i/dt = rho_h (N_i - K_i)_+                                    hosts add capacity when crowded
Network W (directed):
    baseline weights w_ij = observed pre-disaster IRS flow i->j (mean of 2003-04 and 2004-05), or, where the flow is
    suppressed, the PPML gravity expectation capped at the suppression threshold ("empirical" network); or pure
    gravity w_ij = K_i^a K_j^b d_ij^-g.
    Out-edges of the struck city s are scaled so that sum_j W_sj = S_s (evacuation rate per unit occupancy gap).
    In-edges j->s are scaled so that sum_j W_js K_j = S_ret K_s(0) (return rate of the struck city's capacity).
    All other edges are scaled so that the median host out-degree is S_med.
Differences from the CSSS report model (src/, network/): host building is demand-driven, not on a timer; rates are
per capita (unit-free); separate evacuation, return and host couplings instead of one max-normalised m0;
great-circle distances; the struck city's capacity follows measured housing units.
"""
import json
import numpy as np
from scipy.integrate import solve_ivp
from common import metros, metro_flows, hav, NOLA, BASE_YEARS, OUT, MIN_POP, MAX_KM


class Network:
    def __init__(self, min_pop=MIN_POP, max_km=MAX_KM, struck=NOLA):
        self.M = metros(min_pop, max_km)
        self.codes = self.M.index.tolist(); self.n = len(self.codes)
        self.s = self.codes.index(struck)
        self.K0 = self.M.p2005.values / 1e6
        self.r = self.M.natinc_rate_0205.clip(lower=0).values
        lat, lon = self.M.lat.values, self.M.lon.values
        self.D = hav(lat[:, None], lon[:, None], lat[None, :], lon[None, :]); np.fill_diagonal(self.D, np.inf)
        g = json.load(open(OUT / "gravity.json"))["network"]
        self.alpha, self.beta, self.gamma = g["alpha"], g["beta"], g["gamma"]
        idx = {c: i for i, c in enumerate(self.codes)}
        F = metro_flows().reset_index(); F = F[F.o.isin(idx) & F.d.isin(idx)]
        self.flow_base = np.zeros((self.n, self.n))
        for y in BASE_YEARS:
            for o, d, v in zip(F.o, F.d, F[y]):
                self.flow_base[idx[o], idx[d]] += v / len(BASE_YEARS)

    def gravity(self):
        P = self.M.p2005.values
        w = P[:, None] ** self.alpha * P[None, :] ** self.beta / self.D ** self.gamma
        return w * self.flow_base.sum() / w.sum()            # scaled to the total observed baseline flow

    SUPPRESSION_CAP = 20.0   # IRS suppresses flows below 10 returns (~20 exemptions); a filled value cannot exceed that

    def empirical(self):
        return np.where(self.flow_base > 0, self.flow_base, np.minimum(self.gravity(), self.SUPPRESSION_CAP))

    def W(self, S_s, S_ret, S_med, kind="empirical"):
        w = (self.empirical() if kind == "empirical" else self.gravity()).copy()
        np.fill_diagonal(w, 0); s, n = self.s, self.n
        hosts = np.ones(n, bool); hosts[s] = False
        wh = w.copy(); wh[s, :] = 0; wh[:, s] = 0
        W = wh * (S_med / np.median(wh[hosts].sum(1)))
        W[s, :] = w[s] / w[s].sum() * S_s
        col = w[:, s].copy(); col[s] = 0
        W[:, s] = col / (col @ self.K0) * S_ret * self.K0[s]
        return W


def capacity_struck(t, K0s, d0, dinf, rho_s):
    return K0s * (1 - dinf - (d0 - dinf) * np.exp(-rho_s * t))


def simulate(net, W, d0, dinf, rho_s, rho_h, T=4.0, t_eval=None, r=None, K0=None, s=None, kappa=None, lam=0.0,
             rtol=1e-6, atol=1e-9):
    """kappa=None: return migration is gradient-driven through the in-edges of W (any host resident can move to s).
    kappa given: displaced residents are tracked. D_j counts people displaced from s now living in j; they return at
    rate kappa * (1 - phi_s)_+ (when the struck city has room) and settle permanently at rate lam. In-edges of s in W
    are then ignored, so return flows come only from the displaced."""
    K0 = net.K0 if K0 is None else np.asarray(K0, float)
    s = net.s if s is None else s
    r = net.r if r is None else np.asarray(r, float)
    n = len(K0); track = kappa is not None
    Wm = W.copy()
    if track: Wm[:, s] = 0
    def rhs(t, y):
        N, K, D = y[:n], y[n:2 * n].copy(), y[2 * n:3 * n]
        K[s] = capacity_struck(t, K0[s], d0, dinf, rho_s)
        phi = N / K
        F = Wm * N[:, None] * np.maximum(phi[:, None] - phi[None, :], 0)
        growth = r * N * (1 - phi)
        dN = growth + F.sum(0) - F.sum(1)
        dD = np.zeros(n); ret = np.zeros(n)
        if track:
            ret = kappa * max(1 - phi[s], 0.0) * D
            dD = F[s] - ret - lam * D
            dN = dN - ret; dN[s] += ret.sum()
        dK = rho_h * np.maximum(N - K, 0)
        dK[s] = K0[s] * (d0 - dinf) * rho_s * np.exp(-rho_s * t)
        return np.concatenate([dN, dK, dD, [F[s].sum(), F[:, s].sum() + ret.sum(), np.maximum(-growth, 0).sum(),
                                            np.maximum(N - K, 0).sum()], ret])
    K_init = K0.copy(); K_init[s] = capacity_struck(0.0, K0[s], d0, dinf, rho_s)
    y0 = np.concatenate([K0, K_init, np.zeros(n), [0, 0, 0, 0], np.zeros(n)])
    return solve_ivp(rhs, (0, T), y0, t_eval=t_eval, method="LSODA", rtol=rtol, atol=atol)


def unpack(sol, n):
    y = sol.y
    return dict(t=sol.t, N=y[:n], K=y[n:2 * n], D=y[2 * n:3 * n], cum_out=y[3 * n], cum_in=y[3 * n + 1],
                loss=y[3 * n + 2], excess_occ=y[3 * n + 3], cum_ret=y[3 * n + 4:4 * n + 4])
