"""Calibrate the model with every parameter taken from data.

Fixed from data before fitting:
  alpha, beta, gamma   joint PPML on pre-disaster (2004-05) IRS flows into and out of New Orleans   (02_gravity.py)
  r_i                  Census natural increase rate 2002-05 per metro (births minus deaths)
  K_i(0)               Census intercensal population, July 2005
Fitted:
  stage 1: delta, rho_s, S_s   to Census NOLA population 2006-09 (relative to 2001-05 trend) and the IRS timing of excess out-migration
  stage 2: S_med, rho_h        to the IRS destinations of 2005-06 excess out-migration (misallocated share), Nelder-Mead from the best grid point
Writes output/calibration.json.
"""
import json, itertools
import numpy as np
from scipy.optimize import minimize
from common import OUT, parishes, IRS_TOTAL_OUT
from model import M, S_IDX, K0, build_W, simulate, unpack

G = json.load(open(OUT / "gravity.json"))["joint_pre_disaster"]
A, B, GM = G["alpha"], G["beta"], G["gamma"]

# ---- targets
p = parishes()
P = {y: p[f"p{y}"].sum() for y in range(2001, 2010)}
trend = (P[2005] / P[2001]) ** 0.25 - 1
n_obs = np.array([P[y] / (P[2005] * (1 + trend) ** (y - 2005)) for y in (2006, 2007, 2008, 2009)])
t_july = np.array([0.84, 1.84, 2.84, 3.84])            # 1 July 2006..2009 in years since landfall
ex0506 = IRS_TOTAL_OUT["0506"] - IRS_TOTAL_OUT["0405"]
ex0607 = IRS_TOTAL_OUT["0607"] - IRS_TOTAL_OUT["0405"]
f_obs = ex0506 / (ex0506 + ex0607)                       # share of 2-yr excess out-migration in filing year 2005-06
T_APR06, T_APR07 = 0.62, 1.62                            # IRS filing-year boundaries in model time
obs_dest = (M.o0506 - M.o0405).clip(lower=0).values.astype(float); obs_dest[S_IDX] = 0; obs_dest /= obs_dest.sum()

# ---- stage 1
def stage1_loss(lp, S_med=0.01, rho_h=3.0):
    delta, rho_s, S_s = 1 / (1 + np.exp(-lp[0])), np.exp(lp[1]), np.exp(lp[2])
    sol = simulate(build_W(S_s, S_med, A, B, GM), delta, rho_s, rho_h, t_eval=np.r_[T_APR06, t_july[0], T_APR07, t_july[1:]])
    if not sol.success: return 1e3
    u = unpack(sol)
    n_mod = u["N"][S_IDX, [1, 3, 4, 5]] / K0[S_IDX]
    out = np.interp([T_APR06, T_APR07], u["t"], u["cum_out"])
    return ((n_mod - n_obs) ** 2).sum() / 0.005 ** 2 + ((out[0] / out[1] - f_obs) / 0.02) ** 2

r1 = minimize(stage1_loss, [np.log(0.3 / 0.7), np.log(0.12), np.log(4.0)], method="Nelder-Mead",
              options={"xatol": 1e-3, "fatol": 1e-3, "maxiter": 300})
delta, rho_s, S_s = 1 / (1 + np.exp(-r1.x[0])), np.exp(r1.x[1]), np.exp(r1.x[2])
print(f"stage 1: delta={delta:.3f} rho_s={rho_s:.3f} S_s={S_s:.2f} loss={r1.fun:.2f}")

# ---- stage 2
def dest_misalloc(S_med, rho_h):
    sol = simulate(build_W(S_s, S_med, A, B, GM), delta, rho_s, rho_h, t_eval=np.array([T_APR06, 0.84]), T=1.0)
    g = np.clip(sol.y[:len(K0), 0] - K0, 0, None); g[S_IDX] = 0; g /= g.sum()
    return 0.5 * np.abs(g - obs_dest).sum(), g

grid = []
for S_med, rho_h in itertools.product([0.003, 0.01, 0.03, 0.1, 0.3, 1.0], [0.3, 1, 3, 10, 30]):
    tv, g = dest_misalloc(S_med, rho_h)
    grid.append(dict(S_med=S_med, rho_h=rho_h, misalloc=tv, houston_share=float(g[M.name.str.startswith("Houston").values].item())))
best = min(grid, key=lambda r: r["misalloc"])
r2 = minimize(lambda lp: dest_misalloc(np.exp(lp[0]), np.exp(lp[1]))[0], [np.log(best["S_med"]), np.log(best["rho_h"])],
              method="Nelder-Mead", options={"xatol": 1e-2, "fatol": 1e-4, "maxiter": 80})
S_med, rho_h = float(np.exp(r2.x[0])), float(np.exp(r2.x[1]))
print(f"stage 2: S_med={S_med:.4f} rho_h={rho_h:.3f} misalloc={r2.fun:.4f} (grid best {best['misalloc']:.4f}); max Houston share on grid {max(g['houston_share'] for g in grid):.3f}")

# ---- refit stage 1 with stage-2 host parameters (consistency)
r1b = minimize(lambda lp: stage1_loss(lp, S_med, rho_h), r1.x, method="Nelder-Mead", options={"xatol": 1e-3, "fatol": 1e-3, "maxiter": 200})
delta, rho_s, S_s = 1 / (1 + np.exp(-r1b.x[0])), np.exp(r1b.x[1]), np.exp(r1b.x[2])
print(f"stage 1 (refit): delta={delta:.3f} rho_s={rho_s:.3f} S_s={S_s:.2f} loss={r1b.fun:.2f}")

json.dump(dict(alpha=A, beta=B, gamma=GM, delta=delta, rho_s=rho_s, S_s=S_s, S_med=S_med, rho_h=rho_h,
               trend=trend, n_obs=n_obs.tolist(), f_obs=f_obs, stage1_loss=r1b.fun, stage2_misalloc=r2.fun,
               grid=grid, r_nat_nola=float(M.natinc_rate_0205[S_IDX])), open(OUT / "calibration.json", "w"), indent=1)
