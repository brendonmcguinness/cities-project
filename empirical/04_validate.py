"""Out-of-sample checks of the calibrated model against data it was not fitted to:
  (1) destinations of 2005-06 excess out-migration (IRS)       [stage 2 used these: in-sample for S_med, rho_h only]
  (2) origins of 2006-07 excess return migration (IRS)
  (3) host-metro net domestic migration, year to June 2006, above its 2002-05 mean (Census components of change)
  (4) NOLA net domestic migration 2007-2009 above its 2002-05 mean (Census) vs model net inflow
Two versions of the struck city's edges: gravity kernel, and an empirical network proportional to 2004-05 IRS out-flows.
Writes output/validation.json, output/validation_<variant>.csv, output/traj_<variant>.npz."""
import json
import numpy as np
import pandas as pd
from common import OUT
from model import M, S_IDX, K0, N_CITIES, build_W, simulate, unpack

C = json.load(open(OUT / "calibration.json"))
te = np.linspace(0, 4.0, 801)
obs_dest = (M.o0506 - M.o0405).clip(lower=0).values.astype(float)
obs_ret = (M.i0607 - M.i0405).clip(lower=0).values.astype(float)
def shares(x): x = np.clip(np.asarray(x, float), 0, None).copy(); x[S_IDX] = 0; return x / x.sum()
def r2(o, p): return float(1 - ((o - p) ** 2).sum() / ((o - o.mean()) ** 2).sum())
def tv(o, p): return float(0.5 * np.abs(o - p).sum())
obs_dest, obs_ret = shares(obs_dest), shares(obs_ret)
dm_ex = M.dm_excess2006.values.astype(float)
nola_dm_ex = {y: float(M[f"dm{y}"][S_IDX] - M.dm_base[S_IDX]) for y in (2007, 2008, 2009)}
gulf = M.name.str.contains("Gulfport|Pascagoula").values
near = (M.dist.values < 900) & ~gulf; near[S_IDX] = False

base = M.o0405.values.astype(float); base[S_IDX] = 0
grav_row = build_W(1.0, 1.0, C["alpha"], C["beta"], C["gamma"])[S_IDX]
emp = np.where(base > 0, base, grav_row / grav_row.sum() * base.sum() * 0.05)

res = {}
for lab, edges in [("gravity", None), ("empirical", emp)]:
    W = build_W(C["S_s"], C["S_med"], C["alpha"], C["beta"], C["gamma"], struck_edges=edges)
    u = unpack(simulate(W, C["delta"], C["rho_s"], C["rho_h"], t_eval=te, T=4.0))
    t, N, K = u["t"], u["N"], u["K"]; phi = N / K
    Fin = W[:, S_IDX][:, None] * N * np.maximum(phi - phi[S_IDX], 0)            # flows into NOLA by origin
    win = (t >= 0.62) & (t <= 1.62)
    ret = shares(np.trapezoid(Fin[:, win], t[win], axis=1))
    dest = shares(N[:, np.searchsorted(t, 0.62)] - K0)
    gain06 = (N[:, np.searchsorted(t, 0.84)] - K0) * 1e6
    static = shares(W[S_IDX])
    # model net inflow to NOLA per Census year (July-June), from cumulative in/out flows
    net_in = {}
    for y in (2007, 2008, 2009):
        a, b = (y - 1) + 0.5 - 2005.66, y + 0.5 - 2005.66
        ci, co = np.interp([a, b], t, u["cum_in"]), np.interp([a, b], t, u["cum_out"])
        net_in[y] = float(((ci[1] - ci[0]) - (co[1] - co[0])) * 1e6)
    res[lab] = dict(dest_R2=r2(obs_dest, dest), dest_misalloc=tv(obs_dest, dest),
                    static_dest_R2=r2(obs_dest, static), static_dest_misalloc=tv(obs_dest, static),
                    return_R2=r2(obs_ret, ret), return_misalloc=tv(obs_ret, ret),
                    houston_dest=[float(obs_dest[M.name == "Houston TX"].item()), float(dest[M.name == "Houston TX"].item())],
                    houston_return=[float(obs_ret[M.name == "Houston TX"].item()), float(ret[M.name == "Houston TX"].item())],
                    dm_corr_near=float(np.corrcoef(dm_ex[near], gain06[near])[0, 1]), n_near=int(near.sum()),
                    dm_slope_near=float(np.polyfit(gain06[near], dm_ex[near], 1)[0]),
                    nola_net_in_model=net_in, nola_dm_excess_census=nola_dm_ex,
                    cum_out_M=float(u["cum_out"][-1]), cum_in_M=float(u["cum_in"][-1]),
                    crowding_loss=float(u["loss"][-1] * 1e6), n_rel_model=(N[S_IDX, [np.searchsorted(t, x) for x in (0.84, 1.84, 2.84, 3.84)]] / K0[S_IDX]).tolist())
    pd.DataFrame(dict(code=M.code, name=M.name, dist=M.dist.round(0), obs_dest=obs_dest, mod_dest=dest, obs_ret=obs_ret, mod_ret=ret,
                      dm_excess2006=dm_ex.round(0), mod_gain06=gain06.round(0), near=near)).to_csv(OUT / f"validation_{lab}.csv", index=False)
    np.savez(OUT / f"traj_{lab}.npz", t=t, N_s=N[S_IDX], K_s=K[S_IDX], cum_in=u["cum_in"], cum_out=u["cum_out"])
    print(lab, json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in res[lab].items()}))
json.dump(res, open(OUT / "validation.json", "w"), indent=1)
