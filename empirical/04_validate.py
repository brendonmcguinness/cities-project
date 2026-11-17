"""Out-of-sample validation of the calibrated model against data not used in calibration.
  V1 destinations of 2005-06 excess out-migration (IRS), shares across host metros
  V2 origins of 2006-07 excess return migration (IRS), shares
  V3 timing of excess return migration: IRS 2006-07 vs 2007-08 (ratio)
  V4 host net domestic migration in the year to June 2006 above its 2002-05 mean (Census), 62+ metros within 900 km
  V5 host occupancy (persons per housing unit relative to trend) in July 2006 and 2007, 10 main hosts (Census)
Consistency checks (not independent of calibration): NOLA occupancy 2006-09; NOLA net domestic migration 2006-09.
Static benchmarks for V1/V2: national gravity kernel, NOLA-specific gravity kernel (fitted to 2003-05 NOLA flows), and
the observed pre-disaster flow shares. Writes output/validation.json and output/validation_metros.csv."""
import json
import numpy as np
import pandas as pd
from common import OUT, NOLA, OTHER_STRUCK, APRIL, JULY, nola_series, nola_totals, LANDFALL
from model import Network, simulate, unpack
from calib import top_hosts, sim_from

C = json.load(open(OUT / "calibration.json")); G = json.load(open(OUT / "gravity.json"))
A = C["A"]
net = Network(); M = net.M; s = net.s
te = np.linspace(0, 4.0, 801)


def shares(x, mask):
    x = np.clip(np.asarray(x, float), 0, None) * mask; return x / x.sum()
def r2(o, p): return float(1 - ((o - p) ** 2).sum() / ((o - o.mean()) ** 2).sum())
def tv(o, p): return float(0.5 * np.abs(o - p).sum())


hostmask = np.ones(net.n); hostmask[s] = 0
mask_ns = hostmask.copy()
for c in OTHER_STRUCK:
    if c in net.codes: mask_ns[net.codes.index(c)] = 0
obs_dest = shares(M.excess.values, hostmask); obs_ret = shares(M.excess_ret.values, hostmask)
gN = G["nola_metro"]["o_base"]
static = {"national gravity": shares(net.gravity()[s], hostmask),
          "NOLA gravity (pre-disaster)": shares(M.p2005.values ** gN["beta"] * np.where(M.dist.values > 0, M.dist.values, np.inf) ** -gN["gamma"], hostmask),
          "pre-disaster flow shares": shares(M.o_base.values, hostmask)}
T = nola_totals(); in_base = T.loc[["0304", "0405"], "nola_in_total"].mean()
ret_ratio_obs = (T.loc["0607", "nola_in_total"] - in_base) / (T.loc["0708", "nola_in_total"] - in_base)
near = (M.dist.values < 900) & (mask_ns > 0)
hosts = top_hosts(net)

res = {"static_benchmarks": {k: dict(dest_R2=r2(obs_dest, v), dest_misalloc=tv(obs_dest, v)) for k, v in static.items()}}
rows = None
VARIANTS = {"tracked/empirical": (C, "empirical"), "tracked/gravity": (C, "gravity"),
            "gradient/empirical": (C["alternative_gradient_return"], "empirical")}
for kind, (cal, net_kind) in VARIANTS.items():
    B, Cc = cal["B"], cal["C"]
    u = unpack(sim_from(net, A, B, Cc["rho_h"], C["S_med"], net_kind, t_eval=te), net.n)
    t, N, K = u["t"], u["N"], u["K"]; phi = N / K
    if B["ret"] == "tracked":
        cum_ret = u["cum_ret"]
        def window(a, b): return np.interp(b, t, cum_ret.T).T - np.interp(a, t, cum_ret.T).T if False else \
            np.array([np.interp(b, t, cr) - np.interp(a, t, cr) for cr in cum_ret])
    else:
        W = net.W(B["S_s"], B["S_ret"], C["S_med"], net_kind)
        Fin = W[:, s][:, None] * N * np.maximum(phi - phi[s], 0)
        def window(a, b): k = (t >= a) & (t <= b); return np.trapezoid(Fin[:, k], t[k], axis=1)
    dest = shares(N[:, np.searchsorted(t, APRIL["0506"])] - net.K0, hostmask)
    ret = shares(window(APRIL["0506"], APRIL["0607"]), hostmask)
    ret_ratio_mod = window(APRIL["0506"], APRIL["0607"]).sum() / window(APRIL["0607"], APRIL["0708"]).sum()
    gain06 = (N[:, np.searchsorted(t, JULY[2006])] - net.K0) * 1e6
    dm = M.dm_excess2006.values
    occ = {y: (phi[:, np.searchsorted(t, JULY[y])] - 1) for y in (2006, 2007)}
    ns = nola_series()
    nola_occ = [float(phi[s, np.searchsorted(t, x)]) for x in ns["t"]]
    nola_dm_mod = []
    for y in (2006, 2007, 2008, 2009):
        a, b = max(y - 0.5 - LANDFALL, 0), y + 0.5 - LANDFALL
        ci, co = np.interp([a, b], t, u["cum_in"]), np.interp([a, b], t, u["cum_out"])
        nola_dm_mod.append(float(((ci[1] - ci[0]) - (co[1] - co[0])) * 1e6))
    nola_dm_obs = [float(M.iloc[s][f"dm{y}"] - M.iloc[s].dm_base) for y in (2006, 2007, 2008, 2009)]
    H = lambda v: float(v[net.codes.index("26420")])
    res[kind] = dict(
        V1_dest_R2=r2(obs_dest, dest), V1_dest_misalloc=tv(obs_dest, dest), V1_houston=[H(obs_dest), H(dest)],
        V2_return_R2=r2(obs_ret, ret), V2_return_misalloc=tv(obs_ret, ret), V2_houston=[H(obs_ret), H(ret)],
        V3_return_ratio=[float(ret_ratio_obs), float(ret_ratio_mod)],
        V4_dm_corr=float(np.corrcoef(dm[near], gain06[near])[0, 1]), V4_dm_slope=float(np.polyfit(gain06[near], dm[near], 1)[0]), V4_n=int(near.sum()),
        V5_occ=[dict(code=net.codes[i], name=M.iloc[i]["name"], obs2006=float(M.iloc[i].occ2006), mod2006=float(occ[2006][i]),
                     obs2007=float(M.iloc[i].occ2007), mod2007=float(occ[2007][i])) for i in hosts],
        V5_occ_corr=float(np.corrcoef(np.r_[M.iloc[hosts].occ2006, M.iloc[hosts].occ2007], np.r_[occ[2006][hosts], occ[2007][hosts]])[0, 1]),
        nola_occ_model=nola_occ, nola_occ_obs=[float(x) for x in ns["occ"]],
        nola_dm_model=nola_dm_mod, nola_dm_obs=nola_dm_obs,
        totals=dict(cum_out_M=float(u["cum_out"][-1]), cum_in_M=float(u["cum_in"][-1]), crowding_loss=float(u["loss"][-1] * 1e6),
                    excess_occupancy_person_years=float(u["excess_occ"][-1] * 1e6)))
    if kind == "tracked/empirical":
        rows = pd.DataFrame(dict(code=net.codes, name=M.name, dist=M.dist.round(0), obs_dest=obs_dest, mod_dest=dest, obs_ret=obs_ret, mod_ret=ret,
                                 dm_excess2006=dm.round(0), mod_gain06=gain06.round(0), near=near, occ2006=M.occ2006.values, mod_occ2006=occ[2006],
                                 occ2007=M.occ2007.values, mod_occ2007=occ[2007]))
        np.savez(OUT / "traj.npz", t=t, N_s=N[s], K_s=K[s], cum_in=u["cum_in"], cum_out=u["cum_out"], K0s=net.K0[s])
rows.to_csv(OUT / "validation_metros.csv", index=False)
json.dump(res, open(OUT / "validation.json", "w"), indent=1, default=float)
for k, v in res["static_benchmarks"].items(): print(f"static {k:28s} dest R2={v['dest_R2']:.2f} misalloc={v['dest_misalloc']:.3f}")
for kind in VARIANTS:
    v = res[kind]
    print(f"\n[{kind}] V1 dest R2={v['V1_dest_R2']:.2f} misalloc={v['V1_dest_misalloc']:.3f} Houston obs/mod={np.round(v['V1_houston'], 3).tolist()}")
    print(f"  V2 return R2={v['V2_return_R2']:.2f} misalloc={v['V2_return_misalloc']:.3f} Houston={np.round(v['V2_houston'], 3).tolist()}")
    print(f"  V3 return ratio 06-07/07-08 obs={v['V3_return_ratio'][0]:.2f} model={v['V3_return_ratio'][1]:.2f}")
    print(f"  V4 host dm corr={v['V4_dm_corr']:.2f} slope={v['V4_dm_slope']:.2f} (n={v['V4_n']})")
    print(f"  V5 host occupancy corr={v['V5_occ_corr']:.2f}:", [(o['name'][:12], round(o['obs2006'], 3), round(o['mod2006'], 3)) for o in v['V5_occ'][:4]])
    print(f"  NOLA occ model {np.round(v['nola_occ_model'], 3).tolist()} obs {np.round(v['nola_occ_obs'], 3).tolist()}")
    print(f"  NOLA dm model {np.round(np.array(v['nola_dm_model']) / 1e3, 1).tolist()}k obs {np.round(np.array(v['nola_dm_obs']) / 1e3, 1).tolist()}k")
    print(f"  totals {v['totals']}")
