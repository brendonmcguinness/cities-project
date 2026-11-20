"""Calibration routines (used by 03_calibrate.py and 06_sensitivity.py).

Protocol: each parameter block is fitted to one data set; everything else is held out for validation.
  A  d_0, d_inf, rho_s   <- Census housing units of the NOLA metro, July 2006-09, relative to 2001-05 trend
  B  S_s, kappa, lam     <- Census population of the NOLA metro, July 2006-09, relative to trend,
                            and the IRS share of two-year excess out-migration falling in filing year 2005-06
                            (S_s evacuation rate; kappa return rate of displaced residents per unit spare capacity in
                            the struck city; lam rate at which displaced residents settle permanently in their host)
                            Alternative return mechanism (return="gradient"): S_s, S_ret with no tracking of the displaced.
  C  rho_h               <- Census housing units of the 10 metros receiving most evacuees (IRS), July 2006-07,
                            above their 2001-05 trend, in persons (x persons per unit in 2005)
  S_med (host-host coupling) is set to 0: IRS host-to-host flows show no onward migration (see 03_calibrate.py).
Noise levels are not assumed: each block's residual variance is estimated from its own fit, and parameter
standard errors come from the Gauss-Newton approximation  Cov = s^2 (J'J)^-1  on the fitted (log/logit) scale.
"""
import numpy as np
from scipy.optimize import least_squares
from common import nola_series, nola_totals, APRIL, OTHER_STRUCK, NOLA, BASE_YEARS, TOP_K
from model import Network, simulate, unpack, capacity_struck


def fit_A():
    ns = nola_series(); t = np.array(ns["t"]); y = np.array(ns["hu_rel"])
    def res(p):
        d0, dinf, rho = p
        return capacity_struck(t, 1.0, d0, dinf, rho) - y
    sol = least_squares(res, [0.4, 0.12, 0.6], bounds=([0, 0, 1e-3], [0.9, 0.5, 10]))
    # leave-one-year-out (jackknife) spread as uncertainty: 4 points, 3 parameters
    jk = []
    for k in range(len(t)):
        keep = np.arange(len(t)) != k
        r = least_squares(lambda p: (capacity_struck(t[keep], 1.0, *p) - y[keep]), sol.x, bounds=([0, 0, 1e-3], [0.9, 0.5, 10]))
        jk.append(r.x)
    jk = np.array(jk); n = len(t)
    se = np.sqrt((n - 1) / n * ((jk - jk.mean(0)) ** 2).sum(0))
    return dict(d0=sol.x[0], dinf=sol.x[1], rho_s=sol.x[2], se=se.tolist(), resid=sol.fun.tolist(), jackknife=jk.tolist())


def irs_first_share():
    T = nola_totals(); base = T.loc[BASE_YEARS, "nola_out_total"].mean()
    e1, e2 = T.loc["0506", "nola_out_total"] - base, T.loc["0607", "nola_out_total"] - base
    return e1 / (e1 + e2)


def top_hosts(net, k=None):
    k = TOP_K if k is None else k
    m = net.M.drop(index=[NOLA] + [c for c in OTHER_STRUCK if c in net.M.index])
    return [net.codes.index(c) for c in m.sort_values("excess", ascending=False).index[:k]]


def model_B(net, A, p, rho_h, S_med=0.0, kind="empirical", ret="tracked"):
    ns = nola_series()
    te = np.sort(np.r_[ns["t"], APRIL["0506"], APRIL["0607"]])
    if ret == "tracked":
        S_s, kappa, lam = p; W = net.W(S_s, 1.0, S_med, kind)
        sol = simulate(net, W, A["d0"], A["dinf"], A["rho_s"], rho_h, t_eval=te, kappa=kappa, lam=lam)
    else:
        S_s, S_ret = p; sol = simulate(net, net.W(S_s, S_ret, S_med, kind), A["d0"], A["dinf"], A["rho_s"], rho_h, t_eval=te)
    u = unpack(sol, net.n)
    pop = np.array([u["N"][net.s, np.searchsorted(te, t)] for t in ns["t"]]) / net.K0[net.s]
    out = np.interp([APRIL["0506"], APRIL["0607"]], u["t"], u["cum_out"])
    return pop, out[0] / out[1], u, te


def fit_B(net, A, rho_h, S_med=0.0, kind="empirical", ret="tracked", x0=None):
    ns = nola_series(); y = np.array(ns["pop_rel"]); f = irs_first_share()
    names = ["S_s", "kappa", "lam"] if ret == "tracked" else ["S_s", "S_ret"]
    x0 = x0 or ((1.6, 10.0, 0.5) if ret == "tracked" else (4.0, 1.0))
    def parts(lp):
        pop, share, _, _ = model_B(net, A, np.exp(lp), rho_h, S_med, kind, ret)
        return pop - y, np.array([share - f])
    # two residual groups with separately estimated variances (iterated once)
    sig = np.array([0.01, 0.02])
    for _ in range(2):
        sol = least_squares(lambda lp: np.concatenate([parts(lp)[0] / sig[0], parts(lp)[1] / sig[1]]), np.log(x0), diff_step=1e-3)
        rp, rf = parts(sol.x)
        sig = np.array([max(np.sqrt(np.mean(rp ** 2)), 1e-3), max(abs(rf[0]), 0.005)])
    J = sol.jac; cov = np.linalg.pinv(J.T @ J)
    se_log = np.sqrt(np.diag(cov))
    out = {k: float(np.exp(v)) for k, v in zip(names, sol.x)}
    out.update(names=names, ret=ret, se_log=se_log.tolist(), cov_log=cov.tolist(), x_log=sol.x.tolist(),
               ci95=[[float(np.exp(sol.x[i] - 1.96 * se_log[i])), float(np.exp(sol.x[i] + 1.96 * se_log[i]))] for i in range(len(names))],
               pop_model=(rp + y).tolist(), pop_obs=y.tolist(), share_model=float(rf[0] + f), share_obs=float(f), sigma=sig.tolist(),
               chi2=float(np.sum((rp / sig[0]) ** 2) + (rf[0] / sig[1]) ** 2), sse_pop=float(np.sum(rp ** 2)))
    return out


def host_targets(net, hosts):
    M = net.M
    return np.array([[M.iloc[i][f"hu_excess{y}"] * M.iloc[i].ppu2005 / 1e6 for y in (2006, 2007)] for i in hosts])


def sim_from(net, A, B, rho_h, S_med=0.0, kind="empirical", **kw):
    if B["ret"] == "tracked":
        return simulate(net, net.W(B["S_s"], 1.0, S_med, kind), A["d0"], A["dinf"], A["rho_s"], rho_h, kappa=B["kappa"], lam=B["lam"], **kw)
    return simulate(net, net.W(B["S_s"], B["S_ret"], S_med, kind), A["d0"], A["dinf"], A["rho_s"], rho_h, **kw)


def fit_C(net, A, B, S_med=0.0, kind="empirical", hosts=None):
    from common import JULY
    hosts = top_hosts(net) if hosts is None else hosts
    tgt = host_targets(net, hosts); te = np.array([JULY[2006], JULY[2007]])
    def res(lp):
        u = unpack(sim_from(net, A, B, np.exp(lp[0]), S_med, kind, t_eval=te), net.n)
        return ((u["K"][hosts] - net.K0[hosts, None]) - tgt).ravel()
    sol = least_squares(res, [np.log(1.0)], diff_step=1e-3, bounds=([np.log(1e-3)], [np.log(1e3)]))
    s2 = np.mean(sol.fun ** 2) * len(sol.fun) / max(len(sol.fun) - 1, 1)
    se_log = float(np.sqrt(s2 / (sol.jac.T @ sol.jac)[0, 0]))
    return dict(rho_h=float(np.exp(sol.x[0])), se_log=se_log, ci95=[float(np.exp(sol.x[0] - 1.96 * se_log)), float(np.exp(sol.x[0] + 1.96 * se_log))],
                hosts=[net.codes[i] for i in hosts], target_k=tgt.tolist(), model_k=(sol.fun.reshape(tgt.shape) + tgt).tolist())


def calibrate(net=None, S_med=0.0, kind="empirical", ret="tracked", verbose=True, A=None):
    net = Network() if net is None else net
    A = fit_A() if A is None else A
    rho_h, B = 1.0, None
    for it in range(4):                         # B and C interact weakly; iterate to a fixed point
        B = fit_B(net, A, rho_h, S_med, kind, ret, x0=None if B is None else tuple(B[k] for k in B["names"]))
        C = fit_C(net, A, B, S_med, kind)
        if verbose: print(f"  iter {it}: " + " ".join(f"{k}={B[k]:.3f}" for k in B["names"]) + f" rho_h={C['rho_h']:.3f}")
        if abs(np.log(C["rho_h"] / rho_h)) < 0.01: break
        rho_h = C["rho_h"]
    return dict(A=A, B=B, C=C, S_med=S_med, kind=kind, ret=ret, n_metros=net.n)
