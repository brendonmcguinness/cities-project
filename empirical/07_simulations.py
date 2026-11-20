"""Section 4 rerun on the calibrated model: what determines how badly a metro fares when it is struck?

Each of the network's metros is struck in turn by a Katrina-calibrated shock (capacity path of Stage A, behavioural
rates of Stage B, host response of Stage C, S_med = 0), with the empirical migration network supplying destinations.
Severity is measured as excess-occupancy person-years per resident of the struck city: the integral over time of
people living above capacity (N_i - K_i)_+, summed over the struck city and all hosts. Unlike crowding loss, this is
observable in principle (persons per housing unit) and does not scale with the growth rate r.
Two scenarios for the struck city's evacuation rate:
  uniform   S_i = S_s (New Orleans' calibrated rate) for every city
  intensity S_i = S_s * g_i / g_NOLA, where g is the pre-disaster per-capita out-migration rate (an assumption: cities
            whose residents normally move more also evacuate faster)
Also: severity against shock size delta and host capacity response rho_h for New Orleans (replaces report Fig. 3).
Writes output/simulations.json and output/simulations_metros.csv."""
import json
import numpy as np
import pandas as pd
from common import OUT, NOLA
from model import Network, simulate, unpack

C = json.load(open(OUT / "calibration.json")); A, B, Cc = C["A"], C["B"], C["C"]
net = Network(); n = net.n; te = np.linspace(0, 6.0, 601)
w_emp = net.empirical(); np.fill_diagonal(w_emp, 0)
g = net.flow_base.sum(1) / net.M.p2005.values                    # pre-disaster out-migration per resident (metro-to-metro)
g_nola = g[net.s]


def W_for(s, S_s):
    W = np.zeros((n, n)); W[s, :] = w_emp[s] / w_emp[s].sum() * S_s
    return W


def severity(s, S_s, d0=A["d0"], dinf=A["dinf"], rho_s=A["rho_s"], rho_h=Cc["rho_h"]):
    u = unpack(simulate(net, W_for(s, S_s), d0, dinf, rho_s, rho_h, T=te[-1], t_eval=te, s=s, kappa=B["kappa"], lam=B["lam"]), n)
    over = np.maximum(u["N"] - u["K"], 0)
    ex = np.trapezoid(over, te, axis=1)                          # person-years (millions) per city
    K0s = net.K0[s]
    return dict(total=float(ex.sum() / K0s), struck=float(ex[s] / K0s), hosts=float((ex.sum() - ex[s]) / K0s),
                displaced_peak=float(u["D"].sum(0).max() / K0s), pop_2yr=float(u["N"][s, np.searchsorted(te, 2.0)] / K0s))


rows = []
for i, c in enumerate(net.codes):
    for scen, S in [("uniform", B["S_s"]), ("intensity", B["S_s"] * g[i] / g_nola)]:
        r = severity(i, S)
        rows.append(dict(code=c, name=net.M.iloc[i]["name"], pop2005=int(net.M.iloc[i].p2005), scenario=scen, S_i=S, g=g[i], **r))
df = pd.DataFrame(rows); df.to_csv(OUT / "simulations_metros.csv", index=False)

res = {}
for scen in ["uniform", "intensity"]:
    d = df[df.scenario == scen]
    k = np.polyfit(np.log(d.S_i), np.log(d.total), 1) if scen == "intensity" else [np.nan, np.nan]
    res[scen] = dict(severity_median=float(d.total.median()), severity_range=[float(d.total.min()), float(d.total.max())],
                     host_share_median=float((d.hosts / d.total).median()), host_share_max=float((d.hosts / d.total).max()),
                     slope_log_severity_vs_log_S=float(k[0]),
                     corr_log_severity_log_pop=float(np.corrcoef(np.log(d.total), np.log(d.pop2005))[0, 1]),
                     nola=d[d.code == NOLA][["total", "struck", "hosts", "displaced_peak", "pop_2yr"]].iloc[0].to_dict(),
                     highest=d.nlargest(5, "hosts")[["name", "hosts", "total"]].values.tolist())

# sweeps for New Orleans (report Fig. 3 analogue): shock size and host capacity response
s = net.s; sweeps = {"delta": [], "rho_h": []}
for f in np.linspace(0.1, 1.6, 9):                                # scale the initial capacity loss
    d0 = min(A["d0"] * f, 0.95)
    r = severity(s, B["S_s"], d0=d0, dinf=min(A["dinf"] * f, d0)); sweeps["delta"].append(dict(d0=d0, **r))
for rh in [0.05, 0.1, 0.3, 1.0, 3.0, 10.0]:
    r = severity(s, B["S_s"], rho_h=rh); sweeps["rho_h"].append(dict(rho_h=rh, **r))
res["sweeps_nola"] = sweeps
json.dump(res, open(OUT / "simulations.json", "w"), indent=1, default=float)
for scen in ["uniform", "intensity"]:
    print(scen, {k: v for k, v in res[scen].items() if k != "highest"}); print("   highest host burden:", res[scen]["highest"])
print("delta sweep:", [(round(x["d0"], 2), round(x["total"], 4), round(x["hosts"], 5)) for x in sweeps["delta"]])
print("rho_h sweep:", [(x["rho_h"], round(x["total"], 4), round(x["hosts"], 5)) for x in sweeps["rho_h"]])
