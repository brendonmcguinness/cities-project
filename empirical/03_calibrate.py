"""Calibrate the model (protocol in calib.py) and document the evidence for S_med = 0.
Writes output/calibration.json and output/onward_evidence.json."""
import json
import numpy as np
import statsmodels.api as sm
from common import OUT, NOLA, OTHER_STRUCK, BASE_YEARS, metros, metro_flows
from calib import calibrate

# ---- onward migration between hosts: do hosts that received many evacuees send more people on to other hosts?
m = metros(); codes = set(m.index); excl = {NOLA} | set(OTHER_STRUCK)
MF = metro_flows().reset_index(); MF = MF[MF.o.isin(codes) & MF.d.isin(codes)]
base = MF[BASE_YEARS].mean(1)
arr = (m.o0506 - m.o_base).clip(lower=0)
onward = {}
H = MF[~MF.o.isin(excl) & ~MF.d.isin(excl)]
for y in ["0506", "0607", "0708"]:
    x = H[y] - base[H.index]
    net_out = x.groupby(H.o).sum().reindex(m.index).fillna(0) - x.groupby(H.d).sum().reindex(m.index).fillna(0)
    d = m.drop(index=[c for c in excl if c in m.index])
    r = sm.OLS(net_out[d.index], sm.add_constant(arr[d.index])).fit(cov_type="HC1")
    onward[y] = dict(slope=float(r.params.iloc[1]), se=float(r.bse.iloc[1]),
                     ci95=[float(r.params.iloc[1] - 1.96 * r.bse.iloc[1]), float(r.params.iloc[1] + 1.96 * r.bse.iloc[1])], n=int(len(d)))
    print(f"onward {y}: net excess outflow to other hosts per evacuee received = {r.params.iloc[1]:.3f} ± {r.bse.iloc[1]:.3f}")
json.dump(onward, open(OUT / "onward_evidence.json", "w"), indent=1)

# ---- calibration
res = calibrate()                                       # main: displaced residents tracked
alt = calibrate(ret="gradient")                         # comparison: gradient-driven return, no tracking
res["alternative_gradient_return"] = alt
A, B, C = res["A"], res["B"], res["C"]
print(f"A: d0={A['d0']:.3f} dinf={A['dinf']:.3f} rho_s={A['rho_s']:.3f} (jackknife SE {np.round(A['se'], 3).tolist()}); resid {np.round(A['resid'], 4).tolist()}")
for lab, b in [("B (tracked)", B), ("B (gradient)", alt["B"])]:
    print(f"{lab}: " + "  ".join(f"{k}={b[k]:.3f} {np.round(ci, 3).tolist()}" for k, ci in zip(b["names"], b["ci95"]))
          + f"; pop model {np.round(b['pop_model'], 4).tolist()} obs {np.round(b['pop_obs'], 4).tolist()}; share {b['share_model']:.3f} vs {b['share_obs']:.3f}; SSE_pop={b['sse_pop']:.2e}")
print(f"C: rho_h={C['rho_h']:.3f} {np.round(C['ci95'], 3).tolist()}")
json.dump(res, open(OUT / "calibration.json", "w"), indent=1, default=float)
