"""Sensitivity of calibrated parameters and validation metrics to analyst choices.
Each variant reruns 02 (gravity), the full calibration and the validation in a fresh process with different settings.
Writes output/sensitivity.json."""
import json, os, subprocess, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
VARIANTS = {
    "main": {},
    "network: metros >= 150k": {"CAL_MINPOP": "150000"},
    "network: metros >= 500k": {"CAL_MINPOP": "500000"},
    "network: radius 300 km": {"CAL_MAXKM": "300"},
    "network: radius 900 km": {"CAL_MAXKM": "900"},
    "baseline: 2004-05 only": {"CAL_BASE": "0405"},
    "trend window 2002-05": {"CAL_TREND": "2002,2005"},
    "trend window 2003-05": {"CAL_TREND": "2003,2005"},
    "host set: top 5": {"CAL_TOPK": "5"},
    "host set: top 20": {"CAL_TOPK": "20"},
    "host coupling S_med = 0.01": {"CAL_SMED": "0.01"},
    "host coupling S_med = 0.05": {"CAL_SMED": "0.05"},
    "network: gravity kernel": {"CAL_KIND": "gravity"},
}
CODE = r'''
import json, os, numpy as np
from common import OUT
from model import Network
from calib import calibrate, sim_from
import importlib
net = Network()
S_med = float(os.environ.get("CAL_SMED", "0")); kind = os.environ.get("CAL_KIND", "empirical")
r = calibrate(net, S_med=S_med, kind=kind, verbose=False)
json.dump(r, open(OUT / "calibration.json", "w"), default=float)
'''


def run(env, folder):
    folder.mkdir(parents=True, exist_ok=True)
    e = dict(os.environ, **env, CAL_OUT=str(folder))
    subprocess.run([sys.executable, "02_gravity.py"], cwd=HERE, env=e, check=True, capture_output=True)
    subprocess.run([sys.executable, "-c", CODE], cwd=HERE, env=e, check=True, capture_output=True)
    # validation needs an alternative-return entry; reuse main structure
    cal = json.load(open(folder / "calibration.json")); cal["alternative_gradient_return"] = dict(cal)
    json.dump(cal, open(folder / "calibration.json", "w"), default=float)
    subprocess.run([sys.executable, "04_validate.py"], cwd=HERE, env=e, check=True, capture_output=True)
    cal = json.load(open(folder / "calibration.json")); val = json.load(open(folder / "validation.json"))
    v = val["tracked/empirical"] if env.get("CAL_KIND") != "gravity" else val["tracked/gravity"]
    return dict(n_metros=cal["n_metros"], d0=cal["A"]["d0"], dinf=cal["A"]["dinf"], rho_s=cal["A"]["rho_s"],
                S_s=cal["B"]["S_s"], kappa=cal["B"]["kappa"], lam=cal["B"]["lam"], rho_h=cal["C"]["rho_h"],
                pop_sse=cal["B"]["sse_pop"], dest_R2=v["V1_dest_R2"], return_R2=v["V2_return_R2"],
                return_ratio=v["V3_return_ratio"][1], dm_corr=v["V4_dm_corr"], dm_slope=v["V4_dm_slope"], occ_corr=v["V5_occ_corr"])


if __name__ == "__main__":
    out = {}
    for i, (name, env) in enumerate(VARIANTS.items()):
        out[name] = run(env, HERE / "output" / "sensitivity" / f"v{i:02d}")
        print(f"{name:30s} " + " ".join(f"{k}={v:.3g}" for k, v in out[name].items()), flush=True)
    json.dump(out, open(HERE / "output" / "sensitivity.json", "w"), indent=1)
