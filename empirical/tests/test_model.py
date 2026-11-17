"""Correctness tests for empirical/model.py. Run from empirical/:  python3 -m pytest -q tests
They check properties that any correct implementation must have, independent of calibration."""
import json, sys
from pathlib import Path
import numpy as np
import pytest

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent / "src"))
from model import K0, S_IDX, N_CITIES, R_NAT, build_W, simulate, unpack   # noqa: E402

C = json.load(open(HERE / "output" / "calibration.json"))
W_CAL = build_W(C["S_s"], max(C["S_med"], 1e-6), C["alpha"], C["beta"], C["gamma"])


def test_migration_conserves_people():
    """With r = 0 only migration acts, so total population must stay constant."""
    u = unpack(simulate(W_CAL, 0.3, 0.12, 3.0, r=np.zeros(N_CITIES), T=4.0, t_eval=np.linspace(0, 4, 41)))
    tot = u["N"].sum(0)
    assert np.max(np.abs(tot - tot[0])) / tot[0] < 1e-6


def test_no_disaster_means_no_change():
    """delta = 0: every city stays at capacity, no flows, no losses (unlike the original code, where hosts build on a timer)."""
    u = unpack(simulate(W_CAL, 0.0, C["rho_s"], C["rho_h"], T=4.0, t_eval=np.linspace(0, 4, 41)))
    assert np.max(np.abs(u["N"] - K0[:, None])) < 1e-9
    assert np.max(np.abs(u["K"] - K0[:, None])) < 1e-9
    assert u["cum_out"][-1] < 1e-12 and u["loss"][-1] < 1e-12


def test_populations_and_capacities_stay_positive():
    u = unpack(simulate(W_CAL, 0.6, 0.0, 0.0, T=4.0, t_eval=np.linspace(0, 4, 81)))
    assert (u["N"] > 0).all() and (u["K"] > 0).all()


def test_flow_accounting_matches_population_change():
    """With r = 0, the struck city's population change equals cumulative inflow minus outflow."""
    u = unpack(simulate(W_CAL, 0.3, 0.12, 3.0, r=np.zeros(N_CITIES), T=4.0, t_eval=np.linspace(0, 4, 41)))
    dN = u["N"][S_IDX] - u["N"][S_IDX, 0]
    assert np.max(np.abs(dN - (u["cum_in"] - u["cum_out"]))) < 1e-6


def test_reduces_to_original_two_city_model():
    """Two cities, no capacity response, no rebuilding: identical to src/katrina_atlanta_metapop.py (gradient mode, u = 0)."""
    import matplotlib; matplotlib.use("Agg")
    import katrina_atlanta_metapop as src
    p = dict(src.params); p.update(u=0.0, t_s=1.0, migration_mode="gradient")
    t_ref, Y = src.simulate(p, T=31.0, n=3101)
    keep = t_ref > 1.0 + 1e-9; t_ref, Y = t_ref[keep] - 1.0, Y[:, keep]
    K0_ = np.array([p["K_K0"], p["K_A0"]])
    W = np.array([[0, p["m"]], [p["m"], 0]], float)
    sol = simulate(W, p["delta_K"], 0.0, 0.0, r=np.array([p["r_K"], p["r_A"]]), T=30.0, t_eval=t_ref,
                   K0_=K0_, s_idx=0, rtol=1e-9, atol=1e-11)
    assert np.max(np.abs(sol.y[:2] - Y[:2])) < 1e-5


def test_solver_converged():
    """Tightening tolerances by 1000x changes the struck city's trajectory by < 1e-4 (relative)."""
    te = np.linspace(0, 4, 41)
    a = simulate(W_CAL, C["delta"], C["rho_s"], C["rho_h"], T=4.0, t_eval=te).y[S_IDX]
    b = simulate(W_CAL, C["delta"], C["rho_s"], C["rho_h"], T=4.0, t_eval=te, rtol=1e-9, atol=1e-12).y[S_IDX]
    assert np.max(np.abs(a - b) / b) < 1e-4


@pytest.mark.parametrize("S_s", [1.0, 4.0, 16.0])
def test_connectivity_limited_regime(S_s):
    """Analytic limit (paper Eq. regime1): with fast host absorption and slow growth, severity ~ r * delta / S_s."""
    r, delta = 3e-3, 0.3
    W = build_W(S_s, 1e-6, C["alpha"], C["beta"], C["gamma"])
    u = unpack(simulate(W, delta, 0.0, 1e3, r=np.full(N_CITIES, r), T=40.0 / S_s + 5, t_eval=None))
    sev = u["loss"][-1] / K0[S_IDX]
    assert 0.5 < sev / (r * delta / S_s) < 1.5
