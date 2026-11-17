"""Correctness tests for empirical/model.py.  Run from empirical/:  python3 -m pytest -q tests
They check properties any correct implementation must have, independent of the calibrated values."""
import json, sys
from pathlib import Path
import numpy as np
import pytest

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent / "src"))
from model import Network, simulate, unpack, capacity_struck   # noqa: E402

NET = Network()
C = json.load(open(HERE / "output" / "calibration.json")); A, B, CC = C["A"], C["B"], C["C"]
W_CAL = NET.W(B["S_s"], 1.0, 0.0)
W_GRAD = NET.W(1.6, 0.5, 0.01)      # gradient return and some host coupling, to exercise every term
TE = np.linspace(0, 4, 41)


@pytest.mark.parametrize("W,kw", [(W_CAL, dict(kappa=8.0, lam=0.5)), (W_GRAD, {})])
def test_migration_conserves_people(W, kw):
    u = unpack(simulate(NET, W, 0.4, 0.1, 0.7, 2.0, r=np.zeros(NET.n), t_eval=TE, **kw), NET.n)
    tot = u["N"].sum(0)
    assert np.max(np.abs(tot - tot[0])) / tot[0] < 1e-6


@pytest.mark.parametrize("W,kw", [(W_CAL, dict(kappa=8.0, lam=0.5)), (W_GRAD, {})])
def test_no_disaster_means_no_change(W, kw):
    u = unpack(simulate(NET, W, 0.0, 0.0, 0.7, 2.0, t_eval=TE, **kw), NET.n)
    assert np.max(np.abs(u["N"] - NET.K0[:, None])) < 1e-9 and np.max(np.abs(u["K"] - NET.K0[:, None])) < 1e-9
    assert u["cum_out"][-1] < 1e-12 and u["excess_occ"][-1] < 1e-12


def test_positive_and_displaced_bounded():
    u = unpack(simulate(NET, W_CAL, 0.8, 0.3, 0.5, 0.5, t_eval=TE, kappa=8.0, lam=0.5), NET.n)
    assert (u["N"] > 0).all() and (u["K"] > 0).all() and (u["D"] >= -1e-12).all()
    assert (u["D"].sum(0) <= u["cum_out"] + 1e-9).all()          # cannot hold more displaced than ever left


def test_flow_accounting_struck_city():
    u = unpack(simulate(NET, W_CAL, 0.4, 0.1, 0.7, 2.0, r=np.zeros(NET.n), t_eval=TE, kappa=8.0, lam=0.5), NET.n)
    dN = u["N"][NET.s] - u["N"][NET.s, 0]
    assert np.max(np.abs(dN - (u["cum_in"] - u["cum_out"]))) < 1e-6


def test_struck_capacity_follows_housing_curve():
    u = unpack(simulate(NET, W_CAL, A["d0"], A["dinf"], A["rho_s"], 2.0, t_eval=TE, kappa=8.0, lam=0.5), NET.n)
    expect = capacity_struck(TE, NET.K0[NET.s], A["d0"], A["dinf"], A["rho_s"])
    assert np.max(np.abs(u["K"][NET.s] - expect)) / NET.K0[NET.s] < 1e-6


def test_reduces_to_original_two_city_model():
    """Two cities, no rebuilding, no host response, gradient return: identical to src/ (gradient mode, u = 0)."""
    import matplotlib; matplotlib.use("Agg")
    import katrina_atlanta_metapop as src
    p = dict(src.params); p.update(u=0.0, t_s=1.0, migration_mode="gradient")
    t_ref, Y = src.simulate(p, T=31.0, n=3101)
    keep = t_ref > 1.0 + 1e-9; t_ref, Y = t_ref[keep] - 1.0, Y[:, keep]
    K0 = np.array([p["K_K0"], p["K_A0"]]); W = np.array([[0, p["m"]], [p["m"], 0]], float)
    sol = simulate(NET, W, p["delta_K"], p["delta_K"], 1.0, 0.0, T=30.0, t_eval=t_ref, r=np.array([p["r_K"], p["r_A"]]),
                   K0=K0, s=0, rtol=1e-9, atol=1e-11)
    assert np.max(np.abs(sol.y[:2] - Y[:2])) < 1e-5


def test_solver_converged():
    a = simulate(NET, W_CAL, A["d0"], A["dinf"], A["rho_s"], CC["rho_h"], t_eval=TE, kappa=B["kappa"], lam=B["lam"]).y[NET.s]
    b = simulate(NET, W_CAL, A["d0"], A["dinf"], A["rho_s"], CC["rho_h"], t_eval=TE, kappa=B["kappa"], lam=B["lam"], rtol=1e-9, atol=1e-12).y[NET.s]
    assert np.max(np.abs(a - b) / b) < 1e-4


@pytest.mark.parametrize("S_s", [1.0, 4.0, 16.0])
def test_connectivity_limited_regime(S_s):
    """Analytic limit: fast host absorption, slow growth, no rebuilding -> crowding loss ~ r * delta / S_s."""
    r, delta = 3e-3, 0.3
    W = NET.W(S_s, 1.0, 0.0)
    u = unpack(simulate(NET, W, delta, delta, 1.0, 1e3, r=np.full(NET.n, r), T=40.0 / S_s + 5, kappa=8.0, lam=0.5), NET.n)
    sev = u["loss"][-1] / NET.K0[NET.s]
    assert 0.5 < sev / (r * delta / S_s) < 1.5
