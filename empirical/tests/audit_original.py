"""Audit of the original model code (prototype/src/katrina_atlanta_metapop.py, prototype/network/network_metapop.py)."""
import sys, importlib, numpy as np
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "prototype" / "network")); sys.path.insert(0, str(ROOT / "prototype" / "src"))
import matplotlib; matplotlib.use("Agg")
nm = importlib.import_module("network_metapop")
src = importlib.import_module("katrina_atlanta_metapop")
from scipy.integrate import solve_ivp
NAMES, n = nm.NAMES, nm.N_CITIES
si = NAMES.index("New Orleans")

def run_net(p, T):
    t, Y = nm.simulate(p, T=T, n=4000); return t, Y[:n], Y[n:]

print("== O1 conservation (r = 0): migration alone must conserve people")
p = nm.default_params(); p["r"] = np.zeros(n)
t, N, K = run_net(p, 200); print(f"  network: total drift = {abs(N.sum(0)[-1]-N.sum(0)[0])/N.sum(0)[0]:.2e}")
q = dict(src.params); q["r_K"] = q["r_A"] = 0.0
t2, Y2 = src.simulate(q); tot = Y2[0] + Y2[1]; print(f"  two-city: total drift = {abs(tot[-1]-tot[0])/tot[0]:.2e}")

print("\n== O2 'no disaster' test (delta = 0): nothing should change")
p = nm.default_params(); p["delta"] = 0.0
t, N, K = run_net(p, 200); tot = N.sum(0)
print(f"  network: total population {tot[0]:.1f} -> {tot[-1]:.1f} ({tot[-1]/tot[0]-1:+.1%}); capacity of builders grows anyway")
for c in NAMES: i = NAMES.index(c); print(f"     {c:12s} N {N[i,0]:6.2f} -> {N[i,-1]:6.2f}   min over run {N[i].min():6.2f}")
q = dict(src.params); q["delta_K"] = 0.0
t2, Y2 = src.simulate(q); print(f"  two-city: total {Y2[0,0]+Y2[1,0]:.1f} -> {Y2[0,-1]+Y2[1,-1]:.1f}; New Orleans {Y2[0,0]:.2f} -> min {Y2[0].min():.2f} -> {Y2[0,-1]:.2f}")

print("\n== O3 report-like parameters on the toy network (delta=0.45, Kmax=1.3K, u=4 for all hosts, r=3e-4)")
p = nm.default_params(); p.update(delta=0.45, r=np.full(n, 3e-4), K_max=p["K0"]*1.3)
u = np.full(n, 4.0); u[si] = 0; p["u"] = u
t, N, K = run_net(p, 6000); tot = N.sum(0)
k = np.searchsorted(t, p["t_s"]) - 1
print(f"  New Orleans: N {N[si,k]:.2f} -> {N[si,-1]:.2f} ({N[si,-1]/N[si,k]-1:+.1%});  its capacity fell only {-0.45:+.0%}")
print(f"  New Orleans ends at phi = {N[si,-1]/K[si,-1]:.3f} (below its own reduced capacity)")
print(f"  total population {tot[k]:.1f} -> {tot[-1]:.1f} ({tot[-1]/tot[k]-1:+.1%}) -- the disaster ends up *adding* people")
for c in NAMES:
    i = NAMES.index(c)
    print(f"     {c:12s} min N/N0 = {N[i,k:].min()/N[i,k]:.3f}   final N/N0 = {N[i,-1]/N[i,k]:.3f}")

print("\n== O4 unit dependence of u (same cities, populations in 10k instead of 100k units)")
def nola_traj(scale):
    p = nm.default_params(); p.update(delta=0.45, r=np.full(n, 3e-4))
    p["K0"] = p["K0"] * scale; p["N0"] = p["K0"].copy(); p["K_max"] = p["K0"] * 1.3
    u = np.full(n, 4.0); u[si] = 0; p["u"] = u
    t, N, K = run_net(p, 400); k = np.searchsorted(t, p["t_s"]) - 1
    return t, N[si] / N[si, k], N.sum(0) / N.sum(0)[k]
ta, a, ta_tot = nola_traj(1.0); tb, b, tb_tot = nola_traj(10.0)
i = np.searchsorted(ta, 120)
print(f"  New Orleans N/N0 at t=120: units 100k -> {a[i]:.3f}, units 10k -> {b[np.searchsorted(tb,120)]:.3f}  (should be identical)")

print("\n== O5 max-normalisation of W (report: W = m0 w / max w)")
pos = np.array([nm.CITIES[c]["pos"] for c in NAMES]); K0 = np.array([nm.CITIES[c]["K"] for c in NAMES])
def S_nola(pos, K0, g=1.5):
    d = np.linalg.norm(pos[:, None] - pos[None], axis=2); np.fill_diagonal(d, np.inf)
    w = K0[:, None] * K0[None] / d**g; W = 0.4 * w / w.max(); return W[si].sum()
pos2 = np.vstack([pos, [-95.0, 29.6]]); K2 = np.r_[K0, 6.0]          # add Pasadena TX-like suburb next to Houston
print(f"  S_NewOrleans with 6 cities = {S_nola(pos, K0):.4f};  after adding one small city near Houston = {S_nola(pos2, K2):.4f}")

print("\n== O6 distances in degrees (network/): error vs great-circle km")
def hav(a, b):
    a, b = np.radians(a), np.radians(b)
    return 2*6371*np.arcsin(np.sqrt(np.sin((b[1]-a[1])/2)**2 + np.cos(a[1])*np.cos(b[1])*np.sin((b[0]-a[0])/2)**2))
pairs = [("New Orleans", "Houston"), ("New Orleans", "Memphis"), ("New Orleans", "Atlanta")]
for x, y in pairs:
    dd = np.linalg.norm(pos[NAMES.index(x)] - pos[NAMES.index(y)]); dk = hav(pos[NAMES.index(x)], pos[NAMES.index(y)])
    print(f"  {x}-{y}: {dd:.2f} deg = {dk:.0f} km  -> km per degree {dk/dd:.0f}")
print("  (km per degree differs by direction: E-W pairs are shortened by cos(latitude), up to ~30% at 47N)")
