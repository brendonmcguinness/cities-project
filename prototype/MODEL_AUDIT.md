# Model audit (17 Nov 2026)

Two pieces of code were checked: the original model (`prototype/src/katrina_atlanta_metapop.py`, `prototype/network/network_metapop.py`; the
code behind the CSSS report's 52-metro results is not in the repository and could not be checked) and the calibrated model
in `empirical/model.py`. Reproduce with:

```bash
cd empirical
python3 tests/audit_original.py       # original model
python3 -m pytest -q tests            # calibrated model (13 tests)
```

## Original model

| Check | Result |
|---|---|
| Migration conserves people (r = 0) | **OK** (drift ~1e-16) |
| Integration (LSODA, shock applied between two integration phases) | **OK** |
| "No disaster" (δ = 0) leaves the system unchanged | **FAILS.** Host cities start building capacity at `t_s` whatever happens, so with no disaster the toy network still grows by 22% and unaffected cities temporarily lose residents to the builders. |
| Struck city settles at its reduced capacity | **FAILS.** Because hosts build spare capacity independently of demand, the occupancy gradient keeps pulling people out of the struck city. With the report's parameters (δ = 0.45, K_max = 1.3K, u = 4, r = 3e-4) New Orleans falls to 0.44 of its size (−56%) and ends below even its reduced capacity (φ = 0.96). This is the likely source of the report's "long-run loss of 55.5%" for a 45% capacity shock. |
| Total population after a disaster | **FAILS.** It ends about 21% *higher* than before: births fill the new host capacity. In the report's Fig. 2, host gains (~0.96M) exceed New Orleans' loss (~0.67M). |
| Results independent of population units | **FAILS.** `u` is an absolute rate (capacity units per time), so changing units from 100k to 10k people changes New Orleans' trajectory (0.44 → 0.50 of initial size at t = 120), and large cities build relatively more slowly than small ones. |
| Coupling comparable across networks | **QUESTIONABLE.** `W = m0 w / max(w)` ties every edge to the single strongest pair: adding one small suburb next to Houston halves New Orleans' out-degree. Comparisons of severity across networks at equal `m0` (report Fig. 3) are confounded. |
| Distances | **WRONG UNITS.** Euclidean distance in degrees of longitude/latitude; km per degree varies by 10–30% with direction and latitude. |
| Severity metric | **CONCEPTUAL.** Severity is crowding loss through the logistic term, which scales with `r` (≈1%/yr) and is not observable. |

**Consequence:** the report's simulation results (Figs 2–4 and the −0.73 scaling) should be rerun after making host
building demand-driven (or tied to the disaster), making `u` relative to city size, and using great-circle distances.

## Calibrated model (`empirical/model.py`)

All 13 tests in `empirical/tests/test_model.py` pass (run: `cd empirical; python3 -m pytest -q tests`):

- migration conserves people (both return mechanisms; with natural increase switched off the total changes by about 1e-15); the reduced struck-place model used for the disaster panels creates or loses nobody (residents + displaced + settled = 1); no disaster gives exactly no change; populations, capacities and the
  displaced stock stay non-negative and the displaced never exceed those who left;
- the struck city's population change equals its cumulative in- minus out-flows; its capacity follows the fitted housing curve;
- with two cities, no rebuilding and gradient return, it reproduces the original two-city code to 1e-5;
- tightening solver tolerances 1000x changes results by < 1e-4;
- the connectivity-limited regime matches the analytic severity `r delta / S_s` within +-50% for S_s = 1, 4, 16.

Known limitation: host capacity added for evacuees is never removed (it is small and slow: +0.04% total population over 50 years
in the earlier version; the current version has the same form).
