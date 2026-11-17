# Empirical calibration: Hurricane Katrina

This folder confronts the metapopulation model with data on Hurricane Katrina (New Orleans, 29 Aug 2005). Every model
parameter is either taken directly from data or fitted to data; see [`data/DATA.md`](data/DATA.md) for sources and how
each data set maps onto the model.

## Run

```bash
pip install numpy scipy pandas statsmodels matplotlib
cd empirical
bash run_all.sh          # about 3 minutes; outputs go to empirical/output/
```

| Script | What it does | Main outputs |
|---|---|---|
| `01_describe_data.py` | coverage and descriptive statistics, descriptive figure | `data_summary.json`, `fig_data.*` |
| `02_gravity.py` | PPML gravity at county and metro level; joint pre-disaster fit of `alpha`, `beta`, `gamma` | `gravity.json` |
| `03_calibrate.py` | stage 1: `delta`, `rho_s`, `S_s` from Census population and IRS timing; stage 2: `S_med`, `rho_h` from IRS destinations | `calibration.json` |
| `04_validate.py` | out-of-sample checks: return origins (IRS), host and NOLA net domestic migration (Census) | `validation.json`, `validation_*.csv` |
| `05_figures.py` | calibration/validation figure and county gravity figure | `fig_calibration.*`, `fig_gravity.*` |

## Model changes relative to `src/` and `network/`

`model.py` keeps the report's equations (logistic growth, gradient-driven gravity flows) and changes three things the
data required:

1. **Two couplings.** Edges touching the struck city have their own scale (`S_s`, evacuation rate); other edges are scaled
   by `S_med`. With one coupling for the whole network (the report's `m0`), any evacuation speed consistent with the data
   makes onward migration between host cities equally fast and scatters evacuees across the country in weeks.
2. **Rebuilding.** The struck city rebuilds, `dK_s/dt = rho_s (K_s(0) - K_s)`; return migration then emerges from the
   occupancy gradient.
3. **Crowding-driven host capacity.** `dK_i/dt = rho_h (N_i - K_i)_+` instead of a fixed build rate toward `1.3 K`.

Distances are great-circle km (the toy network in `network/` uses degrees of longitude/latitude).

## Results in brief (see `output/*.json`)

- Pre-disaster gravity: `alpha = 1.32 ± 0.06`, `beta = 1.38 ± 0.07`, `gamma = 2.00 ± 0.07`.
- Stage 1: `delta ≈ 0.28`, `rho_s ≈ 0.12 /yr`, `S_s ≈ 3.8 /yr`; NOLA population relative to trend reproduced within
  0.6 percentage points, 2006–09.
- Stage 2 runs to the boundary `S_med → 0`, large `rho_h`: the destination data give **no evidence for capacity-driven
  onward migration** between host cities, and the model's destinations equal the static gravity prediction.
- Out of sample (empirical network for the struck city's edges): destinations R² 0.74, return origins R² 0.62, host
  domestic migration correlation 0.80 (62 metros within 900 km). The model **underestimates** Houston (14% vs 28% of
  evacuees; 21% vs 40% of returnees), underestimates host gains by about a quarter, and spreads return migration too evenly
  over 2006–09.
