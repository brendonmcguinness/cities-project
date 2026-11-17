# Empirical calibration: Hurricane Katrina (and a second test on the 2018 Camp Fire)

Everything here is rebuilt from official IRS and Census files with `bash run_all.sh` (about 15 minutes; `pip install
numpy scipy pandas statsmodels matplotlib xlrd pyshp pyproj pytest`). See [`data/DATA.md`](data/DATA.md) for sources, how each data set
maps onto the model, and limitations, and [`../MODEL_AUDIT.md`](../MODEL_AUDIT.md) for the model checks.

## Model (`model.py`)

Metros are patches with population `N` and capacity `K`; residents move down occupancy gradients `phi = N/K` along a
directed network built from observed pre-disaster IRS flows. Compared with the report model in `src/` and `network/`:
host building is demand-driven (not on a timer), rates are per capita, the evacuation, return and host couplings are
separate (no max-normalised `m0`), distances are great-circle, the struck city's capacity follows measured housing units,
and **displaced residents are tracked**: they return when the struck city has room and settle in their host at a constant
rate. The tracked-return model fits New Orleans' population 19 times better (squared error) than gradient-driven return,
and is the only version that reproduces where returnees came from and when.

## Calibration protocol (`calib.py`, `03_calibrate.py`)

| Stage | Parameters | Fitted to | Estimate (95% CI) |
|---|---|---|---|
| A | capacity lost at landfall `d_0`, permanently `d_inf`, rebuild rate `rho_s` | Census housing units, NOLA metro, 2006-09 | 0.42 (jackknife SE 0.16), 0.11, 0.76/yr |
| B | evacuation `S_s`, return `kappa`, settling `lam` | Census population 2006-09; IRS timing | 1.62/yr (1.54-1.69), 8.8/yr (6.1-12.5), 0.50/yr (0.40-0.64) |
| C | host capacity response `rho_h` | housing units above trend in 10 main host metros, 2006-07 | 1.9/yr (0.4-8.6) |
| — | host-host coupling `S_med` | IRS onward flows (none detected) | 0 |
| — | network | observed 2003-05 IRS flows; gravity fill: alpha 1.02, beta 0.90, gamma 1.12 (38,220 metro pairs) | — |

## Results in brief

- **Validation (not fitted):** destinations of evacuees R² 0.70; origins of returnees R² 0.55; host net domestic
  migration correlation 0.79 (model underestimates host gains: slope 1.55); New Orleans net in-migration 2006-09
  front-loaded as observed (model 43k, 38k, 16k; Census 45k, 28k, 22k).
- **Failures:** Houston's share of evacuees (28%) and returnees (41%) is underestimated by half or more; IRS return timing
  is still too slow (ratio 0.72 vs 1.87); host crowding persists longer than modelled.
- **Robustness:** 13 analyst-choice variants change `S_s`, `kappa`, `lam` by < 15% and validation R² by < 0.04; `rho_h`
  is fragile (0.6-2.3). A pure gravity network performs much worse (destinations R² 0.35).
- **Simulations:** with calibrated parameters, severity (excess-occupancy person-years per resident) is nearly
  independent of which metro is struck (0.239-0.250), scales smoothly with shock size (no threshold) and depends strongly on
  host capacity response (1.14 to 0.15 over `rho_h` = 0.05-10/yr). Hosts bear about half of the crowding.
- **Camp Fire (no refitting):** the Katrina rates overpredict Butte County's population loss (predicted 0.90 and 0.87 of
  trend in 2019-20; observed 0.94 and 0.91): much of the displaced population stayed in the county, doubled up.
