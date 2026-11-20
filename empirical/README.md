# Analysis

Every number and figure in the paper is rebuilt from public files with `python3 00_download.py` and `bash run_all.sh`
(a few hours, mostly the held-out tests; dependencies in [`../requirements.txt`](../requirements.txt)). Sources, how
each data set maps onto the model, and limitations are in [`data/DATA.md`](data/DATA.md); the audit of the original
summer-school model is in [`../prototype/MODEL_AUDIT.md`](../prototype/MODEL_AUDIT.md).

## Model

- `model.py`: network model. Metros are patches with population `N` and capacity `K`; residents move down occupancy
  gradients `phi = N/K` along a directed network built from observed pre-disaster IRS flows (gravity fill for
  suppressed pairs). Displaced residents are tracked: they return when the struck city has room and settle in their
  host at a constant rate. Host capacity responds to crowding; distances are great-circle.
- `struck.py`: the same dynamics reduced to the struck place, used by the disaster panels. `phis` is the occupancy
  residents tolerate before leaving (1 = full occupancy).
- `calib.py`, `common.py`, `figstyle.py`: calibration routines, shared paths and data loaders, figure style.

## Scripts (run order in `run_all.sh`)

| Script | What it does | Main output |
|---|---|---|
| `00_download.py`, `00_build_data.py` | Download IRS and Census files; build processed inputs and verify them against hand-extracted files in `data/legacy/` | `raw/`, `data/` |
| `12_acs.py` | ACS 1-year county tables 2005–2024 (Census API key needed); usable units = units − other vacant − seasonal | `data/acs_county.psv` |
| `01_describe_data.py` | Descriptive statistics of the New Orleans shock | `data_summary.json`, Fig. S1 |
| `02_gravity.py` | Gravity estimates (network-wide and New Orleans), bootstrap | `gravity.json` |
| `03_calibrate.py` | Katrina calibration: capacity path (A), evacuation, return and settling (B), host response (C) | `calibration.json` |
| `04_validate.py` | Out-of-sample validation: destinations, returnee origins, host migration | `validation.json` |
| `05_uncertainty.py`, `06_sensitivity.py` | Jackknife and Monte Carlo propagation; 11 analyst-choice variants | `uncertainty.json`, `sensitivity.json` |
| `07_simulations.py`, `08_campfire.py` | Severity across metros and sweeps; Camp Fire with the Katrina rates | `simulations.json`, `campfire.json` |
| `10_events.py` | Ten-county Census panel: housing-only, tolerated-occupancy and pooled fits; transfer tests | `events.json` |
| `13_acs_panel.py` | 43 pre-specified county–disaster pairs (2008–2024) with ACS-measured capacity | `acs_panel.json` |
| `23_capacity_check.py` | Profile-likelihood intervals for `phis`; refit of five counties with ACS capacity | `capacity_check.json` |
| `14_holdout.py` | Leave-one-disaster-out prediction, 21 counties, 14 disasters; settings by environment variables (`HOLDOUT_TAU`, `HOLDOUT_MIN_LOSS`, `HOLDOUT_ACSCAP`, `HOLDOUT_TAG`) | `holdout*.json` |
| `20_holdout_uncertainty.py` | Cluster bootstrap over disasters, Wilcoxon tests, sensitivity summary | `holdout_uncertainty.json` |
| `24_baseline.py` | Statistical forecasts in the same folds | `baseline.json` |
| `26_pred_intervals.py` | Prediction intervals and their coverage | `pred_intervals.json` |
| `16_damage_geography.py` | FEMA registrations per ZIP code vs `phis` | `damage_geography.json` |
| `browser/hazard_geography.js`, `17_hazard_geography.py` | Pre-disaster share of housing outside flood, surge and levee zones and above 3 m (browser step; result in `data/hazard_geography.json`); tests against `phis` | `hazard_geography.json` |
| `18_holdout_geo.py` | Held-out prediction with `phis` from hazard geography | `holdout_geo.json` |
| `21_fema_mechanism.py` | FEMA registrant records (hotel stays) vs `phis` | `fema_mechanism.json` |
| `22_pulse.py` | Household Pulse Survey displacement vs ACS housing loss (Florida/Ian, Hawaii/Lahaina) | `pulse.json` |
| `25_destinations.py` | Destinations of excess out-migration beyond Katrina (IRS county flows) | `destinations.json` |
| `05_figures.py`, `09_map.py`, `11_schematic.py`, `15_holdout_fig.py`, `19_hazard_fig.py` | Figures | `output/fig_*.pdf` |

Figures in the paper: `fig_schematic` (Fig. 1), `fig_map` (2), `fig_calibration` (3), `fig_events` (4),
`fig_panel` (5), `fig_holdout` (6), `fig_hazard` (7); supplementary `fig_data` (S1), `fig_simulations` (S2).
`paper/figures/` holds copies named by figure number.

## Main results

- **Katrina calibration:** evacuation 1.62/yr, return 8.8/yr per unit spare capacity, settling 0.50/yr, host
  response 1.9/yr (0.4–8.6). Not fitted, the model reproduces evacuee destinations (R² 0.70) and returnee origins
  (R² 0.55), but underestimates Houston's share by half and returns too slowly in the IRS data.
- **Ten-county panel:** housing loss alone fits the flooded New Orleans parishes but not the other events; a
  per-county tolerated occupancy (0.80–1.25) fits all ten. With ACS-measured capacity, tolerated occupancy rises in
  all five covered counties, strongly in Orleans (0.97 to 1.27), so the contrast is robust only for Mississippi and
  Butte.
- **Held-out prediction:** median error 3.0 percentage points with a pooled tolerated occupancy of 1.03, against
  3.6 with 1.0, 4.9 if population tracks capacity, 6.3 for no change, and 4.0–5.6 for statistical forecasts. The gain
  over capacity tracking holds in sensitivity runs, including survey-measured capacity (3.0 vs 6.1); the gain over
  `phis = 1` is not significant.
- **Mechanism checks** are weak or illustrative: hazard geography ranks `phis` only with Census capacity and does not
  improve prediction; FEMA hotel stays are lower where `phis` is higher (6 counties); Household Pulse displacement
  exceeds measured housing loss (Lahaina 2×, Ian 6×); destinations beyond Katrina follow pre-disaster flows only
  weakly (median R² 0.17).
