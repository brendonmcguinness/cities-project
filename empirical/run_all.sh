#!/usr/bin/env bash
# Reproduce every number and figure in the paper from public sources. Outputs in ./output.
# Takes a few hours on a laptop, mostly the held-out tests (14_holdout.py runs six times).
# Needs a Census API key in CENSUS_API_KEY or ~/.census_api_key (12_acs.py, 17_hazard_geography.py --build-hu00,
# 23_capacity_check.py). The browser step (browser/hazard_geography.js) is not rerun; its output is in data/.
set -euo pipefail
cd "$(dirname "$0")"

# --- data
python3 00_download.py          # raw files from irs.gov and census.gov (checksums in raw/MANIFEST.txt)
python3 00_build_data.py        # processed inputs in data/, verified against hand-extracted files in data/legacy/
python3 12_acs.py               # ACS 1-year county housing and household size

# --- Hurricane Katrina: network model
python3 01_describe_data.py     # descriptive statistics, fig_data (Fig. S1)
python3 02_gravity.py           # network-wide and New Orleans gravity estimates, bootstrap
python3 03_calibrate.py         # calibration (stages A-C), onward-migration evidence
python3 04_validate.py          # out-of-sample validation
python3 05_uncertainty.py       # jackknife and Monte Carlo propagation
python3 06_sensitivity.py       # 11 analyst-choice variants and a gravity-only network (output/sensitivity/)
python3 07_simulations.py       # severity across metros and parameter sweeps
python3 08_campfire.py          # Camp Fire 2018 with the Katrina rates, no refitting

# --- disaster panels
python3 10_events.py            # ten-county Census panel: tolerated occupancy, transfer tests
python3 13_acs_panel.py         # 43 pre-specified county-disaster pairs with ACS-measured capacity
python3 23_capacity_check.py    # profile likelihood; ACS-capacity refit of five Census-panel counties

# --- held-out prediction (leave one disaster out)
python3 14_holdout.py           # main run (21 counties, 14 disasters)
for v in "HOLDOUT_TAU=0.05 HOLDOUT_TAG=_tau005" "HOLDOUT_TAU=0.2 HOLDOUT_TAG=_tau02" \
         "HOLDOUT_MIN_LOSS=0.03 HOLDOUT_TAG=_loss3" "HOLDOUT_MIN_LOSS=0.05 HOLDOUT_TAG=_loss5" \
         "HOLDOUT_ACSCAP=1 HOLDOUT_TAG=_acscap"; do
  env $v python3 14_holdout.py  # sensitivity runs
done
python3 20_holdout_uncertainty.py  # cluster bootstrap, Wilcoxon tests, sensitivity summary
python3 24_baseline.py          # statistical baselines in the same folds
python3 26_pred_intervals.py    # prediction intervals and coverage

# --- mechanism and further checks
python3 16_damage_geography.py  # FEMA registrations per ZIP code vs tolerated occupancy
python3 17_hazard_geography.py  # pre-disaster hazard geography vs tolerated occupancy
python3 18_holdout_geo.py       # held-out prediction with tolerated occupancy from hazard geography
python3 21_fema_mechanism.py    # FEMA registrant records (hotel stays) vs tolerated occupancy
python3 22_pulse.py             # Household Pulse displacement vs ACS housing loss
python3 25_destinations.py      # destinations beyond Katrina (IRS county flows)

# --- figures
python3 05_figures.py           # fig_calibration (3), fig_events (4), fig_panel (5), fig_simulations (S2), fig_campfire
python3 09_map.py               # fig_map (2)
python3 11_schematic.py         # fig_schematic (1)
python3 15_holdout_fig.py       # fig_holdout (6)
python3 19_hazard_fig.py        # fig_hazard (7)

python3 -m pytest -q tests      # model tests
