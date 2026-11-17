#!/usr/bin/env bash
# Reproduce every empirical number and figure from official sources. Outputs in ./output. ~15 minutes.
set -euo pipefail
cd "$(dirname "$0")"
python3 00_download.py          # raw files from irs.gov and census.gov (checksums in raw/MANIFEST.txt)
python3 00_build_data.py        # processed inputs in data/, verified against hand-extracted files in data/legacy/
python3 01_describe_data.py     # descriptive statistics, fig_data
python3 02_gravity.py           # network-wide and New Orleans gravity estimates, bootstrap
python3 03_calibrate.py         # calibration (stages A-C), onward-migration evidence
python3 04_validate.py          # out-of-sample validation
python3 05_uncertainty.py       # jackknife and Monte Carlo propagation
python3 07_simulations.py       # severity across all metros, sweeps
python3 08_campfire.py          # second disaster (Camp Fire 2018), no refitting
python3 05_figures.py           # fig_calibration, fig_simulations, fig_campfire
python3 09_map.py               # fig_map (Fig. 1): evacuee destinations and returnee origins
python3 06_sensitivity.py       # 13 analyst-choice variants (writes to output/sensitivity/)
python3 -m pytest -q tests      # model correctness tests
