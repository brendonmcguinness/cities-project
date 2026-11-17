#!/usr/bin/env bash
# Reproduce all empirical results. Outputs in ./output.
set -euo pipefail
cd "$(dirname "$0")"
python3 01_describe_data.py
python3 02_gravity.py
python3 03_calibrate.py
python3 04_validate.py
python3 05_figures.py
