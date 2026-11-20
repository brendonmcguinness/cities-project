# Disasters differ in how much displacement they absorb locally

Code, data and manuscript for *"Disasters differ in how much displacement they absorb locally: a calibrated model
tested on fourteen US disasters"*. The project began as the group project "Ecology of Urban Systems" at the Santa Fe
Institute Complex Systems Summer School 2026.

A metapopulation model in which places have populations and housing capacities: residents leave when occupancy
exceeds what a place can hold, return as it rebuilds, displaced residents settle elsewhere over time, and receiving
cities add housing when crowded. The model is calibrated on Hurricane Katrina from public IRS migration and Census
data, fitted to ten counties struck by four disasters, and tested by predicting 21 struck counties in 14 US disasters
(2005–2024) from the other disasters alone.

## Repository layout

| Path | Contents |
|---|---|
| [`empirical/`](empirical/) | The analysis: data download and processing, model, calibration, validation, disaster panels, held-out tests, figures. Scripts are numbered in run order; see [`empirical/README.md`](empirical/README.md). |
| [`empirical/data/`](empirical/data/) | Processed inputs (documented in [`DATA.md`](empirical/data/DATA.md)); `legacy/` holds hand-extracted files used to verify the processing. |
| [`empirical/output/`](empirical/output/) | Results (JSON) and figures produced by the scripts. |
| [`empirical/browser/`](empirical/browser/) | The hazard-geography computation, which runs in a web browser because the map services are not served to scripts. |
| [`empirical/tests/`](empirical/tests/) | Model tests and the audit of the prototype. |
| [`paper/`](paper/) | Manuscript (`main.tex`, `refs.bib`) and figures. |
| [`prototype/`](prototype/) | The original summer-school models, kept for reference and superseded by `empirical/`. |

## Reproducing the results

Python 3.11 or later:

```bash
pip install -r requirements.txt
cd empirical
python3 00_download.py      # public source files into raw/ (checksums in raw/MANIFEST.txt)
bash run_all.sh             # all analyses, outputs and figures
python3 -m pytest -q tests  # model tests
```

Two steps need extra input:

- The American Community Survey tables (`12_acs.py`) and Census 2000 block counts need a free Census API key, read
  from the environment variable `CENSUS_API_KEY` or from `~/.census_api_key`. The key is never written to the
  repository.
- The hazard geography (`browser/hazard_geography.js`) is computed in a browser console; its result is committed as
  `empirical/data/hazard_geography.json`, so the rest of the pipeline runs without it.

The full pipeline takes a few hours, mostly the held-out tests. The manuscript compiles with `pdflatex` and `bibtex`
from `paper/`.

## Data sources

All data are public: IRS Statistics of Income county-to-county migration; Census population and housing estimates,
2000 and 2010 Census tables and geography; American Community Survey 1-year estimates; OpenFEMA Individual
Assistance data; the Census Household Pulse Survey; the FEMA flood hazard layer, NOAA storm-surge hazard maps and
terrain tiles. Details and versions are in [`empirical/data/DATA.md`](empirical/data/DATA.md) and the manuscript's
Methods.

## Licence

Code is released under the [MIT License](LICENSE). The data are public and remain subject to the terms of their
original providers (IRS, Census Bureau, FEMA, NOAA).
