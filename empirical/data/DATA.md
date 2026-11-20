# Data

Every processed file here is rebuilt from official raw files by `00_download.py` (raw files and SHA-256 checksums in
`raw/MANIFEST.txt`) and `00_build_data.py`. The build checks itself against the files extracted by hand earlier
(`legacy/`) and stops if any New Orleans flow (2004-05 to 2007-08, both directions) or metro population differs.

| File | Built from | Content |
|---|---|---|
| `counties.psv` | Census intercensal county population and housing-unit estimates 2000-2010; 2010 Gazetteer | population and housing units, 1 July 2000-2009, and internal-point coordinates for every US county |
| `metros.psv` | the above + Census Vintage 2009 metro file (2009 delineation) | 366 metros: population and housing units (sums of counties), population-weighted centroid, components of change 2001-2009 (natural increase, net domestic and international migration, years ending 30 June) |
| `county_metro.psv` | 2009 delineation | county -> metro crosswalk |
| `irs_county_flows.psv.gz` | IRS SOI county-to-county migration, filing years 2003-04 to 2007-08 | every county-to-county flow above the disclosure threshold (~80,000 pairs per year): returns and exemptions |
| `irs_nola_totals.psv` | IRS summary records | New Orleans-metro total out- and in-migration including suppressed flows |

The second-event test (`08_campfire.py`) reads its raw files directly: IRS county flows 2016-17 to 2020-21 (CSV) and
Census Vintage 2020 county population/components and California housing-unit estimates.

## Mapping data onto the model

| Model quantity | Data | Where |
|---|---|---|
| capacities `K_i(0)` | metro population, 1 July 2005 | `model.py` |
| growth rates `r_i` | natural increase 2002-05 | `model.py` |
| migration network `W` | observed pre-disaster metro-to-metro IRS flows (mean 2003-04, 2004-05); suppressed pairs filled with the network-wide gravity expectation, capped at the suppression threshold | `model.py`, `02_gravity.py` |
| struck-city capacity path `d_0, d_inf, rho_s` | New Orleans metro housing units 2006-09 relative to 2001-05 trend | `calib.py` stage A |
| evacuation `S_s`, return `kappa`, settling `lam` | New Orleans metro population 2006-09 relative to trend; IRS timing of excess out-migration | stage B |
| host capacity response `rho_h` | housing units above trend in the 10 metros receiving most evacuees, 2006-07 | stage C |
| host-host coupling `S_med = 0` | IRS host-to-host flows: no onward migration from hosts that received evacuees | `03_calibrate.py` |
| validation | IRS destinations 2005-06, return origins and timing 2006-08; Census host domestic migration and occupancy; Camp Fire | `04_validate.py`, `08_campfire.py` |

## Limitations

- IRS counts tax filers and dependants only; non-filers (disproportionately poor, and so disproportionately among Katrina
  evacuees) are missing. Flows below 10 returns are suppressed: 31-32% of normal New Orleans out-migration is assigned to a
  county (71% in 2005-06).
- An IRS filing year compares addresses on returns filed in consecutive springs (2005-06 covers about April 2005 to April 2006).
- Census population and housing estimates are model-based; post-Katrina housing estimates for the struck parishes are special
  estimates. Components of change are Vintage 2009 (not revised to the 2010 Census).
- Hurricane Rita (Sept 2005) also struck Beaumont and Lake Charles and Katrina the Mississippi coast; these metros are
  excluded from host-side targets and validation.
- Excess flows and growth are measured against 2001-05 (or 2003-05) baselines; the mid-2000s Sunbelt migration boom affects
  host-side measures (for example Phoenix appears among the largest gainers in 2005-06).
