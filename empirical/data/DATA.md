# Data

All files are pipe-separated text with `#` comment headers that state their source and a checksum. They were extracted on
17 Nov 2026 from the original public files listed below; the checksums were verified after extraction.

| File | Source | Unit | Years | What it contains |
|---|---|---|---|---|
| `irs_nola_flows_2004_2008.psv` | IRS Statistics of Income, county-to-county migration (`county0405.zip` … `county0708.zip`, irs.gov/statistics/soi-tax-stats-migration-data) | exemptions (≈ persons) | filing years 2004–05 to 2007–08 | Flows **out of** (`oYYYY`) and **into** (`iYYYY`) the 7 parishes of the 2003 New Orleans MSA, summed over parishes, by US destination/origin county. Intra-metro flows excluded. 430 counties. |
| `census_counties_2005.psv` | Census intercensal county estimates 2000–2010 (`co-est00int-tot.csv`); 2010 Census Gazetteer (`Gaz_counties_national.zip`) | persons; degrees | 1 July 2005 | Population and internal-point coordinates of every county with a NOLA flow, plus every other county with ≥100k residents (684 counties). |
| `census_nola_parishes_2000_2009.psv` | Census intercensal county estimates | persons | 1 July 2000–2009 | Annual population of the 7 NOLA parishes. |
| `cbsa_metros_2009def.psv` | Census metro delineation (`cbsa-est2009-alldata.csv`) + intercensal county estimates + Gazetteer | persons; degrees | 2001, 2004–2007 | 196 metropolitan areas (2009 delineation) with ≥250k residents in 2005 or within 600 km of New Orleans. Populations are sums of intercensal county estimates; positions are 2005-population-weighted means of county internal points. |
| `cbsa_components_2009vintage.psv` | Census Vintage 2009 metro components of change (`cbsa-est2009-alldata.csv`) | persons per year | years ending 30 June 2002–2009 | Net domestic migration `dmYYYY`, net international migration 2006, natural-increase rate 2002–05 (births − deaths per resident) for the 196 metros. |
| `county_to_cbsa_flowcounties.txt` | Census 2009 delineation | — | — | County → metro code for the 322 flow counties that lie in the 196 metros. |

## How the data map onto the model

| Model quantity | Data | Script |
|---|---|---|
| initial capacity `K_i(0)` | metro population, 1 July 2005 | `model.py` |
| growth rate `r_i` | metro natural-increase rate, 2002–05 | `model.py` |
| gravity exponents `alpha`, `beta`, `gamma` | IRS flows into and out of NOLA, 2004–05 (before the storm), joint PPML | `02_gravity.py` |
| shock size `delta`, rebuild rate `rho_s`, evacuation rate `S_s` | Census NOLA population July 2006–09 relative to the 2001–05 trend; IRS share of 2-yr excess out-migration in filing year 2005–06 | `03_calibrate.py` |
| host coupling `S_med`, host capacity response `rho_h` | IRS destinations of 2005–06 excess out-migration | `03_calibrate.py` |
| validation | IRS return origins 2006–07; Census net domestic migration of hosts (2005–06) and of NOLA (2006–09) | `04_validate.py` |

## Limitations to keep in mind

- **IRS coverage.** IRS counts tax filers and their dependants. Households that do not file — disproportionately poor, and
  so disproportionately among Katrina evacuees — are missing. County flows below 10 returns are suppressed: only 32% of
  normal (2004–05) NOLA out-migration, and 71% in 2005–06, can be assigned to a destination county.
- **IRS timing.** A filing year compares addresses on returns filed in consecutive springs, so "2005–06" covers moves
  between roughly April 2005 and April 2006, including the evacuation (29 Aug 2005).
- **Census estimates** are model-based (administrative records), not counts. The metro components use the Vintage 2009
  series, which was not revised to the 2010 Census; populations use the revised intercensal series.
- **Other struck places.** Katrina also hit the Mississippi coast (Gulfport, Pascagoula), which lost population in 2005–06;
  the model treats New Orleans as the only struck city.
- **Metro coverage.** The 196 metros hold 222 million people and receive 88% of the identified 2005–06 out-flows; flows
  to smaller places are outside the model.
