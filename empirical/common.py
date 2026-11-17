"""Shared data loading for the empirical calibration of the urban metapopulation model.

All inputs live in ./data (see data/DATA.md). Everything is plain text so it can be inspected and diffed.
"""
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
OUT = HERE / "output"
OUT.mkdir(exist_ok=True)

NOLA = "35380"                      # CBSA code, New Orleans-Metairie-Kenner MSA (2009 delineation)
LANDFALL = 2005 + 240 / 365         # 29 Aug 2005, in decimal years

# NOLA-metro totals from the IRS files (all destinations, including flows below the disclosure threshold)
IRS_TOTAL_OUT = {"0405": 64343, "0506": 258823, "0607": 71891, "0708": 62511}
IRS_TOTAL_IN = {"0405": 58647, "0506": 69189, "0607": 91332, "0708": 76087}


def read(name, **kw):
    return pd.read_csv(DATA / name, sep="|", comment="#", **kw)


def hav(la1, lo1, la2, lo2):
    """Great-circle distance in km."""
    la1, lo1, la2, lo2 = map(np.radians, (la1, lo1, la2, lo2))
    return 2 * 6371 * np.arcsin(np.sqrt(np.sin((la2 - la1) / 2) ** 2
                                        + np.cos(la1) * np.cos(la2) * np.sin((lo2 - lo1) / 2) ** 2))


def flows_county():
    return read("irs_nola_flows_2004_2008.psv", dtype={"fips": str})


def parishes():
    return read("census_nola_parishes_2000_2009.psv", dtype={"fips": str})


def counties():
    """Destination counties with population and distance from the population-weighted NOLA centroid, merged with flows."""
    c = read("census_counties_2005.psv", dtype={"fips": str})
    p = parishes()
    lat0, lon0 = np.average(p.lat, weights=p.p2005), np.average(p.lon, weights=p.p2005)
    d = c.merge(flows_county(), on="fips", how="left").fillna(0)
    d["dist"] = hav(lat0, lon0, d.lat, d.lon)
    d["excess"] = (d.o0506 - d.o0405).clip(lower=0)
    return d


def metros():
    """196 metros with populations, components of change, and NOLA flows aggregated from counties."""
    m = read("cbsa_metros_2009def.psv", dtype={"code": str})
    comp = read("cbsa_components_2009vintage.psv", dtype={"code": str})
    m = m.merge(comp, on="code")
    line = [l for l in open(DATA / "county_to_cbsa_flowcounties.txt") if not l.startswith("#")][0]
    xw = dict(p.split(":") for p in line.strip().split(","))
    f = flows_county()
    f["code"] = f.fips.map(xw)
    cols = [c for c in f.columns if c[0] in "oi" and c[1:].isdigit()]
    m = m.merge(f.dropna(subset=["code"]).groupby("code")[cols].sum(), left_on="code", right_index=True, how="left")
    m[cols] = m[cols].fillna(0)
    s = m.code == NOLA
    m["dist"] = hav(m.lat[s].item(), m.lon[s].item(), m.lat, m.lon)
    m["excess"] = (m.o0506 - m.o0405).clip(lower=0)
    m["excess_ret"] = (m.i0607 - m.i0405).clip(lower=0)
    m["dm_base"] = m[["dm2002", "dm2003", "dm2004", "dm2005"]].mean(axis=1)
    m["dm_excess2006"] = m.dm2006 - m.dm_base
    return m.reset_index(drop=True)
