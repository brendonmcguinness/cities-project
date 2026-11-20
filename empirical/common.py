"""Shared data access for the empirical calibration. All inputs are built from official raw files by
00_download.py and 00_build_data.py (see data/DATA.md)."""
import gzip, os
from functools import lru_cache
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
OUT = Path(os.environ.get("CAL_OUT", HERE / "output"))      # sensitivity variants write to their own folders
OUT.mkdir(parents=True, exist_ok=True)

NOLA = "35380"                                   # New Orleans-Metairie-Kenner MSA (2009 delineation)
NOLA_PARISHES = ["22051", "22071", "22075", "22087", "22089", "22095", "22103"]
OTHER_STRUCK = {"25060": "Gulfport (Katrina)", "37700": "Pascagoula (Katrina)",
                "13140": "Beaumont (Rita)", "29340": "Lake Charles (Rita)"}
LANDFALL = 2005 + 240 / 365                      # 29 Aug 2005
# analyst choices that 06_sensitivity.py varies through environment variables
BASE_YEARS = os.environ.get("CAL_BASE", "0304,0405").split(",")            # pre-disaster IRS filing years
TREND_YEARS = tuple(int(x) for x in os.environ.get("CAL_TREND", "2001,2005").split(","))   # trend window (first, last)
MIN_POP = int(os.environ.get("CAL_MINPOP", "250000")); MAX_KM = float(os.environ.get("CAL_MAXKM", "600"))
TOP_K = int(os.environ.get("CAL_TOPK", "10"))
JULY = {y: y + 0.5 - LANDFALL for y in range(2006, 2010)}   # model time of 1 July
APRIL = {"0506": 0.62, "0607": 1.62, "0708": 2.62}           # model time at the end of each IRS filing year


def read(name, **kw):
    return pd.read_csv(DATA / name, sep="|", comment="#", **kw)


def hav(la1, lo1, la2, lo2):
    la1, lo1, la2, lo2 = map(np.radians, (la1, lo1, la2, lo2))
    return 2 * 6371 * np.arcsin(np.sqrt(np.sin((la2 - la1) / 2) ** 2 + np.cos(la1) * np.cos(la2) * np.sin((lo2 - lo1) / 2) ** 2))


@lru_cache(None)
def counties():
    return read("counties.psv", dtype={"fips": str})


@lru_cache(None)
def metros_all():
    return read("metros.psv", dtype={"code": str}).set_index("code")


@lru_cache(None)
def county_flows():
    return pd.read_csv(DATA / "irs_county_flows.psv.gz", sep="|", comment="#", dtype={"year": str, "orig": str, "dest": str})


@lru_cache(None)
def nola_totals():
    return read("irs_nola_totals.psv", dtype={"year": str}).set_index("year")


def linear_trend(row, prefix, years=None):
    """Linear trend fitted to the trend window (default 2001-2005), returned as a function of calendar year."""
    yrs = np.arange(TREND_YEARS[0], TREND_YEARS[1] + 1) if years is None else np.array(list(years))
    b = np.polyfit(yrs, [row[f"{prefix}{y}"] for y in yrs], 1)
    return lambda t: np.polyval(b, t)


@lru_cache(None)
def metro_flows():
    """Metro-to-metro IRS flows (exemptions) by filing year, all 366 metros; plus NOLA flows by county."""
    xw = read("county_metro.psv", dtype={"fips": str, "code": str})
    m = dict(zip(xw.fips, xw.code)); F = county_flows().copy()
    F["o"], F["d"] = F.orig.map(m), F.dest.map(m)
    M = F.dropna(subset=["o", "d"]); M = M[M.o != M.d]
    return M.groupby(["year", "o", "d"]).exemptions.sum().unstack("year").fillna(0)


def select_metros(min_pop=None, max_km=None):
    """The model network: metros with >= min_pop residents in 2005 or within max_km of New Orleans."""
    min_pop = MIN_POP if min_pop is None else min_pop; max_km = MAX_KM if max_km is None else max_km
    a = metros_all(); s = a.loc[NOLA]
    d = hav(s.lat, s.lon, a.lat, a.lon)
    return a[(a.p2005 >= min_pop) | (d <= max_km)].index.tolist()


@lru_cache(None)
def metros(min_pop=None, max_km=None):
    """Model network table with NOLA flows (out, in), housing, components and derived quantities."""
    a = metros_all(); codes = select_metros(min_pop, max_km); m = a.loc[codes].copy()
    s = a.loc[NOLA]; m["dist"] = hav(s.lat, s.lon, m.lat, m.lon)
    MF = metro_flows()
    for y in ["0304", "0405", "0506", "0607", "0708"]:
        m[f"o{y}"] = MF[y].xs(NOLA, level="o").reindex(m.index).fillna(0) if y in MF else 0
        m[f"i{y}"] = MF[y].xs(NOLA, level="d").reindex(m.index).fillna(0) if y in MF else 0
    m["o_base"] = m[[f"o{y}" for y in BASE_YEARS]].mean(1); m["i_base"] = m[[f"i{y}" for y in BASE_YEARS]].mean(1)
    m["excess"] = (m.o0506 - m.o_base).clip(lower=0)
    m["excess_ret"] = (m.i0607 - m.i_base).clip(lower=0)
    m["dm_base"] = m[[f"dm{y}" for y in range(max(TREND_YEARS[0], 2002), TREND_YEARS[1] + 1)]].mean(1)
    m["dm_excess2006"] = m.dm2006 - m.dm_base
    m["ppu2005"] = m.p2005 / m.hu2005                                   # persons per housing unit
    for y in (2006, 2007, 2008, 2009):
        m[f"hu_excess{y}"] = [r[f"hu{y}"] - linear_trend(r, "hu")(y) for _, r in m.iterrows()]
        m[f"occ{y}"] = [(r[f"p{y}"] / r[f"hu{y}"]) / (linear_trend(r, "p")(y) / linear_trend(r, "hu")(y)) - 1 for _, r in m.iterrows()]
    return m


def nola_series():
    """NOLA metro population and housing units relative to their 2001-05 linear trends, 1 July 2006-2009."""
    r = metros_all().loc[NOLA]; tp, th = linear_trend(r, "p"), linear_trend(r, "hu")
    yrs = [2006, 2007, 2008, 2009]
    return dict(years=yrs, t=[JULY[y] for y in yrs],
                pop_rel=[r[f"p{y}"] / tp(y) for y in yrs], hu_rel=[r[f"hu{y}"] / th(y) for y in yrs],
                occ=[(r[f"p{y}"] / r[f"hu{y}"]) / (tp(y) / th(y)) for y in yrs])
