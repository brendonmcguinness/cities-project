"""Build every processed input in data/ from the raw files in raw/ (see 00_download.py).

Outputs (pipe-separated text):
  data/counties.psv            all US counties: population and housing units 2000-2009 (intercensal), coordinates
  data/metros.psv              all 366 metros (2009 delineation): population, housing units, components of change, centroid
  data/county_metro.psv        county -> metro crosswalk
  data/irs_county_flows.psv.gz every county-to-county IRS flow (exemptions, returns), filing years 2003-04 .. 2007-08
  data/irs_nola_totals.psv     NOLA-metro totals including flows below the disclosure threshold
Checks: the NOLA county flows and metro populations are compared with the files extracted by hand earlier
(data/legacy/), and the run fails if they differ.
"""
import io, re, zipfile, gzip
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
RAW, DATA = HERE / "raw", HERE / "data"
YEARS = ["0304", "0405", "0506", "0607", "0708"]
NOLA_PARISHES = ["22051", "22071", "22075", "22087", "22089", "22095", "22103"]


def csv(name, **kw):
    return pd.read_csv(RAW / name, encoding="latin1", **kw)


# ---------------------------------------------------------------- counties
pop = csv("co-est00int-tot.csv", dtype={"STATE": str, "COUNTY": str})
pop = pop[pop.SUMLEV == 50].copy(); pop["fips"] = pop.STATE.str.zfill(2) + pop.COUNTY.str.zfill(3)
hu = csv("hu-est00int-tot.csv", dtype={"STATE": str, "COUNTY": str, "SUMLEV": str})
hu = hu[hu.SUMLEV == "050"].copy(); hu["fips"] = hu.STATE.str.zfill(2) + hu.COUNTY.str.zfill(3)
z = zipfile.ZipFile(RAW / "Gaz_counties_national.zip")
gaz = pd.read_csv(io.BytesIO(z.read(z.namelist()[0])), sep="\t", dtype={"GEOID": str}, encoding="latin1")
gaz.columns = [c.strip() for c in gaz.columns]
cty = pop[["fips", "CTYNAME", "STNAME"] + [f"POPESTIMATE{y}" for y in range(2000, 2010)]].rename(
    columns={f"POPESTIMATE{y}": f"p{y}" for y in range(2000, 2010)} | {"CTYNAME": "name", "STNAME": "state"})
cty = cty.merge(hu[["fips"] + [f"HUEST_{y}" for y in range(2000, 2010)]].rename(columns={f"HUEST_{y}": f"hu{y}" for y in range(2000, 2010)}), on="fips", how="left")
cty = cty.merge(gaz[["GEOID", "INTPTLAT", "INTPTLONG"]].rename(columns={"GEOID": "fips", "INTPTLAT": "lat", "INTPTLONG": "lon"}), on="fips", how="left")
assert cty.lat.notna().all() and cty.hu2005.notna().all(), "county join failed"

# ---------------------------------------------------------------- metros (2009 delineation)
cb = csv("cbsa-est2009-alldata.csv", dtype={"CBSA": str, "STCOU": str})
mrow = cb[cb.LSAD == "Metropolitan Statistical Area"].copy()
crow = cb[(cb.LSAD == "County or equivalent") & cb.CBSA.isin(mrow.CBSA)].copy()
crow["fips"] = crow.STCOU.str.zfill(5)
xw = crow[["fips", "CBSA"]].drop_duplicates().rename(columns={"CBSA": "code"})
assert xw.fips.isin(cty.fips).all(), "crosswalk county missing"
c2 = cty.merge(xw, on="fips")
agg = c2.groupby("code")[[f"p{y}" for y in range(2000, 2010)] + [f"hu{y}" for y in range(2000, 2010)]].sum()
w = c2.p2005
c2["wlat"], c2["wlon"] = c2.lat * w, c2.lon * w
cen = c2.groupby("code")[["wlat", "wlon", "p2005"]].sum()
agg["lat"], agg["lon"] = cen.wlat / cen.p2005, cen.wlon / cen.p2005
comp = mrow.set_index("CBSA")
keep = {"NAME": "name"}
for y in range(2001, 2010):
    keep |= {f"NATURALINC{y}": f"nat{y}", f"DOMESTICMIG{y}": f"dm{y}", f"INTERNATIONALMIG{y}": f"im{y}"}
met = agg.join(comp[list(keep)].rename(columns=keep))
met.index.name = "code"
met["natinc_rate_0205"] = sum(met[f"nat{y}"] / met[f"p{y-1}"] for y in (2002, 2003, 2004, 2005)) / 4

# ---------------------------------------------------------------- IRS county-to-county flows
def parse_dat(text):
    rows = []
    for l in text.splitlines():
        if len(l) < 60 or not l[:2].strip().isdigit(): continue
        nums = l[50:].split()
        if len(nums) < 2: continue
        rows.append((l[0:2], l[3:6], l[7:9], l[10:13], int(nums[0]), int(nums[1])))
    return rows

def parse_xls(data):
    import xlrd
    s = xlrd.open_workbook(file_contents=data).sheet_by_index(0); rows = []
    for i in range(s.nrows):
        v = s.row_values(i)
        if not (isinstance(v[0], str) and v[0].strip().isdigit() and len(v[0].strip()) <= 2): continue
        try: rows.append((v[0].strip().zfill(2), v[1].strip().zfill(3), v[2].strip().zfill(2), v[3].strip().zfill(3), int(v[6]), int(v[7])))
        except (ValueError, TypeError): continue   # suppressed (-1) or blank
    return rows

flows, totals = [], []
for y in YEARS:
    z = zipfile.ZipFile(RAW / f"county{y}.zip"); names = z.namelist()
    outs = [n for n in names if re.fullmatch(r"(countyout\d{4}(us1)?|co\d{4}us)\.dat", n.lower())]
    if outs:
        assert len(outs) == 1, (y, outs)
        rows = parse_dat(z.read(outs[0]).decode("latin1"))
    else:  # 2003-04 ships only per-state spreadsheets
        rows = []
        for n in names:
            if n.lower().endswith("o.xls") or n.lower().endswith("0.xls"):
                if "outflow" in n.lower(): rows += parse_xls(z.read(n))
    df = pd.DataFrame(rows, columns=["os", "oc", "ds", "dc", "returns", "exemptions"])
    df["orig"], df["dest"] = df.os + df.oc, df.ds + df.dc
    # NOLA-metro totals (all destinations incl. suppressed): sum of '97 000' (US total) records of the 7 parishes
    tot_out = df[df.orig.isin(NOLA_PARISHES) & (df.dest.str[:2] == "97") & (df.dc == "000")].exemptions.sum()
    real = df[(df.os.astype(int) <= 56) & (df.ds.astype(int) <= 56) & (df.oc != "000") & (df.dc != "000") & (df.orig != df.dest)]
    real = real[(real.exemptions >= 0) & (real.returns >= 0)].drop_duplicates(["orig", "dest"])
    real = real.assign(year=y)[["year", "orig", "dest", "returns", "exemptions"]]
    flows.append(real)
    totals.append(dict(year=y, nola_out_total=int(tot_out)))
    print(y, "flows:", len(real), "NOLA total out:", tot_out)
F = pd.concat(flows, ignore_index=True)

# NOLA in-migration totals need the inflow files ('97 000' records with destination = parish)
for t in totals:
    y = t["year"]; z = zipfile.ZipFile(RAW / f"county{y}.zip"); names = z.namelist()
    ins = [n for n in names if re.fullmatch(r"(countyin\d{4}(us1)?|ci\d{4}us)\.dat", n.lower())]
    if ins:
        rows = parse_dat(z.read(ins[0]).decode("latin1"))
        df = pd.DataFrame(rows, columns=["ds", "dc", "os", "oc", "returns", "exemptions"])
        t["nola_in_total"] = int(df[(df.ds + df.dc).isin(NOLA_PARISHES) & (df.os == "97") & (df.oc == "000")].exemptions.sum())
    else:  # 2003-04: Louisiana inflow spreadsheet (destination first, then origin)
        n = [n for n in names if "inflow" in n.lower() and n.lower().endswith("lai.xls")][0]
        rows = parse_xls(z.read(n))
        df = pd.DataFrame(rows, columns=["ds", "dc", "os", "oc", "returns", "exemptions"])
        t["nola_in_total"] = int(df[(df.ds + df.dc).isin(NOLA_PARISHES) & (df.os == "97") & (df.oc == "000")].exemptions.sum())

# ---------------------------------------------------------------- write
DATA.mkdir(exist_ok=True)
hdr = "# Built by 00_build_data.py from raw/ (see raw/MANIFEST.txt). Do not edit by hand.\n"
def write(df, name, index=False):
    with open(DATA / name, "w") as f: f.write(hdr); df.to_csv(f, sep="|", index=index)
write(cty, "counties.psv"); write(met.reset_index(), "metros.psv"); write(xw, "county_metro.psv")
write(pd.DataFrame(totals), "irs_nola_totals.psv")
with gzip.open(DATA / "irs_county_flows.psv.gz", "wt") as f: f.write(hdr); F.to_csv(f, sep="|", index=False)

# ---------------------------------------------------------------- verification against hand-extracted files
leg = DATA / "legacy"
old = pd.read_csv(leg / "irs_nola_flows_2004_2008.psv", sep="|", comment="#", dtype={"fips": str}).set_index("fips").fillna(0)
for y in ["0405", "0506", "0607", "0708"]:
    o = F[(F.year == y) & F.orig.isin(NOLA_PARISHES) & ~F.dest.isin(NOLA_PARISHES)].groupby("dest").exemptions.sum()
    i = F[(F.year == y) & F.dest.isin(NOLA_PARISHES) & ~F.orig.isin(NOLA_PARISHES)].groupby("orig").exemptions.sum()
    assert (o.reindex(old.index).fillna(0) == old[f"o{y}"]).all() and abs(o.sum() - old[f"o{y}"].sum()) == 0, f"out {y} mismatch"
    assert (i.reindex(old.index).fillna(0) == old[f"i{y}"]).all() and abs(i.sum() - old[f"i{y}"].sum()) == 0, f"in {y} mismatch"
om = pd.read_csv(leg / "cbsa_metros_2009def.psv", sep="|", comment="#", dtype={"code": str}).set_index("code")
assert (met.loc[om.index, ["p2001", "p2004", "p2005", "p2006", "p2007"]].values == om[["p2001", "p2004", "p2005", "p2006", "p2007"]].values).all(), "metro populations mismatch"
print("verification against hand-extracted files: OK")
print(pd.DataFrame(totals).to_string(index=False))
