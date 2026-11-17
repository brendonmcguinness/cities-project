"""Download all raw inputs from their official sources into empirical/raw/ and record SHA-256 checksums.

Run once:  python3 00_download.py      (about 75 MB; raw/ is not committed, raw/MANIFEST.txt is)
"""
import hashlib, sys, urllib.request
from pathlib import Path

RAW = Path(__file__).resolve().parent / "raw"
RAW.mkdir(exist_ok=True)
IRS = "https://www.irs.gov/pub/irs-soi/"
CEN = "https://www2.census.gov/"
SOURCES = {
    # IRS Statistics of Income, county-to-county migration (filing-year pairs)
    "county0304.zip": IRS + "2003to2004countymigration.zip",
    "county0405.zip": IRS + "county0405.zip",
    "county0506.zip": IRS + "county0506.zip",
    "county0607.zip": IRS + "county0607.zip",
    "county0708.zip": IRS + "county0708.zip",
    # Census population estimates (intercensal counties; Vintage 2009 metros with components of change)
    "co-est00int-tot.csv": CEN + "programs-surveys/popest/datasets/2000-2010/intercensal/county/co-est00int-tot.csv",
    "hu-est00int-tot.csv": CEN + "programs-surveys/popest/datasets/2000-2010/intercensal/housing/hu-est00int-tot.csv",
    "cbsa-est2009-alldata.csv": CEN + "programs-surveys/popest/datasets/2000-2009/metro/totals/cbsa-est2009-alldata.csv",
    # Second event (Camp Fire, Butte County CA, Nov 2018): IRS 2016-17 .. 2020-21 and Census Vintage 2020
    **{f"countyoutflow{y}.csv": IRS + f"countyoutflow{y}.csv" for y in ["1617", "1718", "1819", "1920", "2021"]},
    **{f"countyinflow{y}.csv": IRS + f"countyinflow{y}.csv" for y in ["1617", "1718", "1819", "1920", "2021"]},
    "co-est2020-alldata.csv": CEN + "programs-surveys/popest/datasets/2010-2020/counties/totals/co-est2020-alldata.csv",
    "HU-EST2020_6.csv": CEN + "programs-surveys/popest/datasets/2010-2020/housing/HU-EST2020_6.csv",
    # Census 2010 Gazetteer (county internal points)
    "Gaz_counties_national.zip": CEN + "geo/docs/maps-data/data/gazetteer/Gaz_counties_national.zip",
}


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""): h.update(b)
    return h.hexdigest()


def main():
    lines = []
    for name, url in SOURCES.items():
        dest = RAW / name
        if not dest.exists():
            print("downloading", url, file=sys.stderr)
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (research; cities-project)"})
            with urllib.request.urlopen(req, timeout=600) as r, open(dest, "wb") as f:
                f.write(r.read())
        lines.append(f"{sha256(dest)}  {dest.stat().st_size:>10d}  {name}  {url}")
    (RAW / "MANIFEST.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
