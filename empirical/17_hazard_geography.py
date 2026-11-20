"""Does pre-disaster hazard geography predict tolerated occupancy?  (Plan B, part 2 and part 3)

Hypothesis: displaced residents stay in the county when much of its housing lies outside the hazard zone.
Measures (defined before computing them; see hazard_geography.js for the in-browser computation):
  coastal storms   share of pre-event housing OUTSIDE the mapped coastal flood hazard =
                   FEMA Special Flood Hazard Area (zones A, V)  OR  NOAA Category 3 storm-surge inundation
                   OR  NOAA 'levee area' (protected low ground the surge model does not resolve)
  inland floods    share outside the FEMA Special Flood Hazard Area
  secondary        share of housing on ground at least 3 m above sea level (coastal storms only)
  fires, tornado   excluded: no pre-event hazard map reachable for the fire counties (pre-2017 CAL FIRE zones and
                   USFS Wildfire Hazard Potential are not served to this workspace), and tornado tracks have none.
Housing: Census 2010 block housing (events from 2010); for 2005-2008 events, Census 2000 block housing allocated to
2010 tracts by land area (Census 2000-2010 block relationship files), distributed within each tract like 2010 housing.
Hazard maps are the current FEMA flood maps (Esri Living Atlas copy of the NFHL) and NOAA's v3 surge hazard maps;
they describe physical exposure, not the pre-event regulatory maps.
Elevation: Mapzen/Tilezen terrarium tiles at zoom 14 (about 9.5 m pixels; USGS 3DEP in the US).
Test: Pearson and Spearman correlation of each measure with phi* (the held-out panel of 14_holdout.py);
then leave-one-disaster-out prediction as in 14_holdout.py, with phi* for the held-out county from a linear
regression of phi* on the primary measure across training counties (fitted with partial pooling).
Inputs: data/hazard_geography.json (written from the browser run). Writes output/hazard_geography.json."""
import csv, collections, io, json, os, sys, urllib.request, zipfile
import numpy as np
from scipy.stats import pearsonr, spearmanr
from common import HERE, OUT



def build_hu00():
    """Census 2000 block housing allocated to 2010 tracts by intersecting land area (pasted into the browser as H.HU00).
    Needs a Census API key (CENSUS_API_KEY or ~/.census_api_key). Writes data/hu00_by_tract10.json."""
    key = os.environ.get("CENSUS_API_KEY") or open(os.path.expanduser("~/.census_api_key")).read().strip()
    out, rel = {}, {}
    for f in ["22071", "22087", "22075", "22051", "28045", "28047", "28059", "22023", "48167"]:
        st, co = f[:2], f[2:]
        if st not in rel:
            z = urllib.request.urlopen(f"https://www2.census.gov/geo/docs/maps-data/data/rel/t00t10/TAB2000_TAB2010_ST_{st}_v2.zip").read()
            txt = zipfile.ZipFile(io.BytesIO(z)).read(f"TAB2000_TAB2010_ST_{st}_v2.txt").decode("latin-1")
            rel[st] = [{k.strip(): (v or "").strip() for k, v in r.items()} for r in csv.DictReader(io.StringIO(txt))]
        a = json.load(urllib.request.urlopen(f"https://api.census.gov/data/2000/dec/sf1?get=H001001&for=block:*&in=state:{st}%20county:{co}%20tract:*&key={key}"))
        hu = {((r[3] + "00") if len(r[3]) == 4 else r[3], r[4]): int(r[0]) for r in a[1:]}
        rows = [r for r in rel[st] if r["COUNTY_2000"] == co]; k_ = lambda r: (r["TRACT_2000"], r["BLK_2000"] + r["BLKSF_2000"])
        land, area = collections.defaultdict(float), collections.defaultdict(float)
        for r in rows:
            land[k_(r)] += float(r["AREALAND_INT"]); area[k_(r)] += float(r["AREALAND_INT"]) + float(r["AREAWATER_INT"] or 0)
        alloc = collections.defaultdict(float)
        for r in rows:
            k = k_(r)
            if k not in hu or r["COUNTY_2010"] != co: continue
            w = float(r["AREALAND_INT"]) / land[k] if land[k] > 0 else ((float(r["AREALAND_INT"]) + float(r["AREAWATER_INT"] or 0)) / area[k] if area[k] > 0 else 0)
            alloc[r["TRACT_2010"]] += hu[k] * w
        out[f] = {t: round(v) for t, v in sorted(alloc.items())}
    json.dump(out, open(HERE / "data" / "hu00_by_tract10.json", "w"), separators=(",", ":"))


if "--build-hu00" in sys.argv:
    build_hu00()
G = json.load(open(HERE / "data" / "hazard_geography.json"))
acs = json.load(open(OUT / "acs_panel.json"))["events"]
rows = []
for g in G["counties"]:
    if g["panel"] == "census":
        e = next(x for x in json.load(open(OUT / "events.json"))["events"] if x["name"] == g["name"])
        phis, loss = e["B"]["phis"], e["shock"]
    else:
        e = next(x for x in acs if x["fips"] == g["fips"] and x["disaster"] == g["disaster"])
        phis, loss = e["B"]["phis"], e["loss"]
    if g["panel"] == "acs" and loss < 0.04:      # same panel as 14_holdout.py
        continue
    rows.append(dict(g, phis=phis, loss=loss))
res = dict(n=len(rows), events=rows, tests={})
for m in ("outside_primary", "outside_sfha", "above3m"):
    sub = [r for r in rows if r.get(m) is not None]
    x, y = np.array([r[m] for r in sub]), np.array([r["phis"] for r in sub])
    pr, sr = pearsonr(x, y), spearmanr(x, y)
    res["tests"][m] = dict(n=len(sub), pearson=float(pr[0]), p_pearson=float(pr[1]), spearman=float(sr[0]), p_spearman=float(sr[1]),
                           slope=float(np.polyfit(x, y, 1)[0]))
    print(f"{m:16s} n={len(sub):2d}  r={pr[0]:+.2f} (p={pr[1]:.2f})  rho={sr[0]:+.2f} (p={sr[1]:.2f})")
for r in sorted(rows, key=lambda r: r["outside_primary"]):
    print(f"{r['name']:22s} {r.get('disaster', ''):18s} outside={r['outside_primary']:.2f} above3m={r['above3m'] if r['above3m'] is None else round(r['above3m'], 2)}  phi*={r['phis']:.3f}")
# sensitivity: phi* from survey-measured capacity for the Katrina counties the ACS covers (23_capacity_check.py)
cc = OUT / "capacity_check.json"
if cc.exists():
    CC = json.load(open(cc))["capacity_check"]; res["tests_acs_capacity"] = {}
    for m in ("outside_primary", "above3m"):
        sub = [r for r in rows if r.get(m) is not None]
        y = np.array([CC[r["name"]]["acs"]["phis"] if (r["panel"] == "census" and r["name"] in CC) else r["phis"] for r in sub]); x = np.array([r[m] for r in sub])
        sr, pr = spearmanr(x, y), pearsonr(x, y)
        res["tests_acs_capacity"][m] = dict(n=len(sub), spearman=float(sr[0]), p_spearman=float(sr[1]), pearson=float(pr[0]), p_pearson=float(pr[1]))
        print(f"ACS-capacity variant {m:16s} rho={sr[0]:+.2f} (p={sr[1]:.3f})  r={pr[0]:+.2f} (p={pr[1]:.2f})")
json.dump(res, open(OUT / "hazard_geography.json", "w"), indent=1)
