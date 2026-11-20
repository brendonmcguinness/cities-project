"""Independent check with direct displacement data: Census Household Pulse Survey (HPS), Phase 3.7 onward.

HPS asks adults "In the past year, were you displaced from your home because of a natural disaster?", the type of
disaster, and how long they were displaced (less than a week; a week to a month; one to six months; more than six
months; never returned). Estimates are state-level, so we use the two events that dominate their state's
displacement in the survey window:
  Florida, Hurricane Ian (28 Sep 2022): weeks 52-60 (Dec 2022 - Aug 2023; Idalia struck 30 Aug 2023), hurricane
  Hawaii, Lahaina fire (8 Aug 2023): weeks 61-63 and Phase 4 cycles 1-7 (Aug 2023 - Aug 2024), fire
Weighted with PWEIGHT; standard errors from the 80 replicate weights (successive difference, 4/80 factor).
The adult counts displaced for more than a month are compared with the housing loss the model uses (ACS usable
units lost in the struck counties times adults per occupied unit) and the duration distribution with the capacity
recovery path. Writes output/pulse.json."""
import io, json, urllib.request, zipfile
import numpy as np
import pandas as pd
from common import HERE, OUT

RAW = HERE / "raw" / "hps"; RAW.mkdir(parents=True, exist_ok=True)
B = "https://www2.census.gov/programs-surveys/demo/datasets/hhp/"
WAVES = {"FL": [f"2023/wk{w}" for w in range(52, 61)],
         "HI": [f"2023/wk{w}" for w in (61, 62, 63)] + [f"2024/cycle{c:02d}" for c in range(1, 8)]}
STFIPS = {"FL": 12, "HI": 15}; TYPE = {"FL": "ND_TYPE1", "HI": "ND_TYPE3"}   # hurricane, fire
DUR = {1: "<1 week", 2: "1 week-1 month", 3: "1-6 months", 4: ">6 months", 5: "never returned"}


def wave(path, st):
    """Rows for one state from one wave: PUF columns needed plus replicate weights."""
    cache = RAW / f"{path.replace('/', '_')}_{st}.csv"
    if cache.exists():
        return pd.read_csv(cache)
    import re
    listing = urllib.request.urlopen(B + path + "/", timeout=120).read().decode("latin-1")
    fn = re.findall(r'href="([^"]*PUF_CSV\.zip)"', listing)[0]
    import time
    z = None
    for a in range(4):
        try:
            z = zipfile.ZipFile(io.BytesIO(urllib.request.urlopen(B + path + "/" + fn, timeout=900).read())); break
        except Exception as e:
            print("retry", path, e, flush=True); time.sleep(60 * (a + 1))
    if z is None:                       # the Census server rejected this file; the wave is skipped and reported
        SKIPPED.append(path); return None
    names = z.namelist(); puf = [n for n in names if n.lower().endswith(".csv") and "puf" in n.lower() and "repwgt" not in n.lower()][0]
    rep = [n for n in names if "repwgt" in n.lower()][0]
    d = pd.read_csv(z.open(puf), usecols=lambda c: c in ("SCRAM", "EST_ST", "PWEIGHT", "ND_DISPLACE", "ND_HOWLONG", "ND_TYPE1", "ND_TYPE3", "TBIRTH_YEAR"))
    d = d[d.EST_ST == STFIPS[st]]
    r = pd.read_csv(z.open(rep)); r = r[r.SCRAM.isin(d.SCRAM)]
    r = r[["SCRAM"] + [c for c in r.columns if c.startswith("PWEIGHT")]]
    d = d.merge(r, on="SCRAM", suffixes=("", "_rep")); d["wave"] = path
    d.to_csv(cache, index=False); return d


def est(d, f):
    """Weighted mean of indicator f (Series of 0/1) over rows d, with replicate SE."""
    w = d["PWEIGHT"]; th = (w * f).sum() / w.sum()
    reps = [c for c in d.columns if c.startswith("PWEIGHT") and c != "PWEIGHT"]
    tr = np.array([(d[c] * f).sum() / d[c].sum() for c in reps])
    return float(th), float(np.sqrt(4 / len(reps) * np.sum((tr - th) ** 2)))


SKIPPED = []
res = {}
for st in ("FL", "HI"):
    D = pd.concat([w for w in (wave(p, st) for p in WAVES[st]) if w is not None], ignore_index=True)
    D = D[D.ND_DISPLACE.isin([1, 2])]
    disp = (D.ND_DISPLACE == 1) & (D[TYPE[st]] == 1)
    out = dict(waves=[p for p in WAVES[st] if p not in SKIPPED], skipped=[p for p in WAVES[st] if p in SKIPPED], respondents=int(len(D)), displaced_respondents=int(disp.sum()))
    out["share_displaced"] = est(D, disp.astype(float))
    long_ = disp & D.ND_HOWLONG.isin([3, 4, 5]); out["share_displaced_over_month"] = est(D, long_.astype(float))
    dd = D[disp & D.ND_HOWLONG.isin(DUR)]
    out["duration"] = {DUR[k]: est(dd, (dd.ND_HOWLONG == k).astype(float)) for k in DUR}
    out["adult_population"] = float(D.groupby("wave").PWEIGHT.sum().mean())   # weighted adults per wave
    out["adults_displaced_over_month"] = out["share_displaced_over_month"][0] * out["adult_population"]
    res[st] = out
    print(st, json.dumps({k: v for k, v in out.items() if k != "waves"}, indent=1), flush=True)

# housing loss the model uses (ACS), in adults: units lost x adults per occupied unit (ACS pop / households x adult share)
acs = pd.read_csv(HERE / "data" / "acs_county.psv", sep="|", comment="#", dtype={"fips": str}).set_index(["fips", "year"])
panel = json.load(open(OUT / "acs_panel.json"))["events"]
cmp = {}
for st, evs in (("FL", [("12071", "Ian 2022"), ("12015", "Ian 2022"), ("12021", "Ian 2022")]), ("HI", [("15009", "Lahaina fire 2023")])):
    lost = 0.0; rows = []
    for f, dis in evs:
        e = next(x for x in panel if x["fips"] == f and x["disaster"] == dis)
        y0 = e["acs_years"][0]; u = acs.loc[(f, y0 - 2), "usable_units"]        # last full pre-event ACS year
        units = max(e["loss"], 0) * u; adults_per_unit = 0.78 * acs.loc[(f, y0 - 2), "hh_size"]   # 0.78: adult share of household members (ACS, US)
        lost += units * adults_per_unit; rows.append(dict(fips=f, loss=e["loss"], units_lost=float(units), cap_path=e["cap_obs"], years=e["acs_years"]))
    cmp[st] = dict(counties=rows, adults_in_lost_housing=float(lost), hps_adults_displaced_over_month=res[st]["adults_displaced_over_month"],
                   ratio=float(res[st]["adults_displaced_over_month"] / lost) if lost > 0 else None)
    print(st, cmp[st], flush=True)
json.dump(dict(hps=res, comparison=cmp), open(OUT / "pulse.json", "w"), indent=1)
