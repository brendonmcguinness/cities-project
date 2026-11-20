"""Where did evacuees go? Destination test beyond Katrina with IRS county-to-county flows.

For each struck county of the held-out panel with released IRS flows for the event filing year, excess
out-migration to each destination county = flow in the event filing year minus the mean of the two preceding years
(exemptions, n2; flows below the IRS disclosure threshold are not reported and count as zero). In the model,
evacuees leave along the migration network in proportion to its weights (Methods), so destination shares should
follow (a) the county's own pre-disaster flow shares; simpler alternatives are gravity kernels P_j^b / d^g with
(b) the national metro estimates (b = 0.90, g = 1.12) and (c) the kernel fitted to New Orleans' county-level baseline
out-migration (b = 1.42, g = 2.16), both from Section 4.2. Agreement is the squared Pearson correlation (R^2) between predicted and observed shares across all
destination counties with a reported flow in any of the three years, and the share of excess out-migration going
to the ten largest predicted destinations. Same-state and out-of-state destinations both count.
Writes output/destinations.json."""
import io, json, zipfile
import numpy as np
import pandas as pd
from common import HERE, OUT

RAW = HERE / "raw"
EVENTS = [("Butte", "06007", "Camp Fire", 2018), ("Bay", "12005", "Michael", 2018), ("Calcasieu", "22019", "Laura", 2020),
          ("Monroe", "12087", "Irma", 2017), ("Onslow", "37133", "Florence", 2018), ("Jefferson TX", "48245", "Harvey", 2017),
          ("Aransas", "48007", "Harvey", 2017)]
gz = pd.read_csv(io.BytesIO(zipfile.ZipFile(RAW / "Gaz_counties_national.zip").read("Gaz_counties_national.txt")), sep="\t", dtype={"GEOID": str}, encoding="latin1")
gz.columns = [c.strip() for c in gz.columns]; gz = gz.set_index("GEOID")
pop = pd.read_csv(RAW / "co-est2020-alldata.csv", encoding="latin1", dtype={"STATE": str, "COUNTY": str})
pop = pop[pop.SUMLEV == 50].assign(f=lambda d: d.STATE + d.COUNTY).set_index("f").POPESTIMATE2015


def flows(fy, fips):
    d = pd.read_csv(RAW / f"countyoutflow{str(fy)[2:]}{str(fy + 1)[2:]}.csv", encoding="latin1", dtype=str)
    d = d[(d.y1_statefips.str.zfill(2) + d.y1_countyfips.str.zfill(3)) == fips]
    d = d[~d.y2_statefips.isin(["96", "97", "98", "57", "58", "59"])]
    d["dest"] = d.y2_statefips.str.zfill(2) + d.y2_countyfips.str.zfill(3)
    d = d[d.dest != fips]
    return pd.to_numeric(d.set_index("dest").n2, errors="coerce").groupby(level=0).sum()


def hav(a, b):
    la1, lo1, la2, lo2 = map(np.radians, (a[0], a[1], b[:, 0], b[:, 1]))
    return 6371 * 2 * np.arcsin(np.sqrt(np.sin((la2 - la1) / 2) ** 2 + np.cos(la1) * np.cos(la2) * np.sin((lo2 - lo1) / 2) ** 2))


res = []
for name, fips, dis, fy in EVENTS:
    f0, f1, f2 = flows(fy - 2, fips), flows(fy - 1, fips), flows(fy, fips)
    D = sorted(set(f0.index) | set(f1.index) | set(f2.index))
    D = [x for x in D if x in gz.index and x in pop.index]
    base = (f0.reindex(D).fillna(0) + f1.reindex(D).fillna(0)) / 2; ex = (f2.reindex(D).fillna(0) - base).clip(lower=0)
    if ex.sum() <= 0:
        continue
    o = ex / ex.sum(); a = base / base.sum()
    dist = hav((gz.loc[fips, "INTPTLAT"], gz.loc[fips, "INTPTLONG"]), gz.loc[D, ["INTPTLAT", "INTPTLONG"]].values.astype(float))
    P = pop.reindex(D).values; dd = np.maximum(dist, 10)
    g = pd.Series(P ** 0.90 / dd ** 1.12, index=D); g /= g.sum()
    gN = pd.Series(P ** 1.42 / dd ** 2.16, index=D); gN /= gN.sum()
    r2 = lambda p: float(np.corrcoef(p, o)[0, 1] ** 2)
    top = lambda p: float(o[p.sort_values(ascending=False).index[:10]].sum())
    within300 = float(o[dist <= 300].sum()); within300_base = float(a[dist <= 300].sum())
    res.append(dict(county=name, fips=fips, disaster=dis, filing_year=f"{fy}-{str(fy + 1)[2:]}", destinations=len(D), excess=float(ex.sum()),
                    baseline=float(base.sum()), r2_baseline_shares=r2(a), r2_gravity=r2(g), r2_gravity_no=r2(gN), top10_baseline=top(a), top10_gravity=top(g),
                    excess_within_300km=within300, baseline_within_300km=within300_base, largest=o.idxmax(), largest_share=float(o.max())))
    print(res[-1], flush=True)
summ = {k: float(np.median([r[k] for r in res])) for k in ("r2_baseline_shares", "r2_gravity", "r2_gravity_no", "top10_baseline", "top10_gravity")}
print(summ)
json.dump(dict(events=res, median=summ), open(OUT / "destinations.json", "w"), indent=1)
