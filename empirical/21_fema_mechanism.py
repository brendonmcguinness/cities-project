"""Mechanism check: did displaced households in counties with higher tolerated occupancy stay in the county?

OpenFEMA 'Individual Assistance Housing Registrants - Large Disasters' records, for every registrant, whether the
home was destroyed or needed repairs to be habitable, whether the household received rental assistance, and the ZIP
code of the rental resource it used. For each struck county in the held-out panel covered by the dataset (Harvey,
Irma, Florence, Michael, Laura; 6 counties) we compute, for primary-residence registrants:
  displaced  share whose home was destroyed or needed habitability repairs
  rental     share receiving FEMA rental assistance
  local      among rental-assistance households with a recorded rental ZIP, the share renting inside the county
  tsa        share checked into FEMA-paid hotels (Transitional Sheltering Assistance)
and compare 'local' with phi* (prediction: higher where displaced households stayed). ZIP codes are assigned to the
county holding most of the ZIP code tabulation area's population (Census 2010 ZCTA-county relationship file).
Writes output/fema_mechanism.json. With six counties this is descriptive."""
import io, json, time, urllib.parse, urllib.request
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from common import HERE, OUT

RAW = HERE / "raw" / "fema"; RAW.mkdir(parents=True, exist_ok=True)
API = "https://www.fema.gov/api/open/v1/IndividualAssistanceHousingRegistrantsLargeDisasters"
FIELDS = "censusBlockId,damagedZipCode,primaryResidence,destroyed,habitabilityRepairsRequired,rentalAssistanceEligible,rentalResourceZipCode,rentalResourceCity,tsaCheckedIn,ownRent"
COUNTIES = [(4332, "48007", "Aransas", "census", "Harvey"), (4332, "48245", "Jefferson TX", "acs", "Harvey 2017"),
            (4337, "12087", "Monroe FL", "acs", "Irma 2017"), (4393, "37133", "Onslow NC", "acs", "Florence 2018"),
            (4399, "12005", "Bay FL", "acs", "Michael 2018"), (4559, "22019", "Calcasieu LA", "acs", "Laura 2020")]


def registrants(d, fips):
    f = RAW / f"registrants_{d}_{fips}.csv"
    if f.exists():
        return pd.read_csv(f, dtype={"censusBlockId": str, "damagedZipCode": str, "rentalResourceZipCode": str})
    rows, skip = [], 0
    q = urllib.parse.quote(f"disasterNumber eq {d} and startswith(censusBlockId,'{fips}')")
    while True:
        for a in range(6):
            try:
                j = json.load(urllib.request.urlopen(f"{API}?$filter={q}&$select={FIELDS}&$top=10000&$skip={skip}&$format=json", timeout=300)); break
            except Exception:
                time.sleep(15 * (a + 1))
        b = j["IndividualAssistanceHousingRegistrantsLargeDisasters"]; rows += b
        if len(b) < 10000: break
        skip += 10000
    df = pd.DataFrame(rows); df.to_csv(f, index=False); return df


rel = pd.read_csv(io.BytesIO(urllib.request.urlopen("https://www2.census.gov/geo/docs/maps-data/data/rel/zcta_county_rel_10.txt").read()),
                  dtype={"ZCTA5": str, "STATE": str, "COUNTY": str})
rel["fips"] = rel.STATE + rel.COUNTY
zip2county = rel.sort_values("ZPOPPCT", ascending=False).drop_duplicates("ZCTA5").set_index("ZCTA5").fips
cen = {e["name"]: e for e in json.load(open(OUT / "events.json"))["events"]}
acs = json.load(open(OUT / "acs_panel.json"))["events"]
rows = []
for d, fips, name, panel, dis in COUNTIES:
    r = registrants(d, fips); r = r[r.primaryResidence.astype(str) == "True"]
    disp = (r.destroyed.astype(str) == "True") | (r.habitabilityRepairsRequired.astype(str) == "True")
    ra = r[r.rentalAssistanceEligible.astype(str) == "True"]
    z = r.rentalResourceZipCode.dropna().astype(str).str.strip(); z = z[z.str.len() >= 5].str[:5]   # mostly blank in the data
    zc = z.map(zip2county)
    phis = cen[name]["B"]["phis"] if panel == "census" else next(e for e in acs if e["fips"] == fips and e["disaster"] == dis)["B"]["phis"]
    rows.append(dict(county=name, disaster=dis, fips=fips, registrants=int(len(r)), displaced=float(disp.mean()),
                     rental_assist=float(len(ra) / len(r)), rental_zip_recorded=int(zc.notna().sum()), rental_zip_coverage=float(zc.notna().sum() / max(len(ra), 1)),
                     local=float((zc == fips).mean()) if zc.notna().any() else None, tsa=float((r.tsaCheckedIn.astype(str) == "True").mean()),
                     phis=float(phis)))
    print(rows[-1], flush=True)
R = pd.DataFrame(rows)
ok = R[R.rental_zip_coverage >= 0.3].dropna(subset=["local"])   # rental location is recorded for <1% of recipients in most events
sr = spearmanr(ok.local, ok.phis) if len(ok) > 2 else None
from scipy.stats import spearmanr as _sp
res = dict(counties=rows, spearman_local_phis=None if sr is None else dict(rho=float(sr[0]), p=float(sr[1]), n=len(ok)),
           spearman_tsa_phis=dict(zip(("rho", "p"), map(float, _sp(R.tsa, R.phis)))), spearman_rental_phis=dict(zip(("rho", "p"), map(float, _sp(R.rental_assist, R.phis)))),
           spearman_displaced_phis=dict(zip(("rho", "p"), map(float, _sp(R.displaced, R.phis)))))
print(R.round(3).to_string(index=False)); print({k: v for k, v in res.items() if k != "counties"})
json.dump(res, open(OUT / "fema_mechanism.json", "w"), indent=1)
