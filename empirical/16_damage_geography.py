"""Does the geography of damage inside the struck county predict tolerated occupancy?

For each event with a FEMA Individual Assistance declaration, valid owner and renter registrations per ZIP code
(OpenFEMA HousingAssistanceOwners / HousingAssistanceRenters, v2 API) are divided by ZIP code tabulation area housing
units (Census 2000 gazetteer for 2005 events, 2010 gazetteer otherwise). Measures of how much housing nearby was
spared: share of county units in ZIPs with fewer than 10/20/30 registrations per 100 units; overall registration
rate; concentration (share of ZIPs holding half of all registrations). Compared with phi* from 10_events.py and
13_acs_panel.py for events losing at least 4% of capacity. Writes output/damage_geography.json."""
import io, json, time, urllib.parse, urllib.request, zipfile
import numpy as np
import pandas as pd
from common import HERE, OUT

RAW = HERE / "raw"; FEMA = RAW / "fema"; FEMA.mkdir(exist_ok=True)
DECL = {  # (panel, county) -> (FEMA disaster number, FEMA county name)
    ("census", "Orleans"): (1603, "Orleans (Parish)"), ("census", "St Bernard"): (1603, "St. Bernard (Parish)"),
    ("census", "Plaquemines"): (1603, "Plaquemines (Parish)"), ("census", "Jefferson"): (1603, "Jefferson (Parish)"),
    ("census", "Hancock"): (1604, "Hancock (County)"), ("census", "Harrison"): (1604, "Harrison (County)"),
    ("census", "Jackson"): (1604, "Jackson (County)"), ("census", "Cameron"): (1607, "Cameron (Parish)"),
    ("census", "Aransas"): (4332, "Aransas (County)"), ("census", "Butte"): (4407, "Butte (County)"),
    ("Ike 2008", "Galveston TX"): (1791, "Galveston (County)"), ("Tuscaloosa tornado 2011", "Tuscaloosa AL"): (1971, "Tuscaloosa (County)"),
    ("Louisiana floods 2016", "East Baton Rouge LA"): (4277, "East Baton Rouge (Parish)"), ("Matthew 2016", "Robeson NC"): (4285, "Robeson (County)"),
    ("Harvey 2017", "Jefferson TX"): (4332, "Jefferson (County)"), ("Irma 2017", "Monroe FL"): (4337, "Monroe (County)"),
    ("Tubbs fire 2017", "Napa CA"): (4344, "Napa (County)"), ("Florence 2018", "Onslow NC"): (4393, "Onslow (County)"),
    ("Michael 2018", "Bay FL"): (4399, "Bay (County)"), ("Laura 2020", "Calcasieu LA"): (4559, "Calcasieu (Parish)"),
    ("Lahaina fire 2023", "Maui HI"): (4724, "Maui (County)"),
}


def fema(dataset, d, c):
    f = FEMA / f"{dataset}_{d}_{c.split(' (')[0].replace(' ', '_').replace('.', '')}.csv"
    if f.exists():
        return pd.read_csv(f, dtype={"zipCode": str})
    rows, skip = [], 0
    while True:
        q = urllib.parse.quote(f"disasterNumber eq {d} and county eq '{c}'")
        for attempt in range(5):
            try:
                j = json.load(urllib.request.urlopen(f"https://www.fema.gov/api/open/v2/{dataset}?$filter={q}&$top=1000&$skip={skip}", timeout=120)); break
            except Exception:
                time.sleep(10)
        b = j[dataset]; rows += b
        if len(b) < 1000: break
        skip += 1000
    df = pd.DataFrame(rows); df.to_csv(f, index=False); return df


def hu00():
    s = {}
    for l in zipfile.ZipFile(RAW / "zcta5.zip").read("zcta5.txt").decode("latin-1").splitlines():
        t = l[66:].split()
        if len(t) >= 2 and l[2:7].strip().isdigit(): s[l[2:7]] = int(t[1])
    return pd.Series(s)


def hu10():
    z = pd.read_csv(io.BytesIO(zipfile.ZipFile(RAW / "Gaz_zcta_national.zip").read("Gaz_zcta_national.txt")), sep="\t", dtype={"GEOID": str})
    z.columns = [c.strip() for c in z.columns]; return z.set_index("GEOID").HU10


H00, H10 = hu00(), hu10()
ev = [(("census", e["name"]), e["B"]["phis"], e["shock"]) for e in json.load(open(OUT / "events.json"))["events"]]
ev += [((e["disaster"], e["name"]), e["B"]["phis"], e["loss"]) for e in json.load(open(OUT / "acs_panel.json"))["events"]]
rows = []
for key, phis, loss in ev:
    if key not in DECL or loss < 0.04: continue
    d, c = DECL[key]
    reg = pd.concat([fema(k, d, c) for k in ("HousingAssistanceOwners", "HousingAssistanceRenters")]).groupby("zipCode").validRegistrations.sum()
    hu = (H00 if d < 2000 else H10).reindex(reg.index); m = hu > 100; reg, hu = reg[m], hu[m]
    rate, tot = reg / hu, hu.sum(); cum = (rate * hu).sort_values(ascending=False).cumsum()
    rows.append(dict(panel=key[0], county=key[1], loss=float(loss), phis=float(phis), zips=int(len(reg)), reg_rate=float(reg.sum() / tot),
                     spared10=float(hu[rate < 0.10].sum() / tot), spared20=float(hu[rate < 0.20].sum() / tot),
                     spared30=float(hu[rate < 0.30].sum() / tot), concentration=float((cum.searchsorted(0.5 * cum.iloc[-1]) + 1) / len(reg))))
R = pd.DataFrame(rows)
corr = {c: dict(pearson=float(np.corrcoef(R[c], R.phis)[0, 1]), spearman=float(R[[c, "phis"]].corr("spearman").iloc[0, 1]))
        for c in ("reg_rate", "spared10", "spared20", "spared30", "concentration", "loss")}
print(R.round(3).to_string(index=False)); print(json.dumps(corr, indent=1))
json.dump(dict(events=rows, correlations=corr, n=len(R)), open(OUT / "damage_geography.json", "w"), indent=1)
