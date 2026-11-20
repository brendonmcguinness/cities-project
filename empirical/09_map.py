"""Fig. 1: where New Orleans evacuees went (IRS excess out-migration 2005-06) and where returnees came from (2006-07),
on the 196-metro network. State outlines: Census cartographic boundary file cb_2018_us_state_20m (raw/)."""
import io
import zipfile
import numpy as np
import matplotlib.pyplot as plt
import shapefile
import matplotlib.patheffects as pe
HALO = [pe.withStroke(linewidth=1.6, foreground="white")]
from pyproj import Transformer
from common import OUT, NOLA, metros
import figstyle

RAW = OUT.parent / "raw"
figstyle.apply()
proj = Transformer.from_crs("EPSG:4269", "EPSG:5070", always_xy=True)   # NAD83 -> CONUS Albers equal area


def states():
    z = zipfile.ZipFile(RAW / "cb_2018_us_state_20m.zip")
    get = lambda ext: io.BytesIO(z.read(f"cb_2018_us_state_20m.{ext}"))
    r = shapefile.Reader(shp=get("shp"), shx=get("shx"), dbf=get("dbf"))
    out = []
    for rec, shp in zip(r.records(), r.shapes()):
        if rec["STUSPS"] in ("AK", "HI", "PR"):
            continue
        pts, parts = np.asarray(shp.points), list(shp.parts) + [len(shp.points)]
        for a, b in zip(parts[:-1], parts[1:]):
            out.append(np.column_stack(proj.transform(pts[a:b, 0], pts[a:b, 1])))
    return out


m = metros().copy()
m["x"], m["y"] = proj.transform(m.lon.values, m.lat.values)
x0, y0 = m.loc[NOLA, ["x", "y"]]
hosts = m.drop(index=NOLA)
hosts = hosts[(hosts.lon > -125) & (hosts.lat > 24) & (hosts.lat < 50)]   # contiguous US only (drops Anchorage, Honolulu)
outlines = states()

fig, axes = plt.subplots(1, 2, figsize=(figstyle.W, 2.55))
panels = [(hosts.excess.clip(lower=0), "a", "evacuees, 2005–06"), (hosts.excess_ret.clip(lower=0), "b", "returnees, 2006–07")]
smax = 40000.0
for ax, (flow, letter, what) in zip(axes, panels):
    for xy in outlines:
        ax.fill(xy[:, 0], xy[:, 1], facecolor="0.95", edgecolor="0.7", lw=0.3, zorder=0)
    ax.scatter(hosts.x, hosts.y, s=0.8, color="0.5", lw=0, zorder=1)
    f = flow[flow > 0].sort_values()
    for c, v in f.items():
        ax.plot([x0, hosts.x[c]], [y0, hosts.y[c]], color="tab:red", lw=0.1 + 1.4 * v / smax, alpha=0.25, zorder=2,
                solid_capstyle="round")
    ax.scatter(hosts.x[f.index], hosts.y[f.index], s=180 * f / smax, facecolor=figstyle.ACC, edgecolor="#67000d",
               lw=0.3, alpha=0.65, zorder=3)
    ax.plot(x0, y0, marker="*", color="k", ms=6, zorder=4)
    for c in f.index[-4:]:
        nm = hosts.name[c].split("-")[0].split(",")[0]
        r = np.sqrt(180 * f[c] / smax / np.pi) + 1.5          # circle radius in points, plus a gap
        dx, dy, ha, va = {"Baton Rouge": (0, r, "center", "bottom"), "Atlanta": (r, 0, "left", "center")}.get(nm, (-r, 0, "right", "center"))
        ax.annotate(nm, (hosts.x[c], hosts.y[c]), xytext=(dx, dy), textcoords="offset points", fontsize=6, ha=ha, va=va, zorder=5, path_effects=HALO)
    ax.annotate("New Orleans", (x0, y0), xytext=(5, -7), textcoords="offset points", fontsize=6, zorder=5, path_effects=HALO)
    # size key, plain
    kx, ky = -2.3e6, 0.5e6
    for i, v in enumerate([1000, 10000, 40000]):
        ax.scatter(kx + i * 4.4e5, ky, s=180 * v / smax, facecolor=figstyle.ACC, edgecolor="#67000d", lw=0.3, alpha=0.65)
        ax.text(kx + i * 4.4e5, ky - 2.0e5, f"{v:,}", ha="center", fontsize=5.5)
    ax.text(kx - 1.0e5, ky + 2.0e5, "migrants", fontsize=6)
    ax.set_xlim(-2.45e6, 2.3e6); ax.set_ylim(0.25e6, 3.25e6)
    ax.set_aspect("equal"); ax.axis("off")
    ax.text(0.0, 1.0, letter, transform=ax.transAxes, fontsize=8, fontweight="bold", va="top")
fig.tight_layout()
fig.savefig(OUT / "fig_map.png"); fig.savefig(OUT / "fig_map.pdf")
print("map written; destinations with flow:", int((panels[0][0] > 0).sum()), "origins with flow:", int((panels[1][0] > 0).sum()))
