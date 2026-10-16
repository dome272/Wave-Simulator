"""Cut a real California beach into Shorebreak Lab's seabed grid.

The grid is 256 x 256 nodes at 2.5 m, rotated so the beach faces row 0 (the
offshore edge), shifted from NAVD88 to mean sea level, with an estimated rock
mask, and written in the app's saved-beach format (beaches/<id>.json).

Seabed sources, both public domain and read straight from NOAA's public S3
mirror (only the window needed is fetched):
  * USGS CoNED topobathymetric model (seamless land + sea, the base);
  * USACE NCMP 2014 California topobathy lidar, which actually surveyed the
    surf zone at many rocky sites where CoNED only interpolates. Where the
    lidar has data it wins; its gaps are filled by blending its difference
    from CoNED smoothly back to zero.

Usage: python3 tools/cut_beach.py <beach id> [--preview out.png]
Needs: rasterio, numpy, scipy (and matplotlib for --preview).
GDAL needs the proxy CA bundle in cloud sessions: CURL_CA_BUNDLE=... GDAL_CURL_CA_BUNDLE=...
"""
import json, math, sys, base64, os
import numpy as np
import rasterio
from rasterio.vrt import WarpedVRT
from rasterio.warp import transform
from rasterio.transform import from_origin
from rasterio.enums import Resampling
from scipy import ndimage, sparse
from scipy.sparse.linalg import spsolve

N, DX = 256, 2.5
EXT = (N - 1) * DX
S3 = "/vsicurl/https://noaa-nos-coastal-lidar-pds.s3.amazonaws.com/dem/"
CONED_CENTRAL = S3 + "CA_Central_CoNED_DEM_2017_8657/CA_Central_CoNED_DEM_2017_m8657_EPSG-3717.vrt"
CONED_SOUTH = S3 + "CA_Southern_CoNED_DEM_2016_8658/CA_Southern_CoNED_DEM_2016_m8658_EPSG-26911.vrt"
LIDAR_2014 = S3 + "USACE_California_Topobathy_DEM_2014_5189/USACE_California_Topobathy_DEM_2014_1.vrt"

# anchor: lon/lat of the waterline at the middle of the beach; facing: compass
# bearing the beach looks out to sea along (row 0 lies that way); shore_frac:
# where the anchor sits between row 0 and the last row; msl: mean sea level
# above NAVD88 at the nearest tide gauge (NOAA datums).
BEACHES = {
    "davenport-landing": dict(
        name="Davenport Landing", place="Santa Cruz County, CA",
        anchor=(-122.21560, 37.02390), facing=235, shore_frac=0.80, along=0,
        coned=CONED_CENTRAL, utm="EPSG:26910", msl=0.85, gauge="Santa Cruz 9413745 / Monterey 9413450",
        shelf_rock=True, sand_radius=110,  # mudstone reef shelves either side of a sand cove
        swell=dict(height=1.8, period=14, bearing=285),
    ),
    "trestles": dict(
        name="Lower Trestles", place="San Clemente, CA",
        anchor=(-117.5895, 33.3830), facing=217, shore_frac=0.80, along=0,
        coned=CONED_SOUTH, utm="EPSG:26911", msl=0.80, gauge="La Jolla 9410230",
        swell=dict(height=1.5, period=15, bearing=200),
    ),
}


def read_window(url, crs, x0, y0, size, cell):
    n = int(round(size / cell))
    tr = from_origin(x0 - size / 2, y0 + size / 2, cell, cell)
    with rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR", GDAL_HTTP_MULTIRANGE="YES", VSI_CACHE="TRUE"):
        with rasterio.open(url) as ds:
            with WarpedVRT(ds, crs=crs, transform=tr, width=n, height=n, resampling=Resampling.average,
                           src_nodata=ds.nodata, nodata=-9999) as v:
                z = v.read(1).astype("f8")
    z[(z < -9000) | (z > 9000)] = np.nan
    return z, tr


def grid_points(b, x0, y0):
    """World (UTM) coordinates of every grid node, shape (N, N) each; row 0 offshore."""
    f = math.radians(b["facing"])
    off = np.array([math.sin(f), math.cos(f)])            # toward the sea
    al = np.array([math.sin(f + math.pi / 2), math.cos(f + math.pi / 2)])  # +x: the right hand of someone on the sand looking out
    jj, ii = np.mgrid[0:N, 0:N].astype("f8")
    s = (ii * DX - EXT / 2) + b["along"]               # alongshore metres from the anchor
    d = b["shore_frac"] * EXT - jj * DX                 # metres offshore of the anchor
    X = x0 + s * al[0] + d * off[0]
    Y = y0 + s * al[1] + d * off[1]
    return X, Y


def sample(z, tr, X, Y):
    col = (X - tr.c) / tr.a - 0.5
    row = (Y - tr.f) / tr.e - 0.5
    return ndimage.map_coordinates(z, [row, col], order=1, mode="nearest", cval=np.nan)


def screened_fill(R, known, length_cells):
    """Fill R where not known: harmonic blend that decays to 0 over ~length_cells."""
    n = R.size
    idx = np.arange(n).reshape(R.shape)
    unk = ~known
    k2 = 1.0 / length_cells ** 2
    rows, cols, vals = [], [], []
    rhs = np.zeros(n)
    for c in idx[unk]:
        j, i = divmod(c, N)
        rows.append(c); cols.append(c); vals.append(-(4 + k2))
        for jj, ii in ((j - 1, i), (j + 1, i), (j, i - 1), (j, i + 1)):
            if not (0 <= jj < N and 0 <= ii < N):
                vals[-1] += 1  # Neumann at the domain edge
                continue
            cc = jj * N + ii
            if known.flat[cc]:
                rhs[c] -= R.flat[cc]
            else:
                rows.append(c); cols.append(cc); vals.append(1.0)
    for c in idx[known]:
        rows.append(c); cols.append(c); vals.append(1.0); rhs[c] = R.flat[c]
    A = sparse.csr_matrix((vals, (rows, cols)), shape=(n, n))
    return spsolve(A, rhs).reshape(R.shape)


def rock_mask(z, wet_lidar, b):
    """Rock where the bed is rough at the 5-10 m scale or where cliffs stand;
    at reef coasts also the shallow shelves the lidar saw, outside the sand beach."""
    smooth = ndimage.gaussian_filter(z, 2.0)
    rough = np.sqrt(ndimage.gaussian_filter((z - smooth) ** 2, 1.5))
    gy, gx = np.gradient(z, DX)
    slope = np.hypot(gx, gy)
    r = np.clip((rough - 0.10) / 0.20, 0, 1)
    r = np.maximum(r, np.clip((slope - 0.45) / 0.35, 0, 1) * (z > -1))  # cliffs and ledges
    r[~wet_lidar & (z < -1)] *= 0.5  # interpolated stretches are smooth by construction: no evidence either way
    if b.get("shelf_rock"):
        jj, ii = np.mgrid[0:N, 0:N].astype("f8")
        dist = np.hypot(ii * DX - (EXT / 2 - b["along"]), jj * DX - b["shore_frac"] * EXT)
        shelf = wet_lidar & (z > -3.0) | ((z > -0.5) & (z < 2.0) & (slope > 0.05))
        r = np.maximum(r, shelf * np.clip((dist - b["sand_radius"]) / 30, 0, 1))
    return ndimage.gaussian_filter(r, 1.0).clip(0, 1)


def cut(bid):
    b = BEACHES[bid]
    lon, lat = b["anchor"]
    (x0,), (y0,) = transform("EPSG:4326", b["utm"], [lon], [lat])
    X, Y = grid_points(b, x0, y0)
    cx, cy = X.mean(), Y.mean()
    size = EXT * math.sqrt(2) + 60
    cell = 1.0
    base, trb = read_window(b["coned"], b["utm"], cx, cy, size, cell)
    lid, trl = read_window(LIDAR_2014, b["utm"], cx, cy, size, cell)
    # 2.5 m cells from 1 m data: average before sampling.
    k = 3
    def avg(z):
        m = np.isfinite(z)
        s = ndimage.uniform_filter(np.where(m, z, 0), k)
        w = ndimage.uniform_filter(m.astype("f8"), k)
        out = s / np.maximum(w, 1e-9)
        out[w < 0.99] = np.nan  # a cell touching a gap counts as a gap
        return out
    zb = sample(avg(base), trb, X, Y)
    validL = np.isfinite(lid)
    zl = sample(np.where(validL, avg(lid), np.nan), trl, X, Y)
    known = np.isfinite(zl)
    known = ndimage.binary_erosion(known, iterations=1)  # lidar gap edges are noisy
    if np.isnan(zb).any():
        raise SystemExit(f"{bid}: CoNED has holes in the window ({np.isnan(zb).mean():.1%})")
    R = np.where(known, zl - zb, 0.0)
    R = screened_fill(R, known, length_cells=20)  # back to CoNED within ~50 m of the lidar
    z = zb + R - b["msl"]
    lidar_wet = known & (z < 0)
    rock = rock_mask(z, lidar_wet, b)
    # Land behind the beach: keep cliffs, but cap very high ground (the renderer and
    # raycast box stop at 60 m and nothing above ~30 m matters to the waves).
    z = np.minimum(z, 45.0)
    stats = dict(
        lidar_cover_wet=float(lidar_wet.sum() / max(1, (z < 0).sum())),
        depth_row0=[float(z[0].min()), float(np.median(z[0])), float(z[0].max())],
    )
    return b, z.astype("f4"), rock.astype("f4"), stats


def encode(b, bid, z, rock, stats):
    zi = np.clip(np.round(z * 100), -32000, 32000).astype("<i2")
    r = np.round(rock * 255).astype("u1")
    return {
        "v": 1, "n": N, "dx": DX, "wl": 0,
        "z": base64.b64encode(zi.tobytes()).decode(),
        "rock": base64.b64encode(r.tobytes()).decode(),
        "real": {
            "id": bid, "name": b["name"], "place": b["place"], "facing": b["facing"],
            "anchor": list(b["anchor"]), "swell": b["swell"],
            "datum": f"metres above mean sea level (NAVD88 - {b['msl']} m, {b['gauge']})",
            "lidarCover": round(stats["lidar_cover_wet"], 3),
            "sources": [
                "USGS CoNED topobathymetric model (public domain)",
                "USACE NCMP 2014 California topobathy lidar (public domain)",
            ],
        },
    }


def preview(path, z, rock, title):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    from matplotlib.colors import LightSource
    fig, ax = plt.subplots(figsize=(8, 8), dpi=100)
    zc = np.clip(z, -15, 15)
    rgb = LightSource(315, 35).shade(zc, cmap=plt.get_cmap("terrain"), vmin=-15, vmax=15, blend_mode="soft", vert_exag=3)
    ax.imshow(rgb, extent=[0, EXT, EXT, 0])
    ax.contour(np.arange(N) * DX, np.arange(N) * DX, zc, levels=[-10, -8, -6, -4, -3, -2, -1, 0], colors="k", linewidths=0.4)
    ax.contour(np.arange(N) * DX, np.arange(N) * DX, rock, levels=[0.5], colors="r", linewidths=0.6)
    ax.set_title(title + " (row 0 = offshore at top; red = rock)")
    fig.savefig(path, bbox_inches="tight")


if __name__ == "__main__":
    bid = sys.argv[1]
    b, z, rock, stats = cut(bid)
    out = os.path.join(os.path.dirname(__file__), "..", "beaches", f"{bid}.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump(encode(b, bid, z, rock, stats), f, separators=(",", ":"))
    print(bid, stats, f"z {z.min():.1f}..{z.max():.1f}", f"rock {np.mean(rock > 0.5):.1%}")
    if "--preview" in sys.argv:
        preview(sys.argv[sys.argv.index("--preview") + 1], z, rock, b["name"])
