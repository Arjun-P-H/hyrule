"""
The form statistics of a reference DEM, measured as this world's are, to set and check the
tolerances in docs/FORMS.md: the slope histogram's bench (< 15°), talus (28–40°), gap (45–60°) and
cliff (> 60°) shares, its modes and whether its talus and cliff modes stand apart
(bench/forms/slopes.ts), geomorphon class shares (2 m cells, 60 m lines of sight) and the radially
averaged power spectrum's exponent over 10–200 m (bench/forms/analyze.py). numpy and Pillow only.

  python3 -I bench/forms/dem.py <dem.tif | dump dir> [--res=<m>] [--crop=<col>,<row>,<size>] [--json=<file>]

A GeoTIFF must be one float32 band in a metric projection: USGS 3DEP 1 m, swissALTI3D 0.5 m and the
EA's 1 m lidar are; Copernicus GLO-30 is in degrees and is reprojected first (for Guilin, say,
gdalwarp -t_srs EPSG:32649 -tr 30 30). The cell size is read from the file's ModelPixelScale tag
unless --res gives it. A dump dir (bench/forms/dump.ts) is read the same way, so a reference and
this world can be compared side by side. No download happens here: the sources are in docs/FORMS.md.
"""
import json, math, os, sys
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analyze import block, geomorphons, gm_shares, psd_radial, slope_deg  # noqa: E402

Image.MAX_IMAGE_PIXELS = None


def arg(k, d=None):
    for a in sys.argv[2:]:
        if a.startswith(f'--{k}='):
            return a[len(k) + 3:]
    return d


def read(path):
    """Heights (rows × cols, float64, NaN where there is no data) and the cell size (m) if known."""
    if os.path.isdir(path):
        m = json.load(open(os.path.join(path, 'meta.json')))
        n = m['n']
        return np.fromfile(os.path.join(path, 'E.f32'), dtype='<f4').reshape(n, n).astype(np.float64), m['step']
    im = Image.open(path)
    if im.mode not in ('F', 'I;16', 'I;16S', 'I'):
        raise SystemExit(f'{path}: mode {im.mode}, not a one-band height raster')
    E = np.asarray(im, dtype=np.float64)
    tags = getattr(im, 'tag_v2', {})
    scale = tags.get(33550)  # ModelPixelScaleTag (sx, sy, sz)
    nodata = tags.get(42113)  # GDAL_NODATA
    if nodata is not None:
        try:
            E[E == float(str(nodata).strip('\x00 '))] = np.nan
        except ValueError:
            pass
    E[(E < -1000) | (E > 9000)] = np.nan
    res = float(scale[0]) if scale else None
    if res is not None and res < 0.01:
        raise SystemExit(f'{path}: cell size {res} looks like degrees; reproject to metres first')
    return E, res


def modes(h):
    """slopes.ts modes: local maxima of the 1° histogram smoothed over 5°, above 4 % of its peak."""
    n = len(h)
    sm = [np.mean(h[max(0, i - 2):min(n, i + 3)]) for i in range(n)]
    peak = max(sm)
    return [i for i in range(n) if sm[i] > (sm[i - 1] if i else -1) and sm[i] >= (sm[i + 1] if i + 1 < n else -1) and sm[i] > 0.04 * peak]


def separate(bins):
    """slopes.ts separate, the form brief's test: in 5° bins, the largest over 30–40° and the largest over
    65–90° each stand at least as high as their neighbours, and the lowest bin between is at most half the lower."""
    t = max(range(6, 8), key=lambda b: (bins[b], -b))
    c = max(range(13, 18), key=lambda b: (bins[b], -b))
    local = lambda k: bins[k] >= (bins[k - 1] if k else 0) and bins[k] >= (bins[k + 1] if k + 1 < len(bins) else 0)
    lo = min(bins[t], bins[c])
    trough = min(bins[t + 1:c]) / lo if lo > 0 else math.inf
    return {'talus_peak_deg': 5 * t, 'cliff_peak_deg': 5 * c, 'trough': round(trough, 2) if trough < math.inf else None, 'separate': bool(local(t) and local(c) and trough <= 0.5)}


def stats(E, s):
    # fill the few holes with the mean of what is there, so gradients stay finite; their cells are dropped
    hole = ~np.isfinite(E)
    if hole.all():
        raise SystemExit('no data')
    E = np.where(hole, np.nanmean(E), E)
    S = slope_deg(E, s)
    ok = ~hole
    ok[[0, -1], :] = ok[:, [0, -1]] = False
    h, _ = np.histogram(S[ok], bins=91, range=(0, 91))
    N = max(int(h.sum()), 1)
    share = lambda lo, hi: round(100 * float(h[lo:hi].sum()) / N, 1)
    out = {
        'cell_m': s, 'size_m': [round(E.shape[1] * s), round(E.shape[0] * s)], 'relief_m': round(float(np.ptp(E[ok])), 1),
        'bench': share(0, 15), 'talus': share(28, 41), 'gap45_60': share(45, 60), 'cliff': share(60, 91),
        'modes_deg': modes(h.tolist()), 'hist_5deg': [share(5 * b, 5 * b + 5) for b in range(18)],
    }
    out['modes_test'] = separate(out['hist_5deg'])
    # geomorphons on 2 m cells (as analyze.py), from the nearest whole block of the native cells
    f = max(1, int(round(2.0 / s)))
    sq = min(E.shape) // f * f
    E2 = block(E[:sq, :sq], f)
    out['geomorphons_2m'] = {k: round(v, 3) for k, v in gm_shares(geomorphons(E2, f * s)).items()}
    # the spectrum over the largest centred square a power of two across
    win = 2 ** int(math.log2(min(E.shape)))
    if win * s >= 400:
        o0, o1 = (E.shape[0] - win) // 2, (E.shape[1] - win) // 2
        _, _, beta = psd_radial(E[o0:o0 + win, o1:o1 + win], s, win=win)
        out['psd_beta_10_200m'] = round(beta, 2)
    return out


def main():
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    E, res = read(sys.argv[1])
    s = float(arg('res', res or 0)) or None
    if not s:
        raise SystemExit('cell size unknown: pass --res=<m>')
    crop = arg('crop')
    if crop:
        c, r, n = (int(v) for v in crop.split(','))
        E = E[r:r + n, c:c + n]
    out = {'source': os.path.basename(sys.argv[1].rstrip('/')), **stats(E, s)}
    js = arg('json')
    if js:
        json.dump(out, open(js, 'w'), indent=1)
    print(json.dumps(out))


if __name__ == '__main__':
    main()
