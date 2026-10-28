"""
Forms analysis of a dump (bench/forms/dump.ts): hillshade, slope, curvature and per-term maps,
slope histograms by landform, a radially averaged power spectrum, geomorphon class shares and the
comb score (a spectral peak along contours on steep cells). numpy and Pillow only.

  python3 -I bench/forms/analyze.py <dump dir>/<place> [--out <dir>]
"""
import json, sys, os, math
import numpy as np
from PIL import Image

def load(d):
    m = json.load(open(os.path.join(d, 'meta.json')))
    n = m['n']
    T = {k: np.fromfile(os.path.join(d, k + '.f32'), dtype='<f4').reshape(n, n).astype(np.float64) for k in m['terms']}
    return m, T

def grad(E, s):
    gz, gx = np.gradient(E, s)
    return gx, gz

def slope_deg(E, s):
    gx, gz = grad(E, s)
    return np.degrees(np.arctan(np.hypot(gx, gz)))

def hillshade(E, s, az=315.0, alt=45.0):
    gx, gz = grad(E, s)
    # light from the north-west (azimuth clockwise from +z toward −x… x east, z south as rows grow)
    a, h = math.radians(az), math.radians(alt)
    lx, lz, ly = -math.sin(a) * math.cos(h) * -1, -math.cos(a) * math.cos(h), math.sin(h)
    nx, nz, ny = -gx, -gz, np.ones_like(gx)
    nl = np.sqrt(nx * nx + nz * nz + 1)
    return np.clip((nx * lx + nz * lz + ny * ly) / nl, 0, 1)

def laplacian(E, s):
    return (np.roll(E, 1, 0) + np.roll(E, -1, 0) + np.roll(E, 1, 1) + np.roll(E, -1, 1) - 4 * E) / (s * s)

def block(a, f):
    n = a.shape[0] // f
    return a[:n * f, :n * f].reshape(n, f, n, f).mean(axis=(1, 3))

def save_gray(a, path, lo=None, hi=None):
    lo = np.nanmin(a) if lo is None else lo
    hi = np.nanmax(a) if hi is None else hi
    im = np.clip((a - lo) / max(hi - lo, 1e-9), 0, 1)
    Image.fromarray((im * 255).astype(np.uint8)).save(path)

def save_div(a, path, lim):
    """Diverging blue–white–red, ±lim."""
    t = np.clip(a / max(lim, 1e-9), -1, 1)
    r = np.where(t > 0, 1, 1 + t); b = np.where(t < 0, 1, 1 - t); g = 1 - np.abs(t)
    Image.fromarray((np.stack([r, g, b], -1) * 255).astype(np.uint8)).save(path)

def slope_hist(S, mask, bins=90):
    h, _ = np.histogram(S[mask], bins=bins, range=(0, 90))
    tot = max(mask.sum(), 1)
    return (h / tot).tolist()

def psd_radial(E, s, win=2048):
    n = E.shape[0]; o = (n - win) // 2
    a = E[o:o + win, o:o + win].copy()
    # detrend a plane
    yy, xx = np.mgrid[0:win, 0:win]
    A = np.c_[xx.ravel(), yy.ravel(), np.ones(win * win)]
    c, *_ = np.linalg.lstsq(A, a.ravel(), rcond=None)
    a -= (A @ c).reshape(win, win)
    w = np.hanning(win); a *= w[:, None] * w[None, :]
    P = np.abs(np.fft.fftshift(np.fft.fft2(a))) ** 2
    ky, kx = np.mgrid[-win // 2:win // 2, -win // 2:win // 2]
    kr = np.hypot(kx, ky)
    kmax = win // 2
    edges = np.unique(np.round(np.logspace(0, math.log10(kmax), 48)).astype(int))
    out = []
    for i in range(len(edges) - 1):
        m = (kr >= edges[i]) & (kr < edges[i + 1])
        if m.any():
            k = 0.5 * (edges[i] + edges[i + 1]) / (win * s)  # cycles per metre
            out.append((k, float(P[m].mean())))
    k = np.array([o[0] for o in out]); p = np.array([o[1] for o in out])
    # exponent over 10–200 m wavelengths
    sel = (k >= 1 / 200) & (k <= 1 / 10)
    beta = -np.polyfit(np.log(k[sel]), np.log(p[sel]), 1)[0] if sel.sum() > 2 else float('nan')
    return k.tolist(), p.tolist(), float(beta)

# Geomorphons (Jasiewicz & Stepinski 2013): ternary pattern of the 8 lines of sight to distance L,
# flatness t; classed on (#lower, #higher).
FL, PK, RI, SH, SP, SL, HL, FS, VL, PT = range(10)
GM_NAMES = ['flat', 'peak', 'ridge', 'shoulder', 'spur', 'slope', 'hollow', 'footslope', 'valley', 'pit']
GM_TABLE = [
    [FL, FL, FL, FS, FS, VL, VL, VL, PT],
    [FL, FL, FS, FS, FS, VL, VL, VL, -1],
    [FL, SH, SL, SL, HL, HL, VL, -1, -1],
    [SH, SH, SL, SL, SL, HL, -1, -1, -1],
    [SH, SH, SP, SL, SL, -1, -1, -1, -1],
    [RI, RI, SP, SP, -1, -1, -1, -1, -1],
    [RI, RI, RI, -1, -1, -1, -1, -1, -1],
    [RI, RI, -1, -1, -1, -1, -1, -1, -1],
    [PK, -1, -1, -1, -1, -1, -1, -1, -1],
]

def geomorphons(E, s, L=60.0, t_deg=1.0):
    n = E.shape[0]; steps = int(round(L / s))
    t = math.tan(math.radians(t_deg))
    pad = steps
    P = np.pad(E, pad, mode='edge')
    minus = np.zeros(E.shape, np.int32); plus = np.zeros(E.shape, np.int32)
    for dx, dz in [(1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1), (0, -1), (1, -1)]:
        dl = s * math.hypot(dx, dz)
        zen = np.full(E.shape, -np.inf); nad = np.full(E.shape, -np.inf)
        for k in range(1, steps + 1):
            if k * dl > L: break
            sh = P[pad + dz * k: pad + dz * k + n, pad + dx * k: pad + dx * k + n]
            ang = (sh - E) / (k * dl)
            zen = np.maximum(zen, ang); nad = np.maximum(nad, -ang)
        # higher (+) where the line of sight rises past t and more than it falls; lower (−) the reverse
        plus += ((zen > t) & (zen > nad)).astype(np.int32)
        minus += ((nad > t) & (nad > zen)).astype(np.int32)
    tab = np.array(GM_TABLE)
    cls = tab[minus, plus]
    return cls

def gm_shares(cls, mask=None):
    c = cls if mask is None else cls[mask]
    tot = max(c.size, 1)
    return {GM_NAMES[i]: float((c == i).sum() / tot) for i in range(10)}

def bilinear(E, x, z):
    """E sampled at fractional grid coords (x column, z row)."""
    n = E.shape[0]
    x = np.clip(x, 0, n - 1.001); z = np.clip(z, 0, n - 1.001)
    i = np.floor(x).astype(int); j = np.floor(z).astype(int)
    fx = x - i; fz = z - j
    return (E[j, i] * (1 - fx) * (1 - fz) + E[j, i + 1] * fx * (1 - fz) + E[j + 1, i] * (1 - fx) * fz + E[j + 1, i + 1] * fx * fz)

def comb_spectrum(E, s, steep_deg=35.0, count=6000, half=128, seed=1, gsmooth=8.0):
    """
    Mean power spectrum along contours through steep cells: from each sampled cell, a line along the
    contour (perpendicular to the gradient smoothed over gsmooth m), 2·half samples at the grid's
    spacing, a quadratic removed, Hann windowed. Returns wavelengths (m), mean power, and the comb
    score: the largest excess (dB) of the spectrum over its power-law fit between 6 and 80 m.
    """
    f = max(1, int(round(gsmooth / s)))
    Es = block(E, f)
    gzs, gxs = np.gradient(Es, s * f)
    S = slope_deg(E, s)
    n = E.shape[0]
    rng = np.random.default_rng(seed)
    m = S > steep_deg
    m[:half + 2, :] = m[-half - 2:, :] = False; m[:, :half + 2] = m[:, -half - 2:] = False
    idx = np.flatnonzero(m)
    if idx.size < 50: return None
    pick = rng.choice(idx, size=min(count, idx.size), replace=False)
    jj, ii = np.unravel_index(pick, E.shape)
    gx = gxs[np.minimum(jj // f, Es.shape[0] - 1), np.minimum(ii // f, Es.shape[1] - 1)]
    gz = gzs[np.minimum(jj // f, Es.shape[0] - 1), np.minimum(ii // f, Es.shape[1] - 1)]
    gl = np.maximum(np.hypot(gx, gz), 1e-9)
    tx, tz = -gz / gl, gx / gl
    u = np.arange(-half, half)
    X = ii[:, None] + tx[:, None] * u[None, :]
    Z = jj[:, None] + tz[:, None] * u[None, :]
    prof = bilinear(E, X, Z)
    A = np.c_[np.ones(2 * half), u, u * u].astype(np.float64)
    coef, *_ = np.linalg.lstsq(A, prof.T, rcond=None)
    res = prof - (A @ coef).T
    w = np.hanning(2 * half)
    P = (np.abs(np.fft.rfft(res * w[None, :], axis=1)) ** 2).mean(axis=0)
    k = np.fft.rfftfreq(2 * half, d=s)
    lam = np.where(k > 0, 1 / np.maximum(k, 1e-12), np.inf)
    sel = (lam >= 6) & (lam <= 80)
    fit = np.polyfit(np.log(k[sel]), np.log(P[sel]), 1)
    exc = 10 * np.log10(P[sel] / np.exp(np.polyval(fit, np.log(k[sel]))))
    j = int(np.argmax(exc))
    return {'lam': lam[1:].tolist(), 'P': P[1:].tolist(), 'score_db': float(exc[j]), 'peak_m': float(lam[sel][j]), 'cells': int(pick.size)}

def main():
    d = sys.argv[1]
    out = sys.argv[sys.argv.index('--out') + 1] if '--out' in sys.argv else os.path.join(d, 'maps')
    os.makedirs(out, exist_ok=True)
    m, T = load(d)
    s = m['step']
    E = T['E']; W = T['W']
    S = slope_deg(E, s)
    res = {'place': m['id'], 'seed': m['seed'], 'x': m['x'], 'z': m['z'], 'E': [float(E.min()), float(E.max())]}
    hs = hillshade(E, s)
    save_gray(block(hs, 2), os.path.join(out, 'hillshade.png'), 0, 1)
    save_gray(block(S, 2), os.path.join(out, 'slope.png'), 0, 75)
    lap = laplacian(block(E, 4), 4 * s)
    save_div(lap, os.path.join(out, 'curvature.png'), 0.05)
    # a full-resolution 256 m crop at the centre
    c = E.shape[0] // 2
    save_gray(hs[c - 256:c + 256, c - 256:c + 256], os.path.join(out, 'hillshade_centre.png'), 0, 1)
    # each term's contribution (detail terms × their mask)
    sc = T['scale']
    for k in ['hills', 'ridges', 'karst', 'mesas', 'gullies']:
        a = T[k] * sc
        lim = max(1e-3, float(np.percentile(np.abs(a), 99.5)))
        res.setdefault('terms', {})[k] = {'p99.5_abs': lim, 'mean': float(a.mean())}
        save_div(block(a, 4), os.path.join(out, f'term_{k}.png'), lim)
        save_gray(block(hillshade(T['U'] + a, s), 4), os.path.join(out, f'shade_{k}.png'), 0, 1)
    for k in ['U', 'terraces', 'valleys']:
        a = T[k]
        lim = max(1e-3, float(np.percentile(np.abs(a - (a.mean() if k == 'U' else 0)), 99.5)))
        res.setdefault('terms', {})[k] = {'p99.5_abs': lim, 'mean': float(a.mean())}
        save_div(block(a - (a.mean() if k == 'U' else 0), 4), os.path.join(out, f'term_{k}.png'), lim)
    # slope histograms by landform, on land (no water)
    dry = W <= E
    masks = {
        'all': dry,
        'karst': dry & (T['karst'] * sc > 2),
        'mesa': dry & (T['mesas'] * sc > 2),
        'valley_wall': dry & (T['valleys'] < -5),
        'gullied': dry & (np.abs(T['gullies'] * sc) > 2),
    }
    res['slope_hist'] = {k: {'share': float(v.mean()), 'h': slope_hist(S, v)} for k, v in masks.items() if v.sum() > 1000}
    k, p, beta = psd_radial(E, s)
    res['psd'] = {'k': k, 'P': p, 'beta_10_200m': beta}
    # geomorphons at 2 m
    E2 = block(E, 4)
    cls = geomorphons(E2, 4 * s)
    res['geomorphons'] = gm_shares(cls)
    pal = np.array([[220, 220, 220], [140, 0, 0], [220, 40, 40], [255, 160, 120], [255, 200, 0], [240, 240, 140], [140, 200, 120], [100, 180, 240], [40, 80, 220], [20, 20, 120]], np.uint8)
    Image.fromarray(pal[cls]).save(os.path.join(out, 'geomorphons.png'))
    cs = comb_spectrum(E, s)
    res['comb'] = cs
    # the same on the land without its gullies, and on the gullies' term alone (attribution)
    cs2 = comb_spectrum(E - T['gullies'] * sc, s)
    res['comb_no_gullies'] = cs2
    json.dump(res, open(os.path.join(out, 'metrics.json'), 'w'))
    print(json.dumps({'place': m['id'], 'comb_db': cs and round(cs['score_db'], 2), 'comb_peak_m': cs and round(cs['peak_m'], 1),
                      'comb_no_gullies_db': cs2 and round(cs2['score_db'], 2), 'beta': round(beta, 2),
                      'gm': {k: round(v, 3) for k, v in res['geomorphons'].items()},
                      'terms': res['terms']}, indent=1))

if __name__ == '__main__':
    main()
