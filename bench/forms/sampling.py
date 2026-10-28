"""
The sampling layer at a place: what the clipmap does to the height function before anything is lit.

Clipmap level l holds E point-sampled at spacing s = 0.5 * 2^l m (prefiltered for dp = s, which fades
the noise but leaves hard edges -- valley walls, karst rims, the coast ramp -- unresolved); the shading
reads it back bilinearly (clipLookup), and past GRAD_FAR its normal is the sun layer's central
difference over +-s at the lattice, itself read back bilinearly. A pixel at distance d reads the level
with s ~ d/250..d/125, so a texel spans 4-8 pixels at every distance.

This resamples a place's 0.5 m dump that way at s = 1, 2, 4, 8 m and measures, on the same contour-
tangent lines through steep cells, the spectrum of the height and of the shading (N.L) against the true
surface's: the excess (dB) the sampling adds at wavelengths of s..4s is the comb/corrugation it makes.

  python3 -I bench/forms/sampling.py <dump dir>/<place> [--crop r0 r1 c0 c1]
"""
import json, os, sys, math
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analyze import load, block, slope_deg, bilinear, save_gray

LIGHT = np.array([-0.5, 0.707, -0.5])  # a low-ish sun from the north-west, as the hillshades
LIGHT = LIGHT / np.linalg.norm(LIGHT)

def lerp_axis(C, k, o, n, axis):
    """C (coarse, spacing k fine cells, first sample at fine index o) linearly interpolated to n fine cells along axis."""
    m = C.shape[axis]
    t = np.clip((np.arange(n) - o) / k, 0, m - 1 - 1e-9)
    i0 = np.floor(t).astype(int); f = t - i0
    a = np.take(C, i0, axis=axis); b = np.take(C, i0 + 1, axis=axis)
    sh = [1, 1]; sh[axis] = n
    f = f.reshape(sh)
    return a * (1 - f) + b * f

def upsample(C, k, o, n):
    return lerp_axis(lerp_axis(C, k, o, n, 1), k, o, n, 0)

def level(E, k, o=None):
    """E point-sampled every k fine cells and read back bilinearly (clipLookup), with the sun layer's
    gradient (central differences over +-1 lattice step, read back bilinearly) for the far normal."""
    n = E.shape[0]
    o = k // 2 if o is None else o
    C = E[o::k, o::k]
    s = 1.0  # in units of fine cells; scaled by the caller
    gx = (np.roll(C, -1, 1) - np.roll(C, 1, 1)) / (2 * k)
    gz = (np.roll(C, -1, 0) - np.roll(C, 1, 0)) / (2 * k)
    gx[:, 0] = gx[:, 1]; gx[:, -1] = gx[:, -2]; gz[0] = gz[1]; gz[-1] = gz[-2]
    return upsample(C, k, o, n), upsample(gx, k, o, n), upsample(gz, k, o, n)

def ndl(gx, gz):
    """N.L for a gradient given in height per fine cell, at fine spacing s (divided out by the caller)."""
    nl = np.sqrt(gx * gx + gz * gz + 1)
    return np.clip((-gx * LIGHT[0] - gz * LIGHT[2] + LIGHT[1]) / nl, 0, 1)

def lines(E, s, steep_deg, count, half, seed=1, gsmooth=8.0):
    """Contour-tangent lines (the gradient smoothed over gsmooth m) through steep cells of the true E."""
    f = max(1, int(round(gsmooth / s)))
    Es = block(E, f)
    gzs, gxs = np.gradient(Es, s * f)
    S = slope_deg(E, s)
    rng = np.random.default_rng(seed)
    m = S > steep_deg
    m[:half + 2, :] = m[-half - 2:, :] = False; m[:, :half + 2] = m[:, -half - 2:] = False
    idx = np.flatnonzero(m)
    if idx.size < 50: return None
    pick = rng.choice(idx, size=min(count, idx.size), replace=False)
    jj, ii = np.unravel_index(pick, E.shape)
    jb = np.minimum(jj // f, Es.shape[0] - 1); ib = np.minimum(ii // f, Es.shape[1] - 1)
    gx, gz = gxs[jb, ib], gzs[jb, ib]
    gl = np.maximum(np.hypot(gx, gz), 1e-9)
    tx, tz = -gz / gl, gx / gl
    u = np.arange(-half, half)
    return ii[:, None] + tx[:, None] * u[None, :], jj[:, None] + tz[:, None] * u[None, :]

def spectrum(F, X, Z, s, detrend=True):
    prof = bilinear(F, X, Z)
    L = X.shape[1]
    if detrend:
        u = np.arange(L) - L / 2
        A = np.c_[np.ones(L), u, u * u]
        coef, *_ = np.linalg.lstsq(A, prof.T, rcond=None)
        prof = prof - (A @ coef).T
    else:
        prof = prof - prof.mean(axis=1, keepdims=True)
    w = np.hanning(L)
    P = (np.abs(np.fft.rfft(prof * w[None, :], axis=1)) ** 2).mean(axis=0)
    k = np.fft.rfftfreq(L, d=s)
    return k[1:], P[1:]

def band_db(k, P, P0, lo, hi):
    lam = 1 / k
    sel = (lam >= lo) & (lam <= hi)
    return float(10 * np.log10(P[sel].sum() / max(P0[sel].sum(), 1e-30)))

def main():
    d = sys.argv[1]
    out = os.path.join(d, 'maps'); os.makedirs(out, exist_ok=True)
    m, T = load(d)
    s = m['step']
    E = T['E']
    n = E.shape[0]
    X, Z = lines(E, s, 35.0, 4000, 256)
    k0, PE0 = spectrum(E, X, Z, s)
    gz0, gx0 = np.gradient(E, s)
    N0 = ndl(gx0, gz0)
    _, PN0 = spectrum(N0, X, Z, s, detrend=False)
    res = {'place': m['id'], 'cells': int(X.shape[0]), 'levels': []}
    crops = []
    c = n // 2
    r0, r1, c0, c1 = (c - 300, c + 300, c - 300, c + 300)
    if '--crop' in sys.argv:
        a = sys.argv.index('--crop'); r0, r1, c0, c1 = [int(v) for v in sys.argv[a + 1:a + 5]]
    crops.append(('true', N0[r0:r1, c0:c1]))
    for lv, spacing in [(1, 1.0), (2, 2.0), (3, 4.0), (4, 8.0)]:
        kf = int(round(spacing / s))
        Er, gxr, gzr = level(E, kf)
        gxr /= s; gzr /= s  # per fine cell -> per metre
        # near normal: the bilinear surface's own slope (gradientAt at a footprint under the spacing)
        gzb, gxb = np.gradient(Er, s)
        Nb = ndl(gxb, gzb)
        Ns = ndl(gxr, gzr)
        k, PE = spectrum(Er, X, Z, s)
        _, PNb = spectrum(Nb, X, Z, s, detrend=False)
        _, PNs = spectrum(Ns, X, Z, s, detrend=False)
        row = {'level': lv, 'spacing_m': spacing,
               'height_db_s_4s': band_db(k, PE, PE0, spacing, 4 * spacing),
               'shade_near_db_s_4s': band_db(k, PNb, PN0, spacing, 4 * spacing),
               'shade_far_db_s_4s': band_db(k, PNs, PN0, spacing, 4 * spacing),
               'shade_far_db_2s_16s': band_db(k, PNs, PN0, 2 * spacing, 16 * spacing),
               'rms_err_steep_m': float(np.sqrt(np.mean((Er - E)[slope_deg(E, s) > 35] ** 2)))}
        res['levels'].append(row)
        crops.append((f'L{lv} {spacing:g} m near', Nb[r0:r1, c0:c1]))
        crops.append((f'L{lv} {spacing:g} m far', Ns[r0:r1, c0:c1]))
    for name, a in crops:
        save_gray(a, os.path.join(out, 'sampling_' + name.replace(' ', '_') + '.png'), 0, 1)
    json.dump(res, open(os.path.join(out, 'sampling.json'), 'w'), indent=1)
    print(json.dumps(res, indent=1))

if __name__ == '__main__':
    main()
