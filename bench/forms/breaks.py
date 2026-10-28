"""
How crisp the walls' breaks of slope are in a dump (bench/forms/dump.ts), for the heightfield's
limits in docs/FORMS.md. From each wall cell (slope ≥ 70°) the line of steepest slope is followed up
to the rim and down to the foot: the rim's width is the run between the 60° and 15° slopes, the foot's
between 60° and 40° (the talus begins under 40°). Each is also given in pixels at 10, 25 and 50 m from
the game's camera (70° vertical field, 900 rows), and the wall's vertical resolution on the clipmap's
0.5 and 1 m lattices (spacing × tan(slope)). Slopes are central differences on the dump's cells, so
a run under two cells is at the floor of what this can see. numpy only.

  python3 -I bench/forms/breaks.py <dump dir>/<place> [--count=4000]
"""
import json, math, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analyze import bilinear, grad  # noqa: E402

PX_PER_M = 900 / (2 * math.tan(math.radians(35)))  # pixels per metre at 1 m distance


def walk(S, gx, gz, x, z, sign, s, stops, reach=40.0, step=0.25):
    """Runs (m) from (x, z) along ±the gradient until the slope first falls under each of stops (descending)."""
    out, k = [], 0
    n = S.shape[0]
    for t in np.arange(step, reach, step):
        g = np.array([bilinear(gx, x, z), bilinear(gz, x, z)])
        gl = math.hypot(*g)
        if gl < 1e-9:
            break
        x += sign * step * g[0] / gl / s
        z += sign * step * g[1] / gl / s
        if not (1 <= x < n - 2 and 1 <= z < n - 2):
            break
        while k < len(stops) and bilinear(S, x, z) < stops[k]:
            out.append(t)
            k += 1
        if k == len(stops):
            return out
    return None


def main():
    d = sys.argv[1]
    count = int(next((a[8:] for a in sys.argv[2:] if a.startswith('--count=')), 4000))
    m = json.load(open(os.path.join(d, 'meta.json')))
    s, n = m['step'], m['n']
    E = np.fromfile(os.path.join(d, 'E.f32'), dtype='<f4').reshape(n, n).astype(np.float64)
    gx, gz = grad(E, s)
    S = np.degrees(np.arctan(np.hypot(gx, gz)))
    j, i = np.nonzero(S[2:-2, 2:-2] >= 70)
    if i.size == 0:
        print(json.dumps({'place': m['id'], 'wall_cells': 0}))
        return
    rng = np.random.default_rng(1)
    pick = rng.choice(i.size, min(count, i.size), replace=False)
    rim, foot, wall = [], [], []
    for p in pick:
        x, z = float(i[p] + 2), float(j[p] + 2)
        wall.append(S[j[p] + 2, i[p] + 2])
        up = walk(S, gx, gz, x, z, +1, s, [60, 15])
        dn = walk(S, gx, gz, x, z, -1, s, [60, 40])
        if up:
            rim.append(up[1] - up[0])
        if dn:
            foot.append(dn[1] - dn[0])
    q = lambda a, p: round(float(np.percentile(a, p)), 2) if a else None
    px = lambda w: {f'{r} m': round(w * PX_PER_M / r) for r in (10, 25, 50)} if w is not None else None
    ws = float(np.median(wall))
    out = {
        'place': m['id'], 'wall_cells': int(i.size), 'sampled': int(pick.size),
        'wall_slope_median_deg': round(ws, 1),
        'rim_run_m': {'median': q(rim, 50), 'p10': q(rim, 10), 'p90': q(rim, 90), 'n': len(rim)},
        'foot_run_m': {'median': q(foot, 50), 'p10': q(foot, 10), 'p90': q(foot, 90), 'n': len(foot)},
        'rim_px_median': px(q(rim, 50)), 'foot_px_median': px(q(foot, 50)),
        'wall_vertical_res_m': {'0.5 m lattice': round(0.5 * math.tan(math.radians(ws)), 2), '1 m lattice': round(math.tan(math.radians(ws)), 2)},
    }
    print(json.dumps(out))


if __name__ == '__main__':
    main()
