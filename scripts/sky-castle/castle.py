"""
The cathedral castle's layout, and the pieces every layout shares (the island, towers, arcades, chapels, halls; the
rotunda: rotunda.py). The references: a white-marble and gold gothic citadel, lapis domes, blue banners, a
golden halo ring behind its central spire, on a floating island of hanging rock).
-----------------------------------------------------------------------------
build(lod) -> (exterior Geo, interior Geo). Blender axes, metres; the front faces -y; the outer plaza is z = 0.
Three terraces climb to the cathedral: the outer plaza ring (z 0), the inner podium (TB_Z), the cathedral's (TC_Z).
"""

import math
import random
import numpy as np
from kit import (Geo, Extras, T, Rz, Rx, S, chain, box, beam, lathe, prism, cylinder, cone, sphere, torus, arch_wall,
                 glass, gothic_window, finial, pinnacle, balustrade, column, stairs, banner, dome, angel, cypress, bush,
                 ivy, water_surface, spout, MARBLE, CREAM, SHADE, GOLD, GOLD_DARK, HALO_GOLD, GLASS,
                 GLASS_WARM, FLAME, ROCK, FLOOR_DARK, FLOOR_LIGHT, FLOOR_ALT, POOL_BED, THRONE, CARPET, ROOF,
                 TAU)

RND = random.Random(7)   # (ornament scatter: set again by build)
# Every layout is drawn at the sizes below and then scaled by SCALE (colossal): all but the stairs' steps, which are
# cut for the scaled rise, a hero's step each (steps).
SCALE = 2.5
ISLAND_R = 300.0       # the plaza's outline radius (before its wobble)
RIM_DEPTH = 22.0       # the arcaded rim band under the plaza's edge
TB_Z, TC_Z = 10.0, 22.0
TB = (205.0, 188.0)    # the inner podium's half extents
NAVE_W, NAVE_L, NAVE_H = 92.0, 230.0, 72.0
NAVE_Y0 = -120.0       # the facade's line
CROSS_Y = 40.0         # the crossing, under the central spire
SPIRE_TOP = 430.0


def outline(n=56):
    """The island's plaza outline [(x, y)], counter-clockwise."""
    pts = []
    for i in range(n):
        a = TAU * i / n
        r = ISLAND_R * (1 + 0.05 * math.sin(3 * a + 1.0) + 0.035 * math.sin(5 * a + 2.0) + 0.02 * math.sin(9 * a))
        pts.append((r * math.cos(a), r * 0.93 * math.sin(a)))
    return pts


def inside(poly, x, y):
    c = False
    for (x0, y0), (x1, y1) in zip(poly, poly[1:] + poly[:1]):
        if (y0 > y) != (y1 > y) and x < x0 + (y - y0) / (y1 - y0) * (x1 - x0): c = not c
    return c


def seg_frame(a, b, z):
    """The matrix setting a panel (made along x, facing -y) on the segment a → b at height z, and its length."""
    dx, dy = b[0] - a[0], b[1] - a[1]
    return chain(T((a[0] + b[0]) / 2, (a[1] + b[1]) / 2, z), Rz(math.atan2(dy, dx))), math.hypot(dx, dy)


def wall_run(g, a, b, z, h, t, bay, lod, kind='window', warm=False, door=None):
    """A wall from a to b (2-D; the outside on the right) of bays: lancet windows, blind arches or a door (bay index)."""
    M, L = seg_frame(a, b, z)
    n = max(1, round(L / bay)); w = L / n
    for i in range(n):
        m = chain(M, T(-L / 2 + w * (i + 0.5), 0, 0))
        if door is not None and i == door % n:
            arch_wall(g, w, h, t, w * 0.62, 0, h * 0.5, w * 0.5, MARBLE, m, n=8 if lod == 0 else 4, reveal=GOLD)
        elif kind == 'window':
            gothic_window(g, w, h, t, MARBLE, m, lod, warm)
        elif lod >= 2 and kind in ('blind', 'open'):
            box(g, w, t, h, MARBLE if kind == 'blind' else SHADE, m)
        elif kind == 'blind':
            out = arch_wall(g, w, h, t, w * 0.6, h * 0.1, h * 0.55, w * 0.5, MARBLE, m, n=6 if lod == 0 else 3, reveal=SHADE)
            glass(g, out, t * 0.25, SHADE, m)
        elif kind == 'open':
            arch_wall(g, w, h, t, w * 0.72, 0, h * 0.58, w * 0.42, MARBLE, m, n=6 if lod == 0 else 3, reveal=CREAM)
        else:
            box(g, w, t, h, MARBLE, chain(m, T(0, 0, 0)))
    if kind == 'blind' and h <= 25 and lod <= 1:
        for i in range(n):
            if RND.random() < 0.28:
                f = (i + 0.2 + 0.6 * RND.random()) / n
                ivy(g, 1.8 + RND.random() * 1.6, RND, chain(M, T(-L / 2 + L * f, -t / 2 - 0.6, h + 0.6)), lod, hang=h * (0.3 + 0.6 * RND.random()))
    # a gold string course and a cornice; on a tall wall, a pinnacle over each pier
    beam(g, (a[0], a[1], z + h - 1.2), (b[0], b[1], z + h - 1.2), t + 1.2, 1.2, GOLD, None)
    beam(g, (a[0], a[1], z + h), (b[0], b[1], z + h), t + 2.0, 1.6, CREAM, None)
    if h >= 20 and lod <= 1 and w < 60:
        for i in range(n + 1):
            pinnacle(g, 1.6, 8.0, chain(M, T(-L / 2 + w * i, 0, h + 1.6)), lod)


def steps(rise, lod):
    """How many steps a flight climbing `rise` (before SCALE) takes: a hero's step each (STEP_UP is 0.5 m), fewer far off."""
    return max(4, math.ceil(rise * SCALE / 0.45)) if lod == 0 else 4


def colossal(g, gi, x):
    """A layout's geometry and its extras scaled by SCALE (build's last step); x None at the coarse levels."""
    for geo in (g, gi) + ((x.leaves, x.water) if x else ()):
        geo.v = [v * SCALE for v in geo.v]
    if x:
        x.props = [(k, S(SCALE) @ M) for k, M in x.props]
        for p in x.plants:
            p[1] = p[1] * SCALE
            p[2] = min(p[2] * SCALE, 3.0)  # (the forest's largest: forest/placement.ts SCALE)
        for f in x.falls:
            f[0] = f[0] * SCALE; f[2] *= SCALE; f[3] *= SCALE
    return g, gi, x


def spouts(g, rnd, lod, n=8):
    """Gold spouts round the rim (none toward the front gate), each pouring a waterfall off the island."""
    poly = outline(56)
    for k in range(n):
        a = TAU * (k + 0.5) / n + 0.25 * (rnd.random() - 0.5)
        if abs(math.remainder(a + math.pi / 2, TAU)) < 0.5: continue  # (the gate)
        i = int(round(a / TAU * len(poly))) % len(poly)
        x, y = poly[i]
        r = math.hypot(x, y)
        spout(g, (x * (r + 3.5) / r, y * (r + 3.5) / r), (x / r, y / r), -RIM_DEPTH * 0.45, lod, width=12.0, reach=420.0)


POOL = (0.0, -TB[1] - 44, 34.0, 9.0)   # the reflecting pool before the cathedral's stairs: middle x, y, half axes


def covered(x0, y0, t, rects):
    """Whether a tile [x0, x0 + t] × [y0, y0 + t] lies wholly in one of rects [(x0, y0, x1, y1)] (a floor over it)."""
    return any(a <= x0 and b <= y0 and x0 + t <= c and y0 + t <= d for a, b, c, d in rects)


def front_wall(g, a, b, z, h, t, bay, lod, gap):
    """A podium's wall from a to b, broken where its front stairs (gap wide, about x = 0) arrive, if they cross it."""
    if a[1] == b[1] and a[1] < 0 and min(a[0], b[0]) < -gap / 2 and max(a[0], b[0]) > gap / 2:
        s = 1 if b[0] > a[0] else -1
        wall_run(g, a, (-s * gap / 2, a[1]), z, h, t, bay, lod, kind='blind')
        wall_run(g, (s * gap / 2, a[1]), b, z, h, t, bay, lod, kind='blind')
    else:
        wall_run(g, a, b, z, h, t, bay, lod, kind='blind')


def rect_corners(cx, cy, hx, hy):
    return [(cx - hx, cy - hy), (cx + hx, cy - hy), (cx + hx, cy + hy), (cx - hx, cy + hy)]


# ------------------------------------------------------------------ the island

def island(g, rnd, lod, skip=None):
    """The island: the slab and its paving (not where skip(x0, y0, tile) says a podium covers it), the arcaded rim, the hanging rock."""
    skip = skip or (lambda x0, y0, t: covered(x0, y0, t, [(-TB[0], -TB[1], TB[0], TB[1])]))
    poly = outline(56 if lod < 2 else 28)
    n = len(poly)
    # the slab, its paving (tiles, for the light's sake and the eye's), the gold trim, the balustrade
    prism(g, poly, -RIM_DEPTH, -0.05, CREAM, col_side=SHADE)
    tile = 12.0 if lod == 0 else 30.0
    shrunk = [(x * 0.97, y * 0.97) for x, y in poly]
    for ix in range(int(-ISLAND_R / tile) - 1, int(ISLAND_R / tile) + 1):
        for iy in range(int(-ISLAND_R / tile) - 1, int(ISLAND_R / tile) + 1):
            x0, y0 = ix * tile, iy * tile
            if skip(x0, y0, tile): continue
            if all(inside(shrunk, x, y) for x, y in ((x0, y0), (x0 + tile, y0), (x0, y0 + tile), (x0 + tile, y0 + tile))):
                c = FLOOR_LIGHT if (ix + iy) % 2 else FLOOR_ALT
                g.add([(x0, y0, 0), (x0 + tile, y0, 0), (x0 + tile, y0 + tile, 0), (x0, y0 + tile, 0)], [(0, 1, 2, 3)], c)
    rim = [(x * 1.004, y * 1.004) for x, y in poly]
    for i in range(n):
        a, b = rim[i], rim[(i + 1) % n]
        wall_run(g, a, b, -RIM_DEPTH, RIM_DEPTH, 3.0, 11.0, lod, kind='blind')
        if lod <= 1 and i % 2 == 0:
            # a gold pendant under the arcade's keystone, and ivy spilling over the edge
            m = (a[0] + b[0]) / 2, (a[1] + b[1]) / 2
            lathe(g, [(0, 0), (2.4, -2), (1.2, -5), (2.0, -8), (0.6, -12), (0, -18)], 8 if lod == 0 else 5, GOLD,
                  T(m[0] * 1.01, m[1] * 1.01, -RIM_DEPTH))
            if rnd.random() < 0.7:
                ivy(g, 3.0 + rnd.random() * 2, rnd, T(m[0] * 1.012, m[1] * 1.012, -1.0), lod, hang=10 + rnd.random() * 14)
    edge = [(x * 0.992, y * 0.992, 0) for x, y in poly]
    balustrade(g, edge, 1.6, None, lod, closed=True)
    if lod <= 1:
        for i in range(0, n, 4):  # statues on the rim's posts
            x, y, _ = edge[i]
            a = math.atan2(y, x)
            angel(g, 9.0, chain(T(x * 0.985, y * 0.985, 0), Rz(a - math.pi / 2)), lod)
    underside(g, poly, rnd, lod)


def vnoise(x, y, seed):
    """The land's value noise (src/lib/noise.ts vnoise, the realm's undersides' too), over numpy arrays: −1 … 1."""
    M = np.uint64(0xFFFFFFFF)
    x, y = np.asarray(x, np.float64), np.asarray(y, np.float64)
    ix, iy = np.floor(x), np.floor(y)
    fx, fy = x - ix, y - iy
    ux, uy = fx ** 3 * (fx * (fx * 6 - 15) + 10), fy ** 3 * (fy * (fy * 6 - 15) + 10)
    u32 = lambda a: (np.asarray(a).astype(np.int64) & 0xFFFFFFFF).astype(np.uint64)
    mul = lambda a, b: (a * np.uint64(b)) & M

    def fin(h):
        h = h ^ (h >> np.uint64(16)); h = mul(h, 0x85ebca6b)
        h = h ^ (h >> np.uint64(13)); h = mul(h, 0xc2b2ae35)
        return (h ^ (h >> np.uint64(16))).astype(np.float64) * (2.0 / 4294967295.0)
    x0 = u32(ix); x1 = (x0 + np.uint64(1)) & M
    y0 = mul(u32(iy), 0x27d4eb2f); y1 = (y0 + np.uint64(0x27d4eb2f)) & M
    st = mul(u32(seed), 0x9e3779b9)
    a, b, c, d = fin(x0 ^ y0 ^ st), fin(x1 ^ y0 ^ st), fin(x0 ^ y1 ^ st), fin(x1 ^ y1 ^ st)
    return a + (b - a) * ux + (c - a) * uy + (a - b - c + d) * ux * uy - 1.0


def underside(g, poly, rnd, lod):
    """
    The hanging rock, as the sky realm's islands' (skyworld/island.ts bottomAt): a cliff wall under the rim, an
    inverted mountain, lumps of rock, hanging spires of two sizes out nearly to the rim, and two or three great spikes
    off the middle — on a polar grid under the outline (each face's colour carries how far out it lies: the shader
    paints the realm's strata by it, modelIsles.wgsl.ts). The great gold pendant hangs from its lowest point.
    """
    R = ISLAND_R
    seed = rnd.randrange(1 << 30)
    cliff = 0.065 * R * (0.8 + 0.4 * rnd.random())
    depth = R * (0.6 + 0.2 * rnd.random())
    lobes = []
    for _ in range(2 + int(rnd.random() * 2)):
        a, d = rnd.random() * TAU, (0.14 + 0.42 * rnd.random()) * R
        lobes.append((math.cos(a) * d, math.sin(a) * d * 0.93, (0.16 + 0.22 * rnd.random()) * R, depth * (0.55 + 0.8 * rnd.random())))
    rings, segs = (120, 240) if lod == 0 else (60, 120) if lod == 1 else (30, 60)
    th = np.linspace(0, TAU, segs, endpoint=False)
    rad = R * (1 + 0.05 * np.sin(3 * th + 1.0) + 0.035 * np.sin(5 * th + 2.0) + 0.02 * np.sin(9 * th))
    rho = np.linspace(1.0, 0.0, rings + 1)[:, None] * np.ones(segs)[None, :]
    qx, qy = rho * (rad * np.cos(th))[None, :], rho * (rad * 0.93 * np.sin(th))[None, :]
    sm = lambda e0, e1, x: np.clip((x - e0) / (e1 - e0), 0, 1) ** 2 * (3 - 2 * np.clip((x - e0) / (e1 - e0), 0, 1))
    wall = cliff * (1 - sm(0.86, 1, rho))
    v = np.maximum(0, 1 - rho / 0.86)
    cone = depth * v ** 0.9 * (0.8 + 0.35 * (0.5 + 0.5 * vnoise(qx / (0.22 * R), qy / (0.22 * R), seed + 5)))
    rock = 0.05 * R * (0.5 + 0.5 * vnoise(qx / (0.09 * R), qy / (0.09 * R), seed + 6)) * (1 - sm(0.8, 1, rho))
    dn = np.maximum(0, vnoise(qx / (0.07 * R), qy / (0.07 * R), seed + 7))
    dn2 = np.maximum(0, vnoise(qx / (0.042 * R), qy / (0.042 * R), seed + 8))
    drip = 0.17 * R * dn ** 3 * (1 - sm(0.6, 0.95, rho)) + 0.07 * R * dn2 ** 3 * (1 - sm(0.4, 0.97, rho))
    b = -RIM_DEPTH - wall - cone - rock - drip
    rim = 1 - sm(0.85, 1, rho)
    for lx, ly, lr, ld in lobes:
        t = np.maximum(0, 1 - np.hypot(qx - lx, qy - ly) / lr)
        spike = -RIM_DEPTH - wall - ld * rim * t ** 1.6 * (0.88 + 0.24 * (0.5 + 0.5 * vnoise((qx - lx) / (0.3 * lr), (qy - ly) / (0.3 * lr), seed + 9)))
        b = np.minimum(b, np.where(t > 0, spike, b))
    b[0] = -RIM_DEPTH  # (the rim ring meets the slab under the arcade)
    verts = np.stack([qx, qy, b], -1).reshape(-1, 3)
    faces, cols = [], []
    for i in range(rings):
        for j in range(segs):
            a0, a1 = i * segs + j, i * segs + (j + 1) % segs
            faces.append((a0, a0 + segs, a1 + segs, a1))
            cols.append((float(rho[i, 0] + rho[i + 1, 0]) / 2, 0.0, 0.0, 0.0, ROCK[4]))
    # (each face its own colour: how far out it lies)
    base = g.n
    g.v.append(verts); g.uv.append(np.zeros((len(verts), 2)))
    g.f += [tuple(base + k for k in f) for f in faces]
    g.c += cols
    g.n += len(verts)
    # the great gold pendant under the lowest point
    k = int(np.argmin(b))
    px, py, pz = verts[k]
    prof = [(0, 0), (16, -6), (9, -18), (14, -30), (5, -44), (8, -54), (2.5, -70), (0, -100)]
    lathe(g, prof, 16 if lod == 0 else 8, GOLD, T(px, py, pz + 10))


# ------------------------------------------------------------------ terraces

def terraces(g, rnd, lod):
    hx, hy = TB
    # the inner podium: a wall of blind arcades, a balustrade, grand stairs at the front and the sides
    cs = rect_corners(0, 0, hx, hy)
    for i in range(4):
        a, b = cs[i], cs[(i + 1) % 4]
        front_wall(g, a, b, 0, TB_Z, 2.4, 10.0, lod, 70.0)
    box(g, hx * 2 - 1, hy * 2 - 1, TB_Z - 0.05, CREAM)
    tile = 8.0 if lod == 0 else 24.0
    px, py0, py1 = NAVE_W / 2 + 40, NAVE_Y0 - 40, NAVE_Y0 + NAVE_L + 30
    gx, gy = hx - 30, hy - 30
    under = [(-px, py0, px, py1)] + [(sx * gx - 21, sy * gy - 21, sx * gx + 21, sy * gy + 21) for sx in (-1, 1) for sy in (-1, 1)]
    for ix in range(int(-hx / tile), int(hx / tile)):
        for iy in range(int(-hy / tile), int(hy / tile)):
            x0, y0 = ix * tile, iy * tile
            if lod == 0 and covered(x0, y0, tile, under): continue
            c = FLOOR_LIGHT if (ix + iy) % 2 else FLOOR_ALT
            g.add([(x0, y0, TB_Z), (x0 + tile, y0, TB_Z), (x0 + tile, y0 + tile, TB_Z), (x0, y0 + tile, TB_Z)], [(0, 1, 2, 3)], c)
    sw = 70.0
    rail = [(-hx + 1, -hy + 1, TB_Z), (-sw / 2 - 2, -hy + 1, TB_Z)]
    balustrade(g, rail, 1.5, None, lod)
    balustrade(g, [(sw / 2 + 2, -hy + 1, TB_Z), (hx - 1, -hy + 1, TB_Z), (hx - 1, hy - 1, TB_Z), (-hx + 1, hy - 1, TB_Z), (-hx + 1, -hy + 1, TB_Z)], 1.5, None, lod)
    stairs(g, sw, TB_Z, 26, steps(TB_Z, lod), MARBLE, T(0, -hy - 26, 0))
    for s in (-1, 1):
        box(g, 6, 28, TB_Z + 2.5, CREAM, T(s * (sw / 2 + 3), -hy - 13, 0))
        angel(g, 14.0, T(s * (sw / 2 + 3), -hy - 23.5, TB_Z + 2.5), lod)
    # a reflecting pool before the stairs, gold-rimmed
    cx, cy, ax, ay = POOL
    pool = [(cx + math.cos(a) * ax, cy + math.sin(a) * ay) for a in np.linspace(0, TAU, 33)[:-1]]
    # a raised basin (the plaza's slab runs under it): a lapis mosaic bed, a marble kerb with a gold coping, water
    prism(g, pool, 0.0, 0.08, POOL_BED)
    prism(g, pool, 0.0, 1.1, CREAM, top=False, bottom=False)
    rim = [(cx + math.cos(a) * (ax + 0.6), cy + math.sin(a) * (ay + 0.6)) for a in np.linspace(0, TAU, 33)[:-1]]
    prism(g, rim, 0.0, 1.0, CREAM, top=False, bottom=False)
    for a, b in zip(pool + pool[:1], pool[1:] + pool[:1]):
        beam(g, (a[0], a[1], 1.0), (b[0], b[1], 1.0), 1.0, 0.25, GOLD, None)
    water_surface(g, pool, 0.85, 0.77)
    # cypresses in rows along the outer plaza
    if lod <= 1:
        for s in (-1, 1):
            for k in range(7):
                cypress(g, 14 + rnd.random() * 4, T(s * (60 + k * 20), -hy - 18, 0), lod)
    # the cathedral's podium
    px, py0, py1 = NAVE_W / 2 + 40, NAVE_Y0 - 40, NAVE_Y0 + NAVE_L + 30
    cp = [(-px, py0), (px, py0), (px, py1), (-px, py1)]
    for i in range(4):
        front_wall(g, cp[i], cp[(i + 1) % 4], TB_Z, TC_Z - TB_Z, 2.0, 9.0, lod, 54.0)
    box(g, px * 2 - 1, py1 - py0 - 1, TC_Z - 0.05, CREAM, T(0, (py0 + py1) / 2, 0))
    tile = 6.0 if lod == 0 else 18.0
    nave = [(-NAVE_W / 2 + 1.6, NAVE_Y0 + 3, NAVE_W / 2 - 1.6, NAVE_Y0 + NAVE_L)]
    for ix in range(int(-px / tile), int(px / tile)):
        for iy in range(int(py0 / tile), int(py1 / tile)):
            x0, y0 = ix * tile, iy * tile
            if lod == 0 and covered(x0, y0, tile, nave): continue
            c = FLOOR_LIGHT if (ix + iy) % 2 else FLOOR_ALT
            g.add([(x0, y0, TC_Z), (x0 + tile, y0, TC_Z), (x0 + tile, y0 + tile, TC_Z), (x0, y0 + tile, TC_Z)], [(0, 1, 2, 3)], c)
    stairs(g, 54, TC_Z - TB_Z, 18, steps(TC_Z - TB_Z, lod), MARBLE, T(0, py0 - 18, TB_Z))
    balustrade(g, [(27, py0 + 1, TC_Z), (px - 1, py0 + 1, TC_Z), (px - 1, py1 - 1, TC_Z), (-px + 1, py1 - 1, TC_Z), (-px + 1, py0 + 1, TC_Z), (-27, py0 + 1, TC_Z)], 1.4, None, lod)


# ------------------------------------------------------------------ the cathedral

def buttress(g, x, y, z, h, out, lod, a=0.0):
    """A stepped buttress against a wall (sticking out along -y in its own axes), a pinnacle on top."""
    m = chain(T(x, y, z), Rz(a))
    box(g, 5.0, out, h * 0.55, MARBLE, chain(m, T(0, -out / 2, 0)))
    box(g, 4.2, out * 0.7, h * 0.3, MARBLE, chain(m, T(0, -out * 0.35, h * 0.55)))
    box(g, 3.6, out * 0.45, h * 0.22, MARBLE, chain(m, T(0, -out * 0.22, h * 0.85)))
    pinnacle(g, 3.8, 22.0, chain(m, T(0, -out * 0.22, h * 1.07)), lod)


def facade_tower(g, x, y, lod, rnd):
    """One of the facade's twin towers: a square shaft of three window tiers, a pierced lantern, a spire."""
    z = TC_Z
    s = 26.0
    tiers = [(0, 62), (62, 44), (106, 36)]
    for k, (z0, h) in enumerate(tiers):
        w = s - k * 3
        cs = rect_corners(x, y, w / 2, w / 2)
        for i in range(4):
            wall_run(g, cs[i], cs[(i + 1) % 4], z + z0, h, 2.6, w / 2 if k else w / 3, lod, kind='window')
        box(g, w, w, 0.6, CREAM, T(x, y, z + z0))
        for cx, cy in cs:
            pinnacle(g, 3.0, 16.0, T(cx, cy, z + z0 + h), lod)
        if lod <= 1:
            balustrade(g, [(cx, cy, z + z0 + h) for cx, cy in cs], 1.4, None, lod, closed=True)
    zt = z + 142
    lathe(g, [(9, 0), (8, 14), (0.01, 14)], 8, MARBLE, T(x, y, zt))
    for k in range(8 if lod < 2 else 0):
        a = TAU * k / 8
        cylinder(g, 0.9, 14, 6, MARBLE, T(x + 9 * math.cos(a), y + 9 * math.sin(a), zt))
    lathe(g, [(9.5, 14), (6, 20), (0, 78)], 8, CREAM, T(x, y, zt), phase=math.pi / 8)
    for k in range(4 if lod < 2 else 0):
        for zz in (30, 46, 62):
            a = TAU * k / 4 + math.pi / 8
            r = 9.5 - 9.5 * (zz - 14) / 64 + 0.6
            beam(g, (x + r * math.cos(a), y + r * math.sin(a), zt + zz), (x + r * math.cos(a + 0.6), y + r * math.sin(a + 0.6), zt + zz), 0.8, 0.8, GOLD, None)
    finial(g, 18, T(x, y, zt + 76), lod)


def cathedral(g, gi, rnd, lod):
    z = TC_Z
    hw = NAVE_W / 2
    y0, y1 = NAVE_Y0, NAVE_Y0 + NAVE_L
    bay = 13.0
    # the nave's walls: tall lancets between buttresses; the transept's arms at the crossing
    tw = 30.0  # the transept arm's half width (along y)
    tx = hw + 44  # its end (x)
    for s in (-1, 1):
        # the side walls, broken by the transept
        for a, b in (((s * hw, y0 + 26), (s * hw, CROSS_Y - tw)), ((s * hw, CROSS_Y + tw), (s * hw, y1))):
            if s < 0: a, b = b, a
            wall_run(g, a, b, z, NAVE_H, 3.0, bay, lod, kind='window')
            yy0, yy1 = sorted((a[1], b[1]))
            nb = max(1, round((yy1 - yy0) / bay))
            for i in range(nb + 1):
                buttress(g, s * hw, yy0 + (yy1 - yy0) * i / nb, z, NAVE_H, 9.0, lod, a=(math.pi / 2 if s > 0 else -math.pi / 2))
        # the transept arm: its two side walls and its end, with a rose window
        a1, b1 = (s * hw, CROSS_Y - tw), (s * tx, CROSS_Y - tw)
        a2, b2 = (s * tx, CROSS_Y + tw), (s * hw, CROSS_Y + tw)
        if s > 0:
            wall_run(g, a1, b1, z, NAVE_H, 3.0, bay, lod, kind='window')
            wall_run(g, b1, a2, z, NAVE_H, 3.0, tw, lod, kind='window', door=1)
            wall_run(g, a2, b2, z, NAVE_H, 3.0, bay, lod, kind='window')
        else:
            wall_run(g, b1, a1, z, NAVE_H, 3.0, bay, lod, kind='window')
            wall_run(g, a2, b1, z, NAVE_H, 3.0, tw, lod, kind='window', door=0)
            wall_run(g, b2, a2, z, NAVE_H, 3.0, bay, lod, kind='window')
        gable(g, (s * tx, CROSS_Y - tw), (s * tx, CROSS_Y + tw), z + NAVE_H, 34, lod, flip=s < 0)
        rose(g, s * (tx + 1.8), CROSS_Y, z + NAVE_H + 12, 11, lod, axis='x')
        for yy in (CROSS_Y - tw, CROSS_Y + tw):
            pinnacle(g, 4.5, 34, T(s * tx, yy, z + NAVE_H), lod)
    # the roofs: the nave's and the transept's, steep, cream with a gold ridge
    roof(g, (0, y0 + 26), (0, y1), hw + 1.5, z + NAVE_H, 40, lod)
    roof(g, (-tx, CROSS_Y), (tx, CROSS_Y), tw + 1.5, z + NAVE_H, 40, lod)
    # the apse: a half ring of warm windows
    pts = [(hw * math.cos(a), y1 + hw * math.sin(a)) for a in np.linspace(0, math.pi, 9)]
    for a, b in zip(pts, pts[1:]):
        wall_run(g, a, b, z, NAVE_H, 3.0, 18, lod, kind='window', warm=True)
        buttress(g, a[0], a[1], z, NAVE_H, 9.0, lod, a=math.atan2(a[1] - y1, a[0]) - math.pi / 2 + math.pi)
    lathe(g, [(hw + 1.5, 0), (hw * 0.6, 22), (0, 34)], 16 if lod < 2 else 8, CREAM, chain(T(0, y1, z + NAVE_H), S(1, 1, 1)), cap_bot=False)
    # the facade: the screen wall with the great portal, a gallery of kings, the rose window, the gable
    fw = NAVE_W + 4
    fy = y0
    m = T(0, fy, z)
    portal(g, m, fw, lod)
    rose(g, 0, fy - 2.2, z + 92, 19, lod, axis='y')
    gable(g, (-hw - 2, fy), (hw + 2, fy), z + 110, 44, lod)
    for s in (-1, 1):
        facade_tower(g, s * (hw + 13), fy + 13, lod, rnd)
        banner(g, 9, 44, chain(T(s * 24, fy - 3.4, z + 70)), lod)
    central_spire(g, lod)
    halo(g, lod)
    interior(gi, rnd, lod)


def portal(g, m, fw, lod):
    """The facade's lower screen: a deep pointed portal (stepped, gold-lined archivolts) between blind bays."""
    h = 110.0
    pw, ph = 34.0, 62.0
    arch_wall(g, fw, h, 6.0, pw, 0, ph - pw * 0.5, pw * 0.75, MARBLE, m, n=10 if lod == 0 else 4, reveal=CREAM)
    steps = 4 if lod == 0 else 2 if lod == 1 else 0
    for k in range(steps):
        w = pw + 4 + k * 4
        arch_wall(g, w + 4, ph + k * 3 + 8, 1.6, w, 0, ph - pw * 0.5, w * 0.75, CREAM if k % 2 else MARBLE,
                        chain(m, T(0, -3.8 - k * 1.6, 0)), n=10 if lod == 0 else 4, reveal=GOLD if k % 2 == 0 else CREAM)
    # the tympanum's gold star over the doorway, a gallery of small arches above the portal
    if lod <= 1:
        for i in range(-6, 7):
            if abs(i) < 1: continue
            out = arch_wall(g, 6.6, 12, 1.2, 4.0, 0, 7, 3.2, MARBLE, chain(m, T(i * 6.8, -3.6, 70)), n=5 if lod == 0 else 3, reveal=SHADE)
            glass(g, out, 0.3, SHADE, chain(m, T(i * 6.8, -3.6, 70)))
        balustrade(g, [(-fw / 2, -4.2, 82), (fw / 2, -4.2, 82)], 1.6, m, lod)


def rose(g, x, y, z, r, lod, axis='y'):
    """A rose window: glowing glass in a gold ring with radial tracery and a ring of foils."""
    M = chain(T(x, y, z), Rx(math.pi / 2)) if axis == 'y' else chain(T(x, y, z), Rz(math.pi / 2), Rx(math.pi / 2))
    n = 32 if lod == 0 else 12
    lathe(g, [(0, -0.1), (r, -0.1), (r, 0.1), (0, 0.1)], n, GLASS, M)
    torus(g, r, r * 0.07, n, 6 if lod == 0 else 4, GOLD, M)
    if lod <= 1:
        torus(g, r * 0.32, r * 0.04, 16, 4, GOLD, M)
        for k in range(12):
            a = TAU * k / 12
            beam(g, (r * 0.32 * math.cos(a), r * 0.32 * math.sin(a), -0.3), (r * math.cos(a), r * math.sin(a), -0.3), r * 0.05, r * 0.05, GOLD, M)
            if lod == 0:
                torus(g, r * 0.14, r * 0.025, 10, 3, GOLD, chain(M, T(r * 0.7 * math.cos(a + 0.26), r * 0.7 * math.sin(a + 0.26), 0)))
    torus(g, r * 1.12, r * 0.09, n, 4, CREAM, M)


def gable(g, a, b, z, rise, lod, flip=False):
    """A gable over the wall a → b: a triangular screen with a gold edge, crockets, and a finial."""
    M, L = seg_frame(a, b, z)
    prism(g, [(-L / 2, 0), (L / 2, 0), (0, rise)], -1.5, 1.5, MARBLE, chain(M, Rx(math.pi / 2)))
    for s in (-1, 1):
        beam(g, (s * L / 2, -1.7, 0), (0, -1.7, rise), 1.4, 1.4, GOLD, M)
        if lod == 0:
            for k in range(1, 6):
                t = k / 6
                sphere(g, 1.1, 5, CREAM, chain(M, T(s * L / 2 * (1 - t), -2.0, rise * t + 1.2)))
    finial(g, 14, chain(M, T(0, 0, rise - 1)), lod)


def roof(g, a, b, hw, z, rise, lod):
    """A gable roof over the line a → b (its half width hw), cream with a gold ridge and gold dormer-spikes."""
    M, L = seg_frame(a, b, z)
    v = [(-L / 2, -hw, 0), (L / 2, -hw, 0), (L / 2, 0, rise), (-L / 2, 0, rise), (-L / 2, hw, 0), (L / 2, hw, 0)]
    g.add(v, [(0, 1, 2, 3), (5, 4, 3, 2), (0, 3, 4), (1, 5, 2)], CREAM, M)
    g.add(v, [(3, 2, 1, 0), (2, 3, 4, 5)], SHADE, M)  # (its underside: the vault seen from within)
    box(g, L, 1.6, 1.6, GOLD, chain(M, T(0, 0, rise - 0.6)))
    if lod <= 1:
        n = int(L / 12)
        for i in range(n):
            x = -L / 2 + (i + 0.5) * L / n
            finial(g, 7, chain(M, T(x, 0, rise + 0.8)), lod)


def central_spire(g, lod):
    """The crossing's tower: stacked octagonal lanterns of lancets, each ringed by pinnacles, and the needle spire."""
    y = CROSS_Y
    tiers = [(TC_Z + NAVE_H, 36.0, 64.0), (TC_Z + NAVE_H + 64, 28.0, 52.0), (TC_Z + NAVE_H + 116, 20.0, 44.0), (TC_Z + NAVE_H + 160, 14.0, 34.0)]
    for k, (z0, r, h) in enumerate(tiers):
        n = 8
        pts = [(r * math.cos(TAU * i / n + math.pi / 8), y + r * math.sin(TAU * i / n + math.pi / 8)) for i in range(n)]
        for i in range(n):
            wall_run(g, pts[i], pts[(i + 1) % n], z0, h, 2.6, 1e9, lod, kind='window')
        lathe(g, [(r * 1.08, 0), (0.01, 0)], n, CREAM, T(0, y, z0), phase=math.pi / 8)
        for px, py in pts:
            pinnacle(g, 3.6 - k * 0.5, 26 - k * 3, T(px * 1.06, y + (py - y) * 1.06, z0 + h), lod)
        if lod <= 1:
            balustrade(g, [(px, py, z0 + h) for px, py in pts], 1.4, None, lod, closed=True)
        for px, py in pts:  # flying ribs (buttress-shafts) up each angle
            box(g, 2.4, 2.4, h, CREAM, T(px * 1.03, y + (py - y) * 1.03, z0))
    z0 = tiers[-1][0] + tiers[-1][2]
    lathe(g, [(12, 0), (9, 10), (3.5, SPIRE_TOP - z0 - 30), (0, SPIRE_TOP - z0 - 26)], 8, CREAM, T(0, y, z0), phase=math.pi / 8)
    for k in range(5 if lod < 2 else 2):
        zz = 18 + k * 26
        r = 9 - 5.5 * zz / (SPIRE_TOP - z0 - 30) + 0.7
        torus(g, r, 0.9, 16 if lod == 0 else 8, 4, GOLD, T(0, y, z0 + zz))
    finial(g, 40, T(0, y, SPIRE_TOP - 34), lod)
    # a gold star at the very top
    star = [((7 if i % 2 == 0 else 2.2) * math.cos(TAU * i / 8 + math.pi / 2), (7 if i % 2 == 0 else 2.2) * math.sin(TAU * i / 8 + math.pi / 2)) for i in range(8)]
    prism(g, star, -0.8, 0.8, GOLD, chain(T(0, y, SPIRE_TOP + 4), Rx(math.pi / 2)))


def halo(g, lod):
    """The golden halo ring behind the central spire: twin rings, radial spokes, four star points, filigree."""
    R = 140.0
    c = (0, CROSS_Y + 34, TC_Z + NAVE_H + 190)
    M = chain(T(*c), Rx(math.pi / 2))
    n = 96 if lod == 0 else 40 if lod == 1 else 24
    glow = HALO_GOLD
    torus(g, R, 3.6, n, 8 if lod == 0 else 5, glow, M)
    torus(g, R * 0.9, 1.6, n, 6 if lod == 0 else 4, glow, M)
    torus(g, R * 1.07, 1.0, n, 4, glow, M)
    spokes = 24 if lod == 0 else 12
    for k in range(spokes):
        a = TAU * k / spokes
        r0 = R * (0.55 if k % 2 == 0 else 0.75)
        beam(g, (r0 * math.cos(a), r0 * math.sin(a), -0.6), (R * 0.9 * math.cos(a), R * 0.9 * math.sin(a), -0.6), 1.0, 1.2, glow, M)
    if lod <= 1:
        torus(g, R * 0.55, 0.9, n // 2, 4, glow, M)
        for k in range(16):  # filigree: small rings between the twin rings
            a = TAU * (k + 0.5) / 16
            torus(g, R * 0.045, 0.45, 12, 3, glow, chain(M, T(R * 0.95 * math.cos(a), R * 0.95 * math.sin(a), 0)))
    for k in range(4):  # the four points of the star, beyond the ring
        a = TAU * k / 4 + math.pi / 2
        L = 60 if k % 2 == 0 else 46
        w = 9.0
        diamond = [(R * 0.86, 0), (R + 8, w), (R + L, 0), (R + 8, -w)]
        prism(g, diamond, -1.6, 1.6, glow, chain(M, Rz(a)))
        if lod <= 1:
            sphere(g, 5, 8, glow, chain(M, Rz(a), T(R + 6, 0, 0)))


# ------------------------------------------------------------------ towers and arcades

def grand_tower(g, gi, x, y, lod, rnd, door):
    """A grand tower: a square keep (doors in the walls `door` lists: 0 south, 1 east, 2 north, 3 west; banners on the
    others), octagonal tiers, an onion dome."""
    z = TB_Z
    s = 42.0
    cs = rect_corners(x, y, s / 2, s / 2)
    for i in range(4):
        wall_run(g, cs[i], cs[(i + 1) % 4], z, 84, 3.2, s / 3, lod, kind='window', door=1 if i in door else None)
    for cx, cy in cs:
        box(g, 7, 7, 88, MARBLE, T(cx, cy, z))
        box(g, 7.6, 7.6, 1.2, GOLD, T(cx, cy, z + 60))
        lathe(g, [(5.0, 0), (5.0, 14), (0.01, 14)], 8, MARBLE, T(cx, cy, z + 88))
        dome(g, 5.4, 13, T(cx, cy, z + 102), lod, 'bell')
    box(g, s, s, 1.0, CREAM, T(x, y, z + 84))
    if lod <= 1:
        balustrade(g, [(cx, cy, z + 85) for cx, cy in cs], 1.6, None, lod, closed=True)
    # banners on the outward faces
    for i in range(4):
        a, b = cs[i], cs[(i + 1) % 4]
        if i in door: continue
        mx, my = (a[0] + b[0]) / 2, (a[1] + b[1]) / 2
        nx, ny = (mx - x) / (s / 2), (my - y) / (s / 2)
        banner(g, 9, 44, chain(T(mx + nx * 2.2, my + ny * 2.2, z + 80), Rz(math.atan2(ny, nx) + math.pi / 2)), lod)
    tiers = [(z + 85, 19.0, 62.0), (z + 147, 15.0, 50.0), (z + 197, 12.5, 22.0)]
    for k, (z0, r, h) in enumerate(tiers):
        n = 8
        pts = [(x + r * math.cos(TAU * i / n + math.pi / 8), y + r * math.sin(TAU * i / n + math.pi / 8)) for i in range(n)]
        for i in range(n):
            wall_run(g, pts[i], pts[(i + 1) % n], z0, h, 2.2, 1e9, lod, kind='window' if k < 2 else 'open')
        if k < 2:
            for px, py in pts:
                pinnacle(g, 2.6, 16, T(x + (px - x) * 1.08, y + (py - y) * 1.08, z0 + h), lod)
            if lod <= 1:
                balustrade(g, [(x + (px - x) * 1.1, y + (py - y) * 1.1, z0 + h) for px, py in pts], 1.4, None, lod, closed=True)
            lathe(g, [(r * 1.18, 0), (0.01, 0)], n, CREAM, T(x, y, z0 + h), phase=math.pi / 8)
    lathe(g, [(13.0, 0), (0.01, 0)], 8, CREAM, T(x, y, z + 219), phase=math.pi / 8)
    dome(g, 13.0, 46, T(x, y, z + 219), lod, 'onion')
    tower_room(gi, x, y, z, s, lod, rnd)


def lesser_tower(g, x, y, lod, rnd, h, r=None, kind='bell', z=0.0):
    """A round tower (on the rim by default): lancet tiers, a gallery, a lapis dome."""
    r = r or 9.0 + rnd.random() * 2
    n = 10 if lod == 0 else 8 if lod == 1 else 6
    pts = [(x + r * math.cos(TAU * i / n), y + r * math.sin(TAU * i / n)) for i in range(n)]
    for i in range(n):
        wall_run(g, pts[i], pts[(i + 1) % n], z, h, 2.0, 1e9, lod, kind='window' if i % 2 else 'blind')
    lathe(g, [(r * 1.25, 0), (r * 1.25, 2), (0.01, 2)], n, CREAM, T(x, y, z + h))
    if lod <= 1:
        balustrade(g, [(x + (px - x) * 1.2, y + (py - y) * 1.2, z + h + 2) for px, py in pts], 1.3, None, lod, closed=True)
    pts2 = [(x + (px - x) * 0.75, y + (py - y) * 0.75) for px, py in pts]
    for i in range(n):
        wall_run(g, pts2[i], pts2[(i + 1) % n], z + h + 2, 16, 1.6, 1e9, lod, kind='open')
    lathe(g, [(r * 0.8, 0), (0.01, 0)], n, CREAM, T(x, y, z + h + 18))
    dome(g, r * 0.78, r * (1.6 if kind == 'bell' else 2.4), T(x, y, z + h + 18), lod, kind)


def arcade(g, a, b, z, lod, levels=2, h=26.0):
    """A two-storey arcade bridge from a to b (2-D): open arches both sides of a walkable deck each storey."""
    M, L = seg_frame(a, b, z)
    w = 14.0
    for k in range(levels):
        zz = k * h
        if lod >= 2:
            box(g, L, w + 2, h, MARBLE, chain(M, T(0, 0, zz)))
            continue
        for s in (-1, 1):
            m = chain(M, T(0, s * w / 2, zz))
            n = max(1, round(L / 11)); bw = L / n
            for i in range(n):
                arch_wall(g, bw, h, 2.0, bw * 0.72, 0, h * 0.55, bw * 0.5, MARBLE, chain(m, T(-L / 2 + bw * (i + 0.5), 0, 0)), n=6 if lod == 0 else 3, reveal=CREAM)
        box(g, L, w + 3, 1.4, CREAM, chain(M, T(0, 0, zz + h)))
        beam(g, (-L / 2, -w / 2 - 1.2, zz + h - 1), (L / 2, -w / 2 - 1.2, zz + h - 1), 0.8, 1.0, GOLD, M)
        beam(g, (-L / 2, w / 2 + 1.2, zz + h - 1), (L / 2, w / 2 + 1.2, zz + h - 1), 0.8, 1.0, GOLD, M)
    top = levels * h + 1.4
    for s in (-1, 1):
        balustrade(g, [(-L / 2, s * (w / 2 + 0.8), top), (L / 2, s * (w / 2 + 0.8), top)], 1.5, M, lod)
    if lod <= 1:
        n = max(1, round(L / 22))
        for i in range(n + 1):
            for s in (-1, 1):
                pinnacle(g, 2.4, 13, chain(M, T(-L / 2 + L * i / n, s * (w / 2 + 0.8), top)), lod)


# ------------------------------------------------------------------ interiors

def interior(gi, rnd, lod):
    """The cathedral's great hall: a checkered marble floor, clustered columns and arcades, chandeliers, banners, the throne."""
    if lod > 0: return
    z = TC_Z + 0.06
    hw = NAVE_W / 2 - 1.6
    y0, y1 = NAVE_Y0 + 3, NAVE_Y0 + NAVE_L
    t = 4.0
    for ix in range(int(-hw / t), int(hw / t)):
        for iy in range(int(y0 / t), int(y1 / t)):
            c = FLOOR_DARK if (ix + iy) % 2 else FLOOR_LIGHT
            x0, yy = ix * t, iy * t
            gi.add([(x0, yy, z), (x0 + t, yy, z), (x0 + t, yy + t, z), (x0, yy + t, z)], [(0, 1, 2, 3)], c)
    # the arcades: columns down both sides of the nave, pointed arches between them, a clerestory gallery above
    cx = 24.0
    sp = 13.0
    ys = np.arange(NAVE_Y0 + 30, y1 - 8, sp)
    for s in (-1, 1):
        for y in ys:
            if abs(y - CROSS_Y) < 30: continue
            column(gi, 1.8, 44, T(s * cx, y, z), lod)
            banner(gi, 4, 18, chain(T(s * (cx - 2.6), y, z + 40), Rz(math.pi / 2 * s)), lod)
        for ya, yb in zip(ys, ys[1:]):
            if abs(ya - CROSS_Y) < 30 or abs(yb - CROSS_Y) < 30: continue
            m = chain(T(s * cx, (ya + yb) / 2, z + 44), Rz(math.pi / 2))
            arch_wall(gi, sp, 18, 2.4, sp - 2.0, 0, 2, (sp - 2) * 0.55, MARBLE, m, n=6, reveal=GOLD)
            arch_wall(gi, sp, 8, 2.0, sp * 0.6, 0, 3.5, sp * 0.35, CREAM, chain(m, T(0, 0, 18)), n=5, reveal=SHADE)
    # transverse ribs across the vault, gold
    for y in ys[::2]:
        beam(gi, (-hw, y, TC_Z + NAVE_H - 2), (0, y, TC_Z + NAVE_H + 34), 1.4, 1.4, GOLD, None)
        beam(gi, (hw, y, TC_Z + NAVE_H - 2), (0, y, TC_Z + NAVE_H + 34), 1.4, 1.4, GOLD, None)
    # chandeliers down the nave
    for y in np.arange(NAVE_Y0 + 40, y1 - 20, 34):
        chandelier(gi, 0, y, TC_Z + 44, TC_Z + NAVE_H + 30, lod)
    chandelier(gi, 0, CROSS_Y, TC_Z + 52, TC_Z + NAVE_H + 30, lod, big=True)
    # the throne in the apse: a dais of steps, the throne, gold standards, a carpet up the nave
    ya = y1 - 6
    for k in range(5):
        lathe(gi, [(30 - k * 4, 0), (30 - k * 4, 1.0), (0.01, 1.0)], 24, FLOOR_LIGHT if k % 2 else MARBLE, T(0, ya, z + k * 1.0))
    tz = z + 5
    box(gi, 9, 7, 3, GOLD, T(0, ya + 2, tz))
    box(gi, 9, 1.6, 16, THRONE, T(0, ya + 5, tz + 3))
    lathe(gi, [(4.6, 0), (4.6, 1), (0.01, 1)], 3, GOLD, chain(T(0, ya + 5, tz + 19), Rx(math.pi / 2), Rz(math.pi / 2)), cap_bot=False)
    finial(gi, 6, T(0, ya + 5, tz + 19), lod)
    for s in (-1, 1):
        box(gi, 1.4, 7, 6, GOLD, T(s * 5.2, ya + 2, tz + 3))
        cylinder(gi, 0.4, 22, 6, GOLD, T(s * 14, ya - 6, z + 2))
        banner(gi, 4, 14, T(s * 14, ya - 6.4, z + 22), lod)
    box(gi, 9, y1 - y0 - 36, 0.12, CARPET, T(0, (y0 + y1 - 36) / 2 + 2, z))
    # glowing light shafts are the windows themselves (stained glass glows from within): nothing more here


def chandelier(gi, x, y, z, zc, lod, big=False):
    """A gold chandelier hung from the vault: a chain, two rings of candles, a finial drop."""
    k = 1.6 if big else 1.0
    beam(gi, (x, y, z + 6 * k), (x, y, zc), 0.4, 0.4, GOLD_DARK, None)
    for r, zz, nc in ((7.0 * k, z, 16), (4.2 * k, z + 4 * k, 10)):
        torus(gi, r, 0.35 * k, 24, 4, GOLD, T(x, y, zz))
        for i in range(nc):
            a = TAU * i / nc
            px, py = x + r * math.cos(a), y + r * math.sin(a)
            cylinder(gi, 0.22 * k, 1.4 * k, 5, CREAM, T(px, py, zz))
            cone(gi, 0.24 * k, 0.8 * k, 4, FLAME, T(px, py, zz + 1.4 * k))
        for i in range(4):
            a = TAU * i / 4
            beam(gi, (x, y, z + 6 * k), (x + r * math.cos(a), y + r * math.sin(a), zz), 0.25, 0.25, GOLD, None)
    lathe(gi, [(0, 0), (1.2 * k, 1.2 * k), (0.4 * k, 3 * k), (0, 4 * k)], 8, GOLD, T(x, y, z - 3 * k))


def tower_room(gi, x, y, z, s, lod, rnd):
    """A grand tower's hall: a checkered floor, four columns, a chandelier, banners, a vaulted gold boss."""
    if lod > 0: return
    t = 3.4
    h = s / 2 - 3.6
    for ix in range(int(-h / t), int(h / t)):
        for iy in range(int(-h / t), int(h / t)):
            c = FLOOR_DARK if (ix + iy) % 2 else FLOOR_LIGHT
            x0, y0 = x + ix * t, y + iy * t
            gi.add([(x0, y0, z + 0.06), (x0 + t, y0, z + 0.06), (x0 + t, y0 + t, z + 0.06), (x0, y0 + t, z + 0.06)], [(0, 1, 2, 3)], c)
    for sx in (-1, 1):
        for sy in (-1, 1):
            column(gi, 1.1, 28, T(x + sx * 8, y + sy * 8, z), lod)
    box(gi, s - 6.6, s - 6.6, 1.2, SHADE, T(x, y, z + 40))
    chandelier(gi, x, y, z + 24, z + 40, lod)
    for k in range(4):
        a = k * math.pi / 2
        banner(gi, 4, 16, chain(T(x + (h - 0.4) * math.cos(a), y + (h - 0.4) * math.sin(a), z + 30), Rz(a + math.pi / 2)), lod)


# ------------------------------------------------------------------ courts, chapels, the palace, the loggia

def chapel(g, x, y, lod, r=15.0, h=40.0, z=TB_Z):
    """A round domed chapel: lancet walls, a ring of pinnacles, a lapis onion dome on a drum."""
    n = 12 if lod == 0 else 8
    pts = [(x + r * math.cos(TAU * i / n), y + r * math.sin(TAU * i / n)) for i in range(n)]
    for i in range(n):
        wall_run(g, pts[i], pts[(i + 1) % n], z, h, 2.4, 1e9, lod, kind='window')
    lathe(g, [(r * 1.08, 0), (0.01, 0)], n, CREAM, T(x, y, z + h))
    for px, py in pts[::2]:
        pinnacle(g, 2.2, 13, T(x + (px - x) * 1.07, y + (py - y) * 1.07, z + h), lod)
    pts2 = [(x + (px - x) * 0.7, y + (py - y) * 0.7) for px, py in pts]
    for i in range(n):
        wall_run(g, pts2[i], pts2[(i + 1) % n], z + h, 12, 1.8, 1e9, lod, kind='window')
    lathe(g, [(r * 0.75, 0), (0.01, 0)], n, CREAM, T(x, y, z + h + 12))
    dome(g, r * 0.72, r * 1.5, T(x, y, z + h + 12), lod, 'onion')


def fountain(g, x, y, lod, z=TB_Z, r=16.0):
    """A round pool with a gold rim and a tiered marble fountain crowned in gold."""
    pool = [(x + r * math.cos(a), y + r * math.sin(a)) for a in np.linspace(0, TAU, 25)[:-1]]
    prism(g, pool, z - 0.4, z + 0.03, POOL_BED)
    water_surface(g, pool, z + 0.8, 0.77)
    lathe(g, [(r, 0), (r + 1.2, 0), (r + 1.2, 1.1), (r, 1.1)], 24 if lod == 0 else 10, CREAM, T(x, y, z), cap_top=False, cap_bot=False)
    lathe(g, [(0.01, 0), (2.4, 0), (1.2, 3), (5.5, 4.2), (5.5, 4.8), (1.0, 5.2), (0.8, 8), (2.8, 8.6), (0.5, 9.5), (0.01, 9.5)], 12 if lod == 0 else 6, MARBLE, T(x, y, z))
    finial(g, 6, T(x, y, z + 9.5), lod)
    for k in range(10):
        a = TAU * k / 10
        bush(g, 4.4, T(x + (r + 5) * math.cos(a), y + (r + 5) * math.sin(a), z), 'blossom' if k % 2 else 'mossy')


def court(g, rnd, x0, x1, y0, y1, lod):
    """A side court between the nave and the side arcade: two domed chapels, a fountain, cypress walks."""
    cx = (x0 + x1) / 2
    chapel(g, cx, y0 + 30, lod)
    chapel(g, cx, y1 - 30, lod, r=13.0, h=34.0)
    fountain(g, cx, (y0 + y1) / 2, lod)
    if lod <= 1:
        for k in range(6):
            yy = (y0 + y1) / 2 - 50 + k * 20
            if abs(yy - (y0 + y1) / 2) < 22: continue
            for sx in (-1, 1):
                cypress(g, 13 + rnd.random() * 3, T(cx + sx * 22, yy, TB_Z), lod)


def palace(g, gi, rnd, lod):
    """The palace behind the apse: a long two-storey block of windows, a central domed pavilion, banners."""
    y0, y1 = NAVE_Y0 + NAVE_L + 36, TB[1] - 6
    hx = 112.0
    cs = rect_corners(0, (y0 + y1) / 2, hx, (y1 - y0) / 2)
    for i in range(4):
        wall_run(g, cs[i], cs[(i + 1) % 4], TB_Z, 26, 2.8, 9.0, lod, kind='window', door=12 if i == 0 else None)
        wall_run(g, cs[i], cs[(i + 1) % 4], TB_Z + 26, 22, 2.8, 9.0, lod, kind='window', warm=True)
    palace_hall(gi, 0, (y0 + y1) / 2, hx - 2, (y1 - y0) / 2 - 2, lod)
    roof(g, (-hx, (y0 + y1) / 2), (hx, (y0 + y1) / 2), (y1 - y0) / 2 + 1, TB_Z + 48, 18, lod)
    chapel(g, 0, (y0 + y1) / 2, lod, r=17.0, h=58.0)
    for s in (-1, 1):
        chapel(g, s * (hx - 4), (y0 + y1) / 2, lod, r=10.0, h=56.0)
        for k in range(3):
            banner(g, 6, 26, T(s * (24 + k * 26), y0 - 1.8, TB_Z + 46), lod)


def palace_hall(gi, cx, cy, hx, hy, lod):
    """The palace's gallery: a checkered floor, a colonnade down its middle, chandeliers, banners on the piers, a gold-ribbed ceiling."""
    if lod > 0: return
    z = TB_Z + 0.06
    t = 3.0
    for ix in range(int(-hx / t), int(hx / t)):
        for iy in range(int(-hy / t), int(hy / t)):
            c = FLOOR_DARK if (ix + iy) % 2 else FLOOR_LIGHT
            x0, y0 = cx + ix * t, cy + iy * t
            gi.add([(x0, y0, z), (x0 + t, y0, z), (x0 + t, y0 + t, z), (x0, y0 + t, z)], [(0, 1, 2, 3)], c)
    for x in np.arange(-hx + 14, hx - 10, 14):
        if abs(x) < 20: continue  # (the chapel's drum stands in the middle)
        for s in (-1, 1):
            column(gi, 1.2, 30, T(cx + x, cy + s * 7, z), lod)
        chandelier(gi, cx + x + 7, cy, TB_Z + 30, TB_Z + 46, lod)
        banner(gi, 3.5, 14, chain(T(cx + x, cy - hy + 0.6, TB_Z + 34)), lod)
        banner(gi, 3.5, 14, chain(T(cx + x, cy + hy - 0.6, TB_Z + 34), Rz(math.pi)), lod)
    box(gi, hx * 2, 1.2, 1.2, GOLD, T(cx, cy - 7, TB_Z + 30))
    box(gi, hx * 2, 1.2, 1.2, GOLD, T(cx, cy + 7, TB_Z + 30))
    for x in np.arange(-hx, hx, 14):
        beam(gi, (cx + x, cy - hy, TB_Z + 46), (cx + x, cy + hy, TB_Z + 46), 1.0, 1.0, GOLD, None)


def loggia(g, rnd, lod):
    """A one-storey arcade round the island's edge (a walk under arches, a terrace on top), broken by a gatehouse at the front."""
    poly = outline(56 if lod < 2 else 28)
    ring = [(x * 0.9, y * 0.9) for x, y in poly]
    n = len(ring)
    for i in range(n):
        a, b = ring[i], ring[(i + 1) % n]
        mx, my = (a[0] + b[0]) / 2, (a[1] + b[1]) / 2
        if my < 0 and abs(mx) < 50: continue  # the gate
        arcade(g, a, b, 0, lod, levels=1, h=15.0)
    # the gatehouse: two towers flanking the way in, an arch between, banners
    gy = min(y for x, y in ring if abs(x) < 50)
    for s in (-1, 1):
        lesser_tower(g, s * 46, gy, lod, rnd, 44)
        banner(g, 6, 22, chain(T(s * 46, gy - 11.5, 34)), lod)
    m = T(0, gy, 0)
    arch_wall(g, 80, 46, 8, 34, 0, 26, 18, MARBLE, m, n=10 if lod == 0 else 4, reveal=GOLD)
    balustrade(g, [(-40, gy - 4, 46), (40, gy - 4, 46)], 1.6, None, lod)
    if lod <= 1:
        for i in range(5):
            pinnacle(g, 2.4, 14, T(-32 + i * 16, gy - 3.6, 46), lod)


# ------------------------------------------------------------------ the whole

def build(lod, seed=7):
    """The castle at a level of detail: (exterior, interior, extras) — the extras (kit.Extras) at lod 0 only, else None."""
    global RND
    rnd = random.Random(seed)
    RND = random.Random(seed + 1)
    extras = Extras() if lod == 0 else None
    g, gi = Geo(lod, extras), Geo(lod, extras)
    island(g, rnd, lod)
    terraces(g, rnd, lod)
    cathedral(g, gi, rnd, lod)
    hx, hy = TB
    gx, gy = hx - 30, hy - 30
    for fx, fy in ((-1, -1), (1, -1), (1, 1), (-1, 1)):
        grand_tower(g, gi, fx * gx, fy * gy, lod, rnd, {(-1, -1): (0, 3), (1, -1): (0, 1), (1, 1): (2, 1), (-1, 1): (2, 3)}[(fx, fy)])
    # arcades: the front towers to the facade towers, and down each side between the grand towers
    fy = NAVE_Y0 + 13
    for s in (-1, 1):
        arcade(g, (s * (gx - 17), -gy), (s * (NAVE_W / 2 + 26), fy), TB_Z, lod)
        arcade(g, (s * gx, -gy + 17), (s * gx, gy - 17), TB_Z, lod, levels=2, h=24.0)
        arcade(g, (s * (gx - 17), gy), (s * (NAVE_W / 2 + 6), NAVE_Y0 + NAVE_L), TB_Z, lod, levels=1, h=30.0)
    for s in (-1, 1):
        x0, x1 = sorted((s * (NAVE_W / 2 + 44), s * (gx - 12)))
        court(g, rnd, x0, x1, -gy + 30, NAVE_Y0 + NAVE_L - 10, lod)
    palace(g, gi, rnd, lod)
    loggia(g, rnd, lod)
    # lesser towers on the loggia, clear of the gate
    for k in range(10):
        a = TAU * k / 10 + 0.31
        if abs(math.sin(a) + 1) < 0.25: continue
        poly_r = ISLAND_R * (1 + 0.05 * math.sin(3 * a + 1.0) + 0.035 * math.sin(5 * a + 2.0) + 0.02 * math.sin(9 * a)) * 0.9
        lesser_tower(g, poly_r * math.cos(a), poly_r * 0.93 * math.sin(a), lod, rnd, 60 + rnd.random() * 40)
    spouts(g, rnd, lod)
    return colossal(g, gi, extras)
