"""
The gothic kit the sky castle is built from (castle.py): a geometry accumulator and the pieces.
-----------------------------------------------------------------------------
Blender axes (z up), metres. A piece is made in its own axes and placed by a 4x4 matrix (T, Rz, S
compose). Every face carries a colour: (r, g, b, glow, material), sRGB albedo, how much it glows (0..1) and its
material (M_*: the texture the game paints on it, modelIsles.wgsl.ts; gold is metal). Besides the solids a build
records Extras: the instanced props, the plants (the land's trees and bushes grow there: gpu/modelIsles.ts), the
waterfalls, the ivy (leaf cards) and the water (surfaces over the pools' beds).
`lod` (0 finest .. 2 coarsest) thins the segment counts and drops the small ornaments.
"""

import math
import numpy as np
from mathutils.geometry import tessellate_polygon

TAU = math.tau


# ------------------------------------------------------------------ placing

def T(x=0.0, y=0.0, z=0.0):
    m = np.eye(4); m[:3, 3] = (x, y, z); return m


def Rz(a):
    c, s = math.cos(a), math.sin(a)
    m = np.eye(4); m[0, 0] = c; m[0, 1] = -s; m[1, 0] = s; m[1, 1] = c; return m


def Rx(a):
    c, s = math.cos(a), math.sin(a)
    m = np.eye(4); m[1, 1] = c; m[1, 2] = -s; m[2, 1] = s; m[2, 2] = c; return m


def Ry(a):
    c, s = math.cos(a), math.sin(a)
    m = np.eye(4); m[0, 0] = c; m[0, 2] = s; m[2, 0] = -s; m[2, 2] = c; return m


def S(x, y=None, z=None):
    m = np.eye(4); m[0, 0] = x; m[1, 1] = x if y is None else y; m[2, 2] = x if z is None else z; return m


def chain(*ms):
    out = np.eye(4)
    for m in ms: out = out @ m
    return out


class Geo:
    """Vertices (with uvs), faces and a colour a face; `x` the build's Extras (None: a coarse level, which records none)."""

    def __init__(self, lod=0, x=None):
        self.lod = lod
        self.v = []      # arrays (n, 3)
        self.uv = []     # arrays (n, 2)
        self.f = []      # lists of index tuples (offset applied)
        self.c = []      # one colour per face
        self.n = 0
        self.x = x

    def prop(self, kind, M):
        if self.x: self.x.props.append((kind, M))

    def plant(self, species, M, scale, palette):
        """A plant of the land's forest (forest/species.ts; palette: skyworld/biomes.ts SKY_FLORA) standing at M's origin."""
        if self.x: self.x.plants.append([species, np.asarray(M)[:3, 3].copy(), scale, palette])

    def fall(self, lip, direction, width, reach):
        """A waterfall pouring from lip (x, y, z) along direction (x, y, unit), width wide, all mist `reach` m down."""
        if self.x: self.x.falls.append([np.asarray(lip, float), np.asarray(direction, float), width, reach])

    def add(self, verts, faces, col, M=None, uv=None):
        v = np.asarray(verts, np.float64).reshape(-1, 3)
        if M is not None:
            v = v @ M[:3, :3].T + M[:3, 3]
        b = self.n
        self.v.append(v)
        self.uv.append(np.zeros((len(v), 2)) if uv is None else np.asarray(uv, np.float64).reshape(-1, 2))
        self.f += [tuple(b + i for i in f) for f in faces]
        self.c += [col] * len(faces)
        self.n += len(v)

    def extend(self, other, M=None):
        v = np.concatenate(other.v) if other.v else np.zeros((0, 3))
        if M is not None and len(v):
            v = v @ M[:3, :3].T + M[:3, 3]
        b = self.n
        self.v.append(v)
        self.uv.append(np.concatenate(other.uv) if other.uv else np.zeros((0, 2)))
        self.f += [tuple(b + i for i in f) for f in other.f]
        self.c += other.c
        self.n += len(v)

    def arrays(self):
        return (np.concatenate(self.v) if self.v else np.zeros((0, 3))), self.f, self.c

    def uvs(self):
        return np.concatenate(self.uv) if self.uv else np.zeros((0, 2))

    def tris(self):
        return sum(len(f) - 2 for f in self.f)


class Extras:
    """What a build records besides its solids, at its finest level: props, plants, falls, the ivy and the water."""

    def __init__(self):
        self.props, self.plants, self.falls = [], [], []
        self.leaves, self.water = Geo(0), Geo(0)


# ------------------------------------------------------------------ colours

# materials (modelIsles.wgsl.ts MI_*: keep in step)
M_PLAIN, M_MARBLE, M_STONE, M_GOLD, M_ROOF, M_CLOTH, M_GLASS, M_ROCK, M_FLOOR, M_POOL, M_LEAF = range(11)

MARBLE = (0.93, 0.90, 0.84, 0.0, M_MARBLE)
CREAM = (0.86, 0.80, 0.69, 0.0, M_STONE)
SHADE = (0.74, 0.70, 0.64, 0.0, M_STONE)        # recessed stone
GOLD = (0.86, 0.64, 0.24, 0.0, M_GOLD)
GOLD_DARK = (0.62, 0.44, 0.16, 0.0, M_GOLD)
HALO_GOLD = (0.95, 0.74, 0.32, 0.22, M_GOLD)    # the halos and crowns: gold that glows
ROOF = (0.14, 0.27, 0.70, 0.0, M_ROOF)          # the domes' lapis
BANNER = (0.12, 0.20, 0.55, 0.0, M_CLOTH)
THRONE = (0.13, 0.22, 0.58, 0.0, M_CLOTH)
CARPET = (0.55, 0.10, 0.12, 0.0, M_CLOTH)
GLASS = (0.16, 0.30, 0.82, 0.42, M_GLASS)       # stained glass, lit from within: deep lapis, not the sky's blue
GLASS_WARM = (0.98, 0.74, 0.40, 0.5, M_GLASS)
ARMILLARY = (0.16, 0.30, 0.82, 0.4, M_GOLD)     # the armillary's lapis globe
FLAME = (1.00, 0.80, 0.45, 1.0, M_PLAIN)
ROCK = (0.66, 0.63, 0.60, 0.0, M_ROCK)
IVY = (0.30, 0.48, 0.18, 0.0, M_LEAF)           # (leaf cards: a tint over the land's leaf sprays)
IVY_DARK = (0.20, 0.36, 0.13, 0.0, M_LEAF)
FLOOR_DARK = (0.30, 0.32, 0.40, 0.0, M_FLOOR)
FLOOR_LIGHT = (0.90, 0.88, 0.83, 0.0, M_FLOOR)
FLOOR_ALT = (0.83, 0.80, 0.74, 0.0, M_FLOOR)
POOL_BED = (0.16, 0.34, 0.55, 0.0, M_POOL)      # a pool's lapis mosaic, under the water
WATER = (0.40, 0.62, 0.78, 0.0, M_POOL)



# ------------------------------------------------------------------ primitives (own axes)

def box(g, sx, sy, sz, col, M=None, center_z=False):
    """A box sx × sy × sz, its base's middle at the origin (or its middle, center_z)."""
    x, y = sx / 2, sy / 2
    z0, z1 = (-sz / 2, sz / 2) if center_z else (0.0, sz)
    v = [(-x, -y, z0), (x, -y, z0), (x, y, z0), (-x, y, z0), (-x, -y, z1), (x, -y, z1), (x, y, z1), (-x, y, z1)]
    f = [(0, 3, 2, 1), (4, 5, 6, 7), (0, 1, 5, 4), (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7)]
    g.add(v, f, col, M)


def beam(g, a, b, w, h, col, M=None):
    """A box from point a to point b (its base's centre line), w wide, h tall."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    d = b - a
    L = math.hypot(d[0], d[1])
    ang = math.atan2(d[1], d[0])
    pitch = math.atan2(d[2], L)
    m = chain(T(*a), Rz(ang), Ry(-pitch), T(math.hypot(L, d[2]) / 2, 0, 0))
    box(g, math.hypot(L, d[2]), w, h, col, m if M is None else M @ m)


def lathe(g, prof, n, col, M=None, cap_top=True, cap_bot=True, phase=0.0):
    """Revolve a profile [(r, z)] about z with n sides; r = 0 points make poles."""
    rings, faces = [], []
    v = []
    for r, z in prof:
        if r <= 1e-6:
            rings.append([len(v)]); v.append((0, 0, z))
        else:
            ring = []
            for k in range(n):
                a = phase + TAU * k / n
                ring.append(len(v)); v.append((r * math.cos(a), r * math.sin(a), z))
            rings.append(ring)
    for A, B in zip(rings, rings[1:]):
        if len(A) == 1 and len(B) == 1: continue
        if len(A) == 1:
            faces += [(A[0], B[k], B[(k + 1) % n]) for k in range(n)]
        elif len(B) == 1:
            faces += [(A[k], B[0], A[(k + 1) % n]) for k in range(n)]
        else:
            faces += [(A[k], A[(k + 1) % n], B[(k + 1) % n], B[k]) for k in range(n)]
    if cap_bot and len(rings[0]) > 1: faces.append(tuple(reversed(rings[0])))
    if cap_top and len(rings[-1]) > 1: faces.append(tuple(rings[-1]))
    g.add(v, faces, col, M)


def prism(g, poly, z0, z1, col, M=None, top=True, bottom=True, col_side=None):
    """A 2-D polygon [(x, y)] (counter-clockwise) extruded from z0 to z1."""
    n = len(poly)
    v = [(x, y, z0) for x, y in poly] + [(x, y, z1) for x, y in poly]
    side = [(i, (i + 1) % n, n + (i + 1) % n, n + i) for i in range(n)]
    g.add(v, side, col_side or col, M)
    caps = []
    tri = tessellate_polygon([[(x, y, 0) for x, y in poly]])
    if bottom: caps += [(c, b, a) for a, b, c in tri]
    if top: caps += [(n + a, n + b, n + c) for a, b, c in tri]
    if caps: g.add(v, caps, col, M)


def cylinder(g, r, h, n, col, M=None, r_top=None):
    lathe(g, [(r, 0), (r if r_top is None else r_top, h)], n, col, M)


def cone(g, r, h, n, col, M=None):
    lathe(g, [(r, 0), (0, h)], n, col, M)


def sphere(g, r, n, col, M=None, rings=None):
    rings = rings or max(3, n // 2)
    prof = [(r * math.sin(math.pi * i / rings), -r * math.cos(math.pi * i / rings)) for i in range(rings + 1)]
    lathe(g, prof, n, col, M)


def torus(g, R, r, n, m, col, M=None, arc=TAU):
    v, f = [], []
    closed = abs(arc - TAU) < 1e-6
    N = n if closed else n + 1
    for i in range(N):
        a = arc * i / n
        for j in range(m):
            b = TAU * j / m
            v.append(((R + r * math.cos(b)) * math.cos(a), (R + r * math.cos(b)) * math.sin(a), r * math.sin(b)))
    for i in range(n):
        i2 = (i + 1) % N
        for j in range(m):
            j2 = (j + 1) % m
            f.append((i * m + j, i2 * m + j, i2 * m + j2, i * m + j2))
    g.add(v, f, col, M)


# ------------------------------------------------------------------ gothic profiles

def pointed_arch(w, spring, rise, n):
    """Points of a pointed arch over a span w (x from -w/2 to w/2), springing at z = spring, its apex rise above."""
    # the left arc is centred on the spring line at cx (the right arc mirrors it); its radius fits span and rise
    h = w / 2
    R = (h * h + rise * rise) / (2 * h)
    cx = R - h
    t_end = math.acos(max(-1.0, min(1.0, cx / R)))
    left = [(cx - R * math.cos(t_end * i / n), spring + R * math.sin(t_end * i / n)) for i in range(n + 1)]
    left[-1] = (0.0, spring + rise)
    return left + [(-x, z) for x, z in reversed(left[:-1])]


def arch_wall(g, w, h, t, ow, sill, spring, rise, col, M=None, n=6, reveal=None):
    """
    A wall panel w wide, h tall, t thick (in the xz plane, y from -t/2 to t/2) pierced by a pointed opening
    ow wide from sill to spring + rise. Returns the opening's outline [(x, z)] for its glass or tracery.
    """
    y0, y1 = -t / 2, t / 2
    rc = reveal or col
    hw = w / 2
    arch = pointed_arch(ow, spring, rise, n)
    # below the sill, the piers either side, then the spandrel above the arch: quads from each arch point up to the top
    if sill > 0: box(g, w, t, sill, col, M)
    pw = (w - ow) / 2
    if pw > 1e-3:
        for s in (-1, 1):
            box(g, pw, t, h - sill, col, chain(M if M is not None else np.eye(4), T(s * (hw - pw / 2), 0, sill)))
    v, f = [], []
    for (xa, za), (xb, zb) in zip(arch, arch[1:]):
        k = len(v)
        v += [(xa, y0, za), (xb, y0, zb), (xb, y0, h), (xa, y0, h), (xa, y1, za), (xb, y1, zb), (xb, y1, h), (xa, y1, h)]
        f += [(k, k + 1, k + 2, k + 3), (k + 5, k + 4, k + 7, k + 6)]
    g.add(v, f, col, M)
    # the reveal round the opening (its soffit), and the outer piers' tops above the springing
    v, f = [], []
    for (xa, za), (xb, zb) in zip(arch, arch[1:]):
        k = len(v)
        v += [(xa, y0, za), (xb, y0, zb), (xb, y1, zb), (xa, y1, za)]
        f.append((k, k + 3, k + 2, k + 1))
    g.add(v, f, rc, M)
    return [(-ow / 2, sill)] + arch + [(ow / 2, sill)]


def glass(g, outline, y, col, M=None, frame=None, mullions=0, lod=0):
    """A pane over an opening's outline [(x, z)] at depth y, with gold tracery (mullions) when fine enough."""
    tri = tessellate_polygon([[(x, z, 0) for x, z in outline]])
    v = [(x, y, z) for x, z in outline]
    g.add(v, [tuple(t) for t in tri] + [tuple(reversed(t)) for t in tri], col, M)
    if frame and lod == 0 and mullions:
        xs = [x for x, _ in outline]; zs = [z for _, z in outline]
        x0, x1, z0, z1 = min(xs), max(xs), min(zs), max(zs)
        bw = (x1 - x0) * 0.035
        for i in range(1, mullions + 1):
            x = x0 + (x1 - x0) * i / (mullions + 1)
            box(g, bw, bw * 1.6, (z1 - z0) * 0.72, frame, chain(M if M is not None else np.eye(4), T(x, y, z0)))
        for f in (0.3, 0.55):
            box(g, x1 - x0, bw * 1.6, bw, frame, chain(M if M is not None else np.eye(4), T(0, y, z0 + (z1 - z0) * f)))
        torus(g, (x1 - x0) * 0.2, bw * 0.5, 10, 4, frame, chain(M if M is not None else np.eye(4), T(0, y, z0 + (z1 - z0) * 0.8), Rx(math.pi / 2)))


def gothic_window(g, w, h, t, col, M=None, lod=0, warm=False, frame=GOLD):
    """A wall bay w × h with a tall lancet window: the arch_wall and its stained glass (far off: a pane on a plain wall)."""
    ow = w * 0.56
    spring = h * 0.62
    rise = ow * 0.95
    sill = h * 0.12
    if lod >= 2:
        box(g, w, t, h, col, M)
        out = [(-ow / 2, sill)] + pointed_arch(ow, spring, rise, 2) + [(ow / 2, sill)]
        glass(g, out, -t / 2 - 0.15, GLASS_WARM if warm else GLASS, M)
        return
    out = arch_wall(g, w, h, t, ow, sill, spring, rise, col, M, n=6 if lod == 0 else 3, reveal=SHADE)
    glass(g, out, 0.0, GLASS_WARM if warm else GLASS, M, frame, 3 if w > 6 else 2 if w > 3 else 1, lod)


# ------------------------------------------------------------------ ornaments

def protos(lod):
    """The instanced pieces at a level of detail: kind -> Geo in the piece's own axes."""
    out = {}
    g = Geo(lod); pinnacle_mesh(g, 1.0, PINNACLE_H, None, lod); out['pinnacle'] = g
    g = Geo(lod); baluster_mesh(g, lod); out['baluster'] = g
    return out


def finial(g, h, M=None, lod=0, col=GOLD):
    """A gold finial: a knob, a collar and a needle."""
    n = 8 if lod == 0 else 4
    lathe(g, [(0, 0), (0.18 * h, 0.05 * h), (0.07 * h, 0.15 * h), (0.16 * h, 0.3 * h), (0.07 * h, 0.42 * h), (0.035 * h, 0.5 * h), (0, h)], n, col, M)


def pinnacle(g, s, h, M=None, lod=0, col=MARBLE, tip=GOLD):
    """A gothic pinnacle on an s-wide square (an instanced prop: PROTOS['pinnacle'] is 1 wide, PINNACLE_H tall)."""
    g.prop('pinnacle', chain(M if M is not None else np.eye(4), S(s, s, h / PINNACLE_H)))


PINNACLE_H = 6.0


def pinnacle_mesh(g, s, h, M=None, lod=0, col=MARBLE, tip=GOLD):
    """The pinnacle itself: shaft, gablets on each face, a crocketed spirelet, a gold finial."""
    if lod >= 2:
        box(g, s, s, h * 0.45, col, M)
        cone(g, s * 0.7, h * 0.55, 4, col, chain(M if M is not None else np.eye(4), T(0, 0, h * 0.45), Rz(math.pi / 4)))
        return
    base = M if M is not None else np.eye(4)
    box(g, s, s, h * 0.42, col, base)
    for k in range(4):
        m = chain(base, Rz(k * math.pi / 2), T(0, -s / 2, h * 0.36))
        prism(g, [(-s / 2, 0), (s / 2, 0), (0, s * 0.55)], -s * 0.06, s * 0.06, col, chain(m, Rx(math.pi / 2)))
    lathe(g, [(s * 0.62, h * 0.42), (s * 0.5, h * 0.48), (0.0, h * 0.92)], 4, col, base, phase=math.pi / 4)
    if lod == 0:
        for i in range(3):  # crockets climbing the spirelet's edges
            z = h * (0.52 + 0.12 * i); r = s * 0.5 * (1 - (z - h * 0.48) / (h * 0.44))
            for k in range(4):
                a = math.pi / 4 + k * math.pi / 2
                sphere(g, s * 0.09, 4, col, chain(base, T(r * math.cos(a) * 1.15, r * math.sin(a) * 1.15, z)))
    finial(g, h * 0.22, chain(base, T(0, 0, h * 0.86)), lod, tip)


def balustrade(g, pts, h, M=None, lod=0, col=MARBLE, closed=False, rail=CREAM):
    """A balustrade along a polyline [(x, y, z)]: rails, posts every ~4 m, balusters between (when fine)."""
    pts = [np.asarray(p, float) for p in pts]
    if closed: pts = pts + [pts[0]]
    for a, b in zip(pts, pts[1:]):
        d = b - a; L = float(np.linalg.norm(d[:2]))
        if L < 1e-3: continue
        beam(g, a + (0, 0, h * 0.86), b + (0, 0, h * 0.86), h * 0.22, h * 0.14, rail, M)
        if lod >= 2: continue
        beam(g, a, b, h * 0.2, h * 0.1, rail, M)
        posts = max(1, int(L / (4 if lod == 0 else 8)))
        for i in range(posts + 1):
            p = a + d * i / posts
            box(g, h * 0.24, h * 0.24, h * 0.9, col, chain(M if M is not None else np.eye(4), T(*p)))
        nb = int(L / (h * 0.33))
        yaw = math.atan2(d[1], d[0])
        for i in range(nb):
            p = a + d * (i + 0.5) / nb
            g.prop('baluster', chain(M if M is not None else np.eye(4), T(*p), Rz(yaw), S(h)))


def baluster_mesh(g, lod=0, col=MARBLE):
    """A baluster 1 tall (PROTOS['baluster']: balustrade scales it by its height)."""
    lathe(g, [(0.07, 0.1), (0.11, 0.3), (0.05, 0.55), (0.07, 0.76)], 6 if lod == 0 else 4, col, cap_top=False, cap_bot=False)


def column(g, r, h, M=None, lod=0, col=MARBLE, cap=GOLD):
    """A column: a moulded base, a clustered shaft (four shafts round a core when fine), a gold capital."""
    n = 10 if lod == 0 else 6
    base = M if M is not None else np.eye(4)
    lathe(g, [(r * 1.5, 0), (r * 1.5, r * 0.5), (r * 1.25, r * 0.8), (r * 1.1, r * 1.1)], n, col, base)
    cylinder(g, r, h - r * 2.6, n, col, chain(base, T(0, 0, r * 1.1)))
    if lod == 0:
        for k in range(4):
            a = k * math.pi / 2 + math.pi / 4
            cylinder(g, r * 0.35, h - r * 2.6, 6, col, chain(base, T(r * math.cos(a), r * math.sin(a), r * 1.1)))
    lathe(g, [(r * 1.05, h - r * 1.5), (r * 1.7, h - r * 0.4), (r * 1.7, h)], n, cap, base, cap_bot=False)


def stairs(g, w, rise, run, steps, col, M=None):
    """A flight of steps w wide climbing `rise` over `run` along +y, from the origin."""
    for i in range(steps):
        box(g, w, run / steps * (steps - i), rise / steps, col,
            chain(M if M is not None else np.eye(4), T(0, run / steps * i + run / steps * (steps - i) / 2, rise / steps * i)))


def banner(g, w, h, M=None, lod=0):
    """A hanging banner: a blue cloth with a gold border, a gold compass-star emblem in a ring, a pointed foot."""
    base = M if M is not None else np.eye(4)
    t = 0.3
    tail = w * 0.5
    prism(g, [(-w / 2, -h + tail), (w / 2, -h + tail), (w / 2, 0), (-w / 2, 0)], -t / 2, t / 2, BANNER, chain(base, Rx(math.pi / 2)))
    prism(g, [(-w / 2, -h + tail), (0, -h), (w / 2, -h + tail)], -t / 2, t / 2, BANNER, chain(base, Rx(math.pi / 2)))
    beam(g, (-w * 0.62, 0, 0.1), (w * 0.62, 0, 0.1), w * 0.06, w * 0.06, GOLD, base)
    for s in (-1, 1):
        sphere(g, w * 0.06, 6, GOLD, chain(base, T(s * w * 0.65, 0, 0.13)))
    if lod >= 2: return
    e = w * 0.03
    for s in (-1, 1):
        beam(g, (s * (w / 2 - e), -t, -h + tail), (s * (w / 2 - e), -t, -e), e * 1.4, e * 1.4, GOLD, base)
    cz = -h * 0.38
    torus(g, w * 0.3, e, 20 if lod == 0 else 10, 4, GOLD, chain(base, T(0, -t, cz), Rx(math.pi / 2)))
    star = []
    for k in range(16):
        r = w * (0.36 if k % 4 == 0 else 0.12 if k % 2 else 0.08)
        a = TAU * k / 16 + math.pi / 2
        star.append((r * math.cos(a), r * math.sin(a)))
    prism(g, star, -t * 0.8, t * 0.2, GOLD, chain(base, T(0, 0, cz), Rx(math.pi / 2)))
    if lod == 0:
        for z in (-h * 0.72, -h * 0.82):
            beam(g, (-w * 0.3, -t, z), (w * 0.3, -t, z), e, e, GOLD, base)


def dome(g, r, h, M=None, lod=0, kind='onion', col=ROOF, rib=GOLD):
    """A dome on a drum's top (z = 0): onion or bell, gold ribs, a lantern-collar and a tall gold finial."""
    n = 16 if lod == 0 else 8 if lod == 1 else 6
    if kind == 'onion':
        prof = [(r, 0), (r * 1.12, h * 0.2), (r * 1.08, h * 0.38), (r * 0.8, h * 0.58), (r * 0.42, h * 0.78), (r * 0.16, h * 0.92), (0, h)]
    else:
        prof = [(r, 0), (r * 0.98, h * 0.3), (r * 0.82, h * 0.6), (r * 0.52, h * 0.85), (r * 0.18, h * 0.97), (0, h)]
    base = M if M is not None else np.eye(4)
    lathe(g, prof, n, col, base)
    lathe(g, [(r * 1.04, -r * 0.08), (r * 1.08, 0), (r * 1.04, r * 0.08)], n, rib, base, cap_top=False, cap_bot=False)
    if lod <= 1:
        for k in range(8):
            a = TAU * k / 8
            ca, sa = math.cos(a), math.sin(a)
            for (r0, z0), (r1, z1) in zip(prof, prof[1:]):
                p0 = (r0 * 1.02 * ca, r0 * 1.02 * sa, z0); p1 = (r1 * 1.02 * ca, r1 * 1.02 * sa, z1)
                w = r * 0.035
                beam(g, (p0[0], p0[1], p0[2] - w / 2), (p1[0], p1[1], p1[2] - w / 2), w, w, rib, base)
    finial(g, h * 0.7, chain(base, T(0, 0, h * 0.95)), lod, rib)


def angel(g, h, M=None, lod=0, col=MARBLE):
    """A winged statue on a plinth: a robed figure, head and gold halo, arms holding a sword point-down, two layered wings."""
    base = M if M is not None else np.eye(4)
    n = 12 if lod == 0 else 6
    box(g, h * 0.45, h * 0.45, h * 0.32, CREAM, base)
    box(g, h * 0.5, h * 0.5, h * 0.04, GOLD, chain(base, T(0, 0, h * 0.32)))
    box(g, h * 0.4, h * 0.4, h * 0.04, CREAM, chain(base, T(0, 0, h * 0.36)))
    b = chain(base, T(0, 0, h * 0.4))
    # the robe falls in a flare to the plinth, gathers at the waist, the shoulders broaden
    lathe(g, [(h * 0.15, 0), (h * 0.13, h * 0.08), (h * 0.1, h * 0.25), (h * 0.075, h * 0.36), (h * 0.1, h * 0.44),
              (h * 0.07, h * 0.5), (h * 0.03, h * 0.52), (0, h * 0.52)], n, col, b)
    sphere(g, h * 0.05, n, col, chain(b, T(0, 0, h * 0.57)))
    torus(g, h * 0.06, h * 0.007, 16, 3, GOLD, chain(b, T(0, -h * 0.02, h * 0.62), Rx(0.3)))
    if lod <= 1:
        # arms forward to the sword's hilt; the sword, point-down before the robe
        for s in (-1, 1):
            beam(g, (s * h * 0.09, -h * 0.01, h * 0.46), (s * h * 0.025, -h * 0.1, h * 0.36), h * 0.035, h * 0.035, col, b)
        box(g, h * 0.018, h * 0.008, h * 0.32, GOLD, chain(b, T(0, -h * 0.12, h * 0.04)))
        box(g, h * 0.09, h * 0.02, h * 0.015, GOLD, chain(b, T(0, -h * 0.12, h * 0.35)))
    for s in (-1, 1):
        # a wing: long flight feathers fanned up and back, a shorter row of coverts over their roots
        w = chain(b, T(s * h * 0.05, h * 0.06, h * 0.45), Rz(-s * 1.05))
        for row, (L0, nf) in enumerate(((0.46, 9 if lod == 0 else 4), (0.24, 6 if lod == 0 else 0))):
            for k in range(nf):
                t = k / max(nf - 1, 1)
                a = math.radians(28 + 72 * t)
                L = h * L0 * (0.62 + 0.38 * math.sin(math.pi * (0.25 + 0.6 * t)))
                fw = h * (0.055 if row == 0 else 0.07)
                leaf = [(0, -fw * 0.4), (L * 0.55, -fw * 0.5), (L, 0), (L * 0.55, fw * 0.5), (0, fw * 0.4)]
                # (the leaf into the wing's plane, xz, then swung up from outward by a)
                prism(g, leaf, -h * 0.006, h * 0.006, col,
                      chain(w, T(0, h * 0.008 * (row + k * 0.15), 0), Ry(-a if s > 0 else a - math.pi), Rx(math.pi / 2)))


def cypress(g, h, M=None, lod=0):
    """A cypress: the land's pine (forest/species.ts: 17 m at scale 1), dark green, h tall."""
    g.plant('pine', M if M is not None else np.eye(4), h / 17.0, 'fern')


def bush(g, r, M=None, palette='mossy'):
    """A bush of the land's (1.6 m across at scale 1), r across."""
    g.plant('bush', M if M is not None else np.eye(4), r / 1.6, palette)


def leaf_card(L, w, h, col, M, u0=0.0, v0=0.0, u1=1.0, v1=1.0):
    """A card w × h (in its xz plane, hanging from its top edge at the origin) showing a leaf spray (uv)."""
    L.add([(-w / 2, 0, 0), (w / 2, 0, 0), (w / 2, 0, -h), (-w / 2, 0, -h)], [(0, 1, 2, 3)], col, M,
          uv=[(u0, v0), (u1, v0), (u1, v1), (u0, v1)])


def ivy(g, size, rnd, M=None, lod=0, hang=0.0):
    """Ivy, of the land's leaf sprays: a clump of crossed cards, and (hang > 0) a curtain of cards draping that far down."""
    if not g.x or lod > 0: return
    L = g.x.leaves
    base = M if M is not None else np.eye(4)
    tint = lambda: IVY if rnd.random() < 0.6 else IVY_DARK
    for k in range(3):
        leaf_card(L, size * 2.2, size * 1.6, tint(), chain(base, T(0, 0, size * 0.9), Rz(k * math.pi / 3 + rnd.random())))
    if hang > 0:
        w = size * 2.2
        cols = max(1, round(w / size))
        for c in range(cols):
            x = -w / 2 + w * (c + 0.5) / cols
            n = max(1, int(hang * (0.4 + 0.6 * rnd.random()) / size))
            for r in range(n):
                s = size * (1.25 - 0.35 * r / max(n, 1))
                leaf_card(L, s * 1.3, s * 1.5, tint(),
                          chain(base, T(x + (rnd.random() - 0.5) * size * 0.4, -size * 0.35 - 0.05 * r, -r * size * 0.95), Rz((rnd.random() - 0.5) * 0.5)))


def water_surface(g, poly, z, depth):
    """Water over a pool's bed: the polygon [(x, y)] at height z, `depth` over the bed (its uv.x: depth / 16 m)."""
    if not g.x: return
    tri = tessellate_polygon([[(x, y, 0) for x, y in poly]])
    g.x.water.add([(x, y, z) for x, y in poly], [tuple(t) for t in tri], WATER, uv=[(depth / 16.0, 0)] * len(poly))


def spout(g, p, out, z, lod, width=5.0, reach=360.0):
    """A gold spout on a rim at p (x, y), pointing out along `out` (unit x, y), its lip at height z: a waterfall pours from it."""
    a = math.atan2(out[1], out[0])
    m = chain(T(p[0], p[1], z), Rz(a - math.pi / 2))
    # a lion-mouthed channel: a gold trough jutting out, a boss over it, a basin-lip under it
    box(g, width * 1.2, 7.0, 1.2, GOLD, chain(m, T(0, -3.5, -1.2)))
    for s in (-1, 1):
        box(g, 0.8, 7.0, 2.4, GOLD, chain(m, T(s * width * 0.6, -3.5, -1.2)))
        pinnacle(g, 1.4, 6.0, chain(m, T(s * width * 0.6, -6.6, 1.2)), lod)
    sphere(g, width * 0.45, 10 if lod == 0 else 6, GOLD, chain(m, T(0, -1.0, 2.2), S(1.0, 0.8, 0.9)))
    lathe(g, [(0, 0), (width * 0.5, -1.0), (width * 0.3, -3.0), (0, -6.0)], 8 if lod == 0 else 5, GOLD, chain(m, T(0, -1.0, -1.4)))
    g.fall((p[0] + out[0] * 7.0, p[1] + out[1] * 7.0, z), out, width, reach)
