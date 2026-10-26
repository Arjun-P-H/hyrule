"""
The rotunda castle: a second layout of the same kit and style (castle.py the first).
-----------------------------------------------------------------------------
build(lod, seed) -> (exterior, interior, extras), as castle.build. Round terraces climb to a great lapis dome on a
two-tier drum over a round hall, crowned by a horizontal gold halo round its lantern. Four long galleries radiate
from it, each ending in a gabled front between two domed turrets (the front one a deep portal); a ring of eight
round towers with onion domes stands round them, the four between the galleries joined by arcades; gardens with
fountains fill the quarters. Inside: the round hall (a ring of columns and arcades, a great chandelier, a gold
armillary sphere) and the four galleries.
"""

import math
import random
import numpy as np
import castle as C
from castle import (wall_run, arcade, chapel, fountain, chandelier, lesser_tower, loggia, roof, gable, rose, portal,
                    buttress, seg_frame, island, steps, colossal, spouts, TB_Z, TC_Z)
from kit import (Geo, Extras, T, Rz, Rx, S, chain, box, beam, lathe, prism, cylinder, sphere, torus, arch_wall, finial,
                 pinnacle, balustrade, column, stairs, banner, dome, angel, cypress, MARBLE, CREAM, SHADE, GOLD,
                 HALO_GOLD, ARMILLARY, CARPET, FLOOR_DARK, FLOOR_LIGHT, FLOOR_ALT, TAU)

RP, RC = 212.0, 168.0      # the outer and inner round podiums' radii
RH = 62.0                  # the round hall's (the rotunda's wall)
HALL_H = 70.0
WING_END = 150.0           # the galleries reach this far from the middle
WING_W, WING_H = 44.0, 46.0
FRONT_W, FRONT_H = 84.0, 70.0   # (wide enough for the portal's gallery of arches)
TOWER_R = 195.0            # the ring of towers


def circle(r, n, z=0.0, phase=0.0):
    return [(r * math.cos(TAU * i / n + phase), r * math.sin(TAU * i / n + phase), z) for i in range(n)]


def round_podium(g, r, z0, z1, lod, tile, stairs_w, skip):
    """A round terrace from z0 up to z1: blind arcades round it, paving, a balustrade broken by the front stairs."""
    n = 48 if lod < 2 else 24
    pts = [(x, y) for x, y, _ in circle(r, n, 0, math.pi / n)]
    for i in range(n):
        a, b = pts[i], pts[(i + 1) % n]
        if (a[1] + b[1]) / 2 < 0 and max(abs(a[0]), abs(b[0])) < stairs_w / 2 + 6: continue  # (where the stairs arrive)
        wall_run(g, a, b, z0, z1 - z0, 2.4, 1e9, lod, kind='blind')
    prism(g, pts, z0, z1 - 0.05, CREAM)
    for ix in range(int(-r / tile) - 1, int(r / tile) + 1):
        for iy in range(int(-r / tile) - 1, int(r / tile) + 1):
            x0, y0 = ix * tile, iy * tile
            if max(math.hypot(x, y) for x, y in ((x0, y0), (x0 + tile, y0), (x0, y0 + tile), (x0 + tile, y0 + tile))) > r - 2: continue
            if lod == 0 and skip(x0, y0, tile): continue
            c = FLOOR_LIGHT if (ix + iy) % 2 else FLOOR_ALT
            g.add([(x0, y0, z1), (x0 + tile, y0, z1), (x0 + tile, y0 + tile, z1), (x0, y0 + tile, z1)], [(0, 1, 2, 3)], c)
    # the balustrade round the edge, open at the front for the stairs
    a0 = math.asin(min(1.0, (stairs_w / 2 + 2) / r))
    arc = [(r - 1.2) * np.array([math.cos(a), math.sin(a)]) for a in np.linspace(-math.pi / 2 + a0, 3 * math.pi / 2 - a0, n)]
    balustrade(g, [(x, y, z1) for x, y in arc], 1.5, None, lod)
    rise = z1 - z0
    stairs(g, stairs_w, rise, rise * 2.6, steps(rise, lod), MARBLE, T(0, -r - rise * 2.6 + 1, z0))
    for s in (-1, 1):
        box(g, 6, rise * 2.6, rise + 2.5, CREAM, T(s * (stairs_w / 2 + 3), -r - rise * 1.3 + 1, z0))
        angel(g, 13.0, T(s * (stairs_w / 2 + 3), -r - rise * 2.6 + 3.5, z1 + 2.5), lod)


def great_dome(g, gi, lod):
    """The rotunda: a ring of lancet bays and buttresses, two drums of windows, the lapis dome, the lantern, the crown."""
    z = TC_Z
    n = 16
    ring = [(RH * math.cos(TAU * i / n + math.pi / n), RH * math.sin(TAU * i / n + math.pi / n)) for i in range(n)]
    for i in range(n):
        a, b = ring[i], ring[(i + 1) % n]
        mid = math.atan2(a[1] + b[1], a[0] + b[0])
        to_wing = min(abs(math.remainder(mid - k * math.pi / 2, TAU)) for k in range(4)) < 0.1
        wall_run(g, a, b, z, HALL_H, 3.2, 1e9, lod, kind='window', door=0 if to_wing else None)
        if not to_wing:
            buttress(g, a[0], a[1], z, HALL_H, 10.0, lod, a=math.atan2(a[1], a[0]) + math.pi / 2)
    lathe(g, [(RH + 2, 0), (0.01, 0)], n, CREAM, T(0, 0, z + HALL_H), phase=math.pi / n)
    # the drums: a wide one of tall lancets ringed by pinnacles, a narrower one of rose-like oculi (lancets)
    zz = z + HALL_H
    for k, (r, h) in enumerate(((56.0, 40.0), (50.0, 28.0))):
        pts = [(r * math.cos(TAU * i / n + math.pi / n), r * math.sin(TAU * i / n + math.pi / n)) for i in range(n)]
        for i in range(n):
            wall_run(g, pts[i], pts[(i + 1) % n], zz, h, 2.6, 1e9, lod, kind='window')
        for px, py in pts[::2] if k == 0 else []:
            pinnacle(g, 3.0, 20, T(px * 1.05, py * 1.05, zz + h), lod)
        if lod <= 1:
            balustrade(g, [(px * 1.03, py * 1.03, zz + h) for px, py in pts], 1.4, None, lod, closed=True)
        lathe(g, [(r + 1.5, 0), (0.01, 0)], n, CREAM, T(0, 0, zz + h), phase=math.pi / n)
        zz += h
    # the dome (a tall bell, gold-ribbed), the lantern of open arches, its onion and needle
    dome(g, 49.0, 104.0, T(0, 0, zz), lod, 'bell')
    zl = zz + 94.0
    pts = [(12 * math.cos(TAU * i / 8), 12 * math.sin(TAU * i / 8)) for i in range(8)]
    for i in range(8):
        wall_run(g, pts[i], pts[(i + 1) % 8], zl, 24, 1.6, 1e9, lod, kind='open')
    lathe(g, [(13.5, 0), (0.01, 0)], 8, CREAM, T(0, 0, zl + 24))
    dome(g, 11.5, 30, T(0, 0, zl + 24), lod, 'onion')
    crown(g, zl + 16, lod)
    round_hall(gi, lod)


def crown(g, z, lod):
    """The halo crown: a horizontal gold ring round the lantern, star points out from it, spikes up, a thinner ring tilted through it."""
    glow = HALO_GOLD
    R = 96.0
    n = 96 if lod == 0 else 40 if lod == 1 else 24
    torus(g, R, 3.0, n, 8 if lod == 0 else 5, glow, T(0, 0, z))
    torus(g, R * 0.86, 1.4, n, 6 if lod == 0 else 4, glow, T(0, 0, z + 3))
    for k in range(16):
        a = TAU * k / 16
        L = 48 if k % 4 == 0 else 24 if k % 2 == 0 else 12
        w = 8.0 if k % 4 == 0 else 4.5
        diamond = [(R - 4, 0), (R + 6, w), (R + L, 0), (R + 6, -w)]
        prism(g, diamond, -1.2, 1.2, glow, chain(T(0, 0, z), Rz(a)))
        if k % 2 == 0:  # a spike up from the ring
            lathe(g, [(2.2, 0), (0.6, 14 if k % 4 == 0 else 8), (0, 18 if k % 4 == 0 else 11)], 6, glow, T(R * math.cos(a), R * math.sin(a), z + 2))
    if lod <= 1:
        torus(g, R * 1.15, 1.0, n, 4, glow, chain(T(0, 0, z + 10), Rx(0.42)))
        for k in range(24):  # spokes to the lantern
            a = TAU * (k + 0.5) / 24
            beam(g, (16 * math.cos(a), 16 * math.sin(a), z - 2), (R * 0.86 * math.cos(a), R * 0.86 * math.sin(a), z + 2), 0.8, 0.8, glow, None)


def wing(g, gi, a, lod, front=False):
    """A gallery from the rotunda out along angle a: lancet walls and buttresses, a roof, a gabled end between two domed turrets."""
    z = TC_Z
    w, h = (FRONT_W, FRONT_H) if front else (WING_W, WING_H)
    ca, sa = math.cos(a), math.sin(a)
    P = lambda along, across: (along * ca - across * sa, along * sa + across * ca)
    r0 = RH - 6
    for s in (-1, 1):
        p, q = P(r0, s * w / 2), P(WING_END, s * w / 2)
        if s > 0: p, q = q, p  # (the outside on the right)
        wall_run(g, p, q, z, h, 3.0, 12.0, lod, kind='window')
        nb = max(1, round((WING_END - RH) / 12))
        for i in range(1, nb + 1):
            bx, by = P(RH + (WING_END - RH) * i / nb, s * w / 2)
            buttress(g, bx, by, z, h, 8.0, lod, a=a + (math.pi if s > 0 else 0.0))
    roof(g, P(r0, 0), P(WING_END, 0), w / 2 + 1.5, z + h, 30 if front else 22, lod)
    # the end wall, from e2 to e1 (so it faces out along a)
    e1, e2 = P(WING_END, w / 2), P(WING_END, -w / 2)
    if front:
        # the front (always toward -y): the portal's screen, its rose, its gable; its towers are the turrets
        portal(g, T(*P(WING_END, 0), z), w + 4, lod)
        rose(g, *P(WING_END + 2.2, 0), z + 92, 17, lod, axis='y')
        gable(g, e2, e1, z + 110, 40, lod)
    else:
        wall_run(g, e2, e1, z, h, 3.0, w / 3, lod, kind='window', door=1)
        gable(g, e2, e1, z + h, 26, lod)
    for s in (-1, 1):
        tx, ty = P(WING_END + 4, s * (w / 2 + 6))
        chapel(g, tx, ty, lod, r=10.0, h=(h + 34) if front else (h + 18), z=z)
    gallery(gi, P(RH + 2, 0), P(WING_END - 2, 0), w - 7, z, h, lod)


def gallery(gi, a, b, w, z, h, lod):
    """A long hall from a to b, w wide: a checkered floor, two colonnades, chandeliers, banners, gold transverse ribs."""
    if lod > 0: return
    M, L = seg_frame(a, b, z + 0.06)
    t = 3.0
    for ix in range(int(-L / 2 / t), int(L / 2 / t)):
        for iy in range(int(-w / 2 / t), int(w / 2 / t)):
            c = FLOOR_DARK if (ix + iy) % 2 else FLOOR_LIGHT
            x0, y0 = ix * t, iy * t
            gi.add([(x0, y0, 0), (x0 + t, y0, 0), (x0 + t, y0 + t, 0), (x0, y0 + t, 0)], [(0, 1, 2, 3)], c, M)
    for x in np.arange(-L / 2 + 8, L / 2 - 4, 12):
        for s in (-1, 1):
            column(gi, 1.3, h * 0.62, chain(M, T(x, s * w * 0.3, 0)), lod)
            banner(gi, 3.4, 13, chain(M, T(x + 6, s * (w / 2 - 0.5), h * 0.7), Rz(0 if s < 0 else math.pi)), lod)
        beam(gi, (x, -w / 2, h - 2), (x, 0, h + 12), 1.0, 1.0, GOLD, M)
        beam(gi, (x, w / 2, h - 2), (x, 0, h + 12), 1.0, 1.0, GOLD, M)
    for x in np.arange(-L / 2 + 14, L / 2 - 8, 24):
        mx = M @ np.array([x, 0, 0, 1.0])
        chandelier(gi, mx[0], mx[1], z + h * 0.55, z + h + 10, lod)
    box(gi, L - 6, 7, 0.12, CARPET, M)


def round_hall(gi, lod):
    """The hall under the dome: a polar checkered floor, a ring of columns and arcades, a ceiling of gold ribs round an
    oculus, a great chandelier, and a gold armillary sphere on a stepped plinth at its heart."""
    if lod > 0: return
    z = TC_Z + 0.06
    rings, sectors = 12, 48
    for i in range(rings):
        r0, r1 = (RH - 2) * i / rings, (RH - 2) * (i + 1) / rings
        for k in range(sectors):
            a0, a1 = TAU * k / sectors, TAU * (k + 1) / sectors
            c = FLOOR_DARK if (i + k) % 2 else FLOOR_LIGHT
            gi.add([(r0 * math.cos(a0), r0 * math.sin(a0), z), (r1 * math.cos(a0), r1 * math.sin(a0), z),
                    (r1 * math.cos(a1), r1 * math.sin(a1), z), (r0 * math.cos(a1), r0 * math.sin(a1), z)], [(0, 1, 2, 3)], c)
    n = 16
    rc = RH - 15
    cols = [(rc * math.cos(TAU * i / n), rc * math.sin(TAU * i / n)) for i in range(n)]
    for (x, y), (x2, y2) in zip(cols, cols[1:] + cols[:1]):
        column(gi, 1.6, 40, T(x, y, z), lod)
        m, L = seg_frame((x, y), (x2, y2), z + 40)
        arch_wall(gi, L, 14, 2.2, L - 2.6, 0, 2, (L - 2.6) * 0.55, MARBLE, m, n=6, reveal=GOLD)
        banner(gi, 3.0, 14, chain(T(x * 0.93, y * 0.93, z + 38), Rz(math.atan2(y, x) + math.pi / 2)), lod)
    # the ceiling over the hall (beneath the drums): a ring with a gold oculus, gold ribs to it
    zc = TC_Z + HALL_H - 1.0
    ring = [((RH + 1) * math.cos(TAU * i / 32), (RH + 1) * math.sin(TAU * i / 32)) for i in range(32)]
    hole = [(14 * math.cos(TAU * i / 32), 14 * math.sin(TAU * i / 32)) for i in range(32)]
    v = [(x, y, zc) for x, y in ring] + [(x, y, zc) for x, y in hole]
    f = [(i, 32 + i, 32 + (i + 1) % 32, (i + 1) % 32) for i in range(32)]
    gi.add(v, f, SHADE)
    torus(gi, 14, 1.0, 32, 4, GOLD, T(0, 0, zc - 0.6))
    for k in range(16):
        a = TAU * k / 16
        beam(gi, ((RH - 2) * math.cos(a), (RH - 2) * math.sin(a), zc - 1.4), (14 * math.cos(a), 14 * math.sin(a), zc - 1.4), 1.2, 1.2, GOLD, None)
    chandelier(gi, 0, 0, TC_Z + 36, zc, lod, big=True)
    # the armillary sphere
    for k in range(4):
        lathe(gi, [(16 - k * 3, 0), (16 - k * 3, 0.9), (0.01, 0.9)], 32, FLOOR_LIGHT if k % 2 else MARBLE, T(0, 0, z + k * 0.9))
    zb = z + 3.6
    lathe(gi, [(2.6, 0), (1.2, 2), (1.0, 6), (2.2, 7.5), (0.01, 7.5)], 12, GOLD, T(0, 0, zb))
    c = T(0, 0, zb + 15)
    for tilt, spin in ((0, 0), (math.pi / 2, 0), (math.pi / 2, math.pi / 2), (0.41, 0.3)):
        torus(gi, 7.0, 0.35, 40, 4, GOLD, chain(c, Rz(spin), Rx(tilt)))
    sphere(gi, 1.8, 12, ARMILLARY, c)
    beam(gi, (0, 0, zb + 6.5), (0, 0, zb + 24), 0.4, 0.4, GOLD, None)


def quarter_garden(g, rnd, a, lod):
    """A garden in the quarter between two galleries: a fountain ringed by bushes, cypress walks, a small domed chapel."""
    r = 112.0
    x, y = r * math.cos(a), r * math.sin(a)
    fountain(g, x, y, lod, z=TC_Z, r=12.0)
    if lod <= 1:
        for k in range(8):
            b = TAU * k / 8 + a
            cypress(g, 12 + rnd.random() * 3, T(x + 26 * math.cos(b), y + 26 * math.sin(b), TC_Z), lod)
    chapel(g, (r + 30) * math.cos(a), (r + 30) * math.sin(a), lod, r=9.0, h=26.0, z=TC_Z)


def build(lod, seed=11):
    rnd = random.Random(seed)
    C.RND = random.Random(seed + 1)
    extras = Extras() if lod == 0 else None
    g, gi = Geo(lod, extras), Geo(lod, extras)
    island(g, rnd, lod, skip=lambda x0, y0, t: max(math.hypot(x, y) for x, y in ((x0, y0), (x0 + t, y0), (x0, y0 + t), (x0 + t, y0 + t))) < RP)

    def under_hall(x0, y0, t):
        """A tile the rotunda's or a gallery's floor covers."""
        cs = ((x0, y0), (x0 + t, y0), (x0, y0 + t), (x0 + t, y0 + t))
        if all(math.hypot(x, y) < RH - 1 for x, y in cs): return True
        widths = (WING_W, WING_W, WING_W, FRONT_W)  # (the galleries along +x, +y, -x, -y)
        return any(all(abs(y if k % 2 == 0 else x) < widths[k] / 2 - 3 and RH < (x if k % 2 == 0 else y) * (1 if k < 2 else -1) < WING_END - 2
                       for x, y in cs) for k in range(4))
    round_podium(g, RP, 0, TB_Z, lod, 8.0 if lod == 0 else 24.0, 64, lambda x0, y0, t: max(math.hypot(x, y) for x, y in ((x0, y0), (x0 + t, y0), (x0, y0 + t), (x0 + t, y0 + t))) < RC)
    round_podium(g, RC, TB_Z, TC_Z, lod, 6.0 if lod == 0 else 18.0, 48, under_hall)
    great_dome(g, gi, lod)
    for k in range(4):
        wing(g, gi, k * math.pi / 2, lod, front=(k == 3))
        quarter_garden(g, rnd, k * math.pi / 2 + math.pi / 4, lod)
    # the ring of towers between the galleries, onion-domed; arcades join the pairs either side of each diagonal
    towers = [(TOWER_R * math.cos(TAU * k / 8 + math.pi / 8), TOWER_R * math.sin(TAU * k / 8 + math.pi / 8)) for k in range(8)]
    for k, (x, y) in enumerate(towers):
        lesser_tower(g, x, y, lod, rnd, 120 if k % 2 == 0 else 96, r=15.0 if k % 2 == 0 else 12.0, kind='onion', z=TB_Z)
    for k in range(0, 8, 2):
        (x0, y0), (x1, y1) = towers[k], towers[k + 1]
        L = math.hypot(x1 - x0, y1 - y0)
        ux, uy = (x1 - x0) / L, (y1 - y0) / L
        arcade(g, (x0 + ux * 18, y0 + uy * 18), (x1 - ux * 18, y1 - uy * 18), TB_Z, lod, levels=2, h=26.0)
    loggia(g, rnd, lod)
    # lesser towers on the loggia, clear of the gate
    for k in range(8):
        a = TAU * k / 8 + 0.2
        if abs(math.sin(a) + 1) < 0.3: continue
        pr = C.ISLAND_R * (1 + 0.05 * math.sin(3 * a + 1.0) + 0.035 * math.sin(5 * a + 2.0) + 0.02 * math.sin(9 * a)) * 0.9
        lesser_tower(g, pr * math.cos(a), pr * 0.93 * math.sin(a), lod, rnd, 50 + rnd.random() * 30)
    spouts(g, rnd, lod)
    return colossal(g, gi, extras)
