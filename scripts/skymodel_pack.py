"""
Packing a sky model for the game (src/lib/skymodels.ts parseSkyModel): shared by bake-sky-model.py (a .glb) and
sky-castle/build.py (the procedural castle). Runs inside Blender (bpy, mathutils).
-----------------------------------------------------------------------------
Game axes: y up (Blender's z), z = Blender's -y. The file, gzipped: 'SKYM', u32 header length, the JSON header,
then every section at a 4-byte boundary, in the header's order:
  - each part's levels of detail, then each prototype's: 20-byte vertices (position unorm16x3 over the mesh's own
    box + a u16: material << 8 | sky visibility (u8); normal snorm8x3 + ambient occlusion snorm8; uv unorm16x2;
    colour unorm8x3 sRGB + glow unorm8) and u32 triangle indices;
  - the props (instances of the prototypes, sorted into PROP_CHUNK squares): 32 bytes each, f32 x, y, z, yaw, sx, sy, sz (game
    axes, m), u8 sky visibility, u8 occlusion, u16 0;
  - the floors: on a `cell` grid, every level surface under each cell, highest first: u32 offsets (a cell's layers
    run from its offset to the next cell's; nx · nz + 1 of them), then the layers, each a u16: its height (1 + 15
    bits over the floors' height range) << 1 | blocked (something stands within a body above it there: a wall, a
    column, a railing).
"""

import gzip, json, math, struct
import numpy as np
import bpy
from mathutils.bvhtree import BVHTree

FLOOR_LAYERS = 64   # (at most, a cell: under a dome its shell, drums and ceilings stack a dozen over the hall's floor)
PROP_CHUNK = 128.0


def to_game(a):
    return np.stack([a[:, 0], a[:, 2], -a[:, 1]], 1)


def mesh_corners(obj, color='col', ao='ao', vis='vis', uv_xf=None):
    """Per triangle corner, in game axes: position, normal, uv (or zeros), colour + glow, occlusion, sky visibility, material."""
    m = obj.data
    m.calc_loop_triangles()
    nt = len(m.loop_triangles)
    loops = np.empty(nt * 3, np.int32); m.loop_triangles.foreach_get('loops', loops)
    polys = np.empty(nt, np.int32); m.loop_triangles.foreach_get('polygon_index', polys)
    lv = np.empty(len(m.loops), np.int32); m.loops.foreach_get('vertex_index', lv)
    co = np.empty(len(m.vertices) * 3, np.float32); m.vertices.foreach_get('co', co); co = co.reshape(-1, 3)
    nrm = np.empty(len(m.loops) * 3, np.float32); m.corner_normals.foreach_get('vector', nrm); nrm = nrm.reshape(-1, 3)
    P, N = to_game(co[lv[loops]]), to_game(nrm[loops])
    if uv_xf is not None and m.uv_layers:
        uv = np.empty(len(m.loops) * 2, np.float32); m.uv_layers[0].data.foreach_get('uv', uv); uv = uv.reshape(-1, 2)
        uv = uv * uv_xf[0] + uv_xf[1]
        uv[:, 1] = 1.0 - uv[:, 1]
        UV = uv[loops]
    else:
        UV = np.zeros((len(loops), 2), np.float32)
    if color in m.attributes:
        a = m.attributes[color]
        c = np.empty(len(a.data) * 4, np.float32); a.data.foreach_get('color', c); c = c.reshape(-1, 4)
        C = c[polys] if a.domain == 'FACE' else c[loops]
        C = np.repeat(C, 3, axis=0) if a.domain == 'FACE' else C
    else:
        C = np.tile([1.0, 1.0, 1.0, 0.0], (len(loops), 1))

    def corner_scalar(name):
        if name not in m.attributes: return np.ones(len(loops), np.float32)
        a = m.attributes[name]
        c = np.empty(len(a.data) * 4, np.float32); a.data.foreach_get('color', c); c = c.reshape(-1, 4)[:, 0]
        return c[loops] if a.domain == 'CORNER' else c[lv[loops]]
    MAT = np.zeros(len(loops), np.int32)
    if 'mat' in m.attributes:
        mf = np.empty(len(m.polygons), np.int32); m.attributes['mat'].data.foreach_get('value', mf)
        MAT = np.repeat(mf[polys], 3)
    return P, N, UV, C, corner_scalar(ao), corner_scalar(vis), MAT


def encode(P, N, UV, C, AO, VIS, MAT=None):
    """One mesh's vertex and index bytes (shared corners merged), its box and counts."""
    lo, hi = P.min(0), P.max(0)
    ext = np.maximum(hi - lo, 1e-6)
    rows = np.zeros((len(P), 10), np.uint16)
    rows[:, 0:3] = np.round(np.clip((P - lo) / ext, 0, 1) * 65535)
    mat = np.zeros(len(P), np.int64) if MAT is None else np.asarray(MAT, np.int64)
    rows[:, 3] = (mat << 8) | np.round(np.clip(VIS, 0, 1) * 255).astype(np.int64)
    nn = N / np.maximum(np.linalg.norm(N, axis=1, keepdims=True), 1e-8)
    b4 = np.empty((len(P), 4), np.int8)
    b4[:, 0:3] = np.round(np.clip(nn, -1, 1) * 127)
    b4[:, 3] = np.round(np.clip(AO, 0, 1) * 127)
    rows[:, 4:6] = np.frombuffer(b4.tobytes(), np.uint16).reshape(-1, 2)
    rows[:, 6:8] = np.round(np.clip(UV, 0, 1) * 65535)
    c4 = np.round(np.clip(C, 0, 1) * 255).astype(np.uint8)

    rows[:, 8:10] = np.frombuffer(c4.tobytes(), np.uint16).reshape(-1, 2)
    verts, index = np.unique(rows, axis=0, return_inverse=True)
    return verts.tobytes(), index.astype(np.uint32).ravel().tobytes(), {
        'verts': len(verts), 'tris': len(P) // 3, 'lo': lo.tolist(), 'hi': hi.tolist()}


BODY = (1.1, 1.6)    # m over a floor: where a body is tested for walls (clear of a stair's next tread, 0.45 m up)
BODY_REACH = 0.55    # m: a surface this near the cell's middle there blocks it


def floors(tri_P, up=0.4, cell=1.0, layers=FLOOR_LAYERS):
    """The floor layers under a set of triangles (game axes, corners in threes): its header entry and its bytes."""
    lo, hi = tri_P.min(0), tri_P.max(0)
    bvh = BVHTree.FromPolygons(tri_P.tolist(), np.arange(len(tri_P)).reshape(-1, 3).tolist())
    nx, nz = int(math.ceil((hi[0] - lo[0]) / cell)), int(math.ceil((hi[2] - lo[2]) / cell))
    cells = []
    top = float(hi[1]) + 1
    for iz in range(nz):
        z = lo[2] + (iz + 0.5) * cell
        for ix in range(nx):
            x = lo[0] + (ix + 0.5) * cell
            # every level surface under the cell (either winding: the kit's and the scans' faces are not all outward)
            ys, y = [], top
            while len(ys) < layers:
                hit, n, _, _ = bvh.ray_cast((x, y, z), (0, -1, 0))
                if hit is None: break
                if abs(n[1]) > up and (not ys or ys[-1] - hit[1] > 0.3): ys.append(hit[1])
                y = hit[1] - 0.05
            cells.append([(yy, any(bvh.find_nearest((x, yy + h, z), BODY_REACH)[0] is not None for h in BODY)) for yy in ys])
    all_y = [yy for c in cells for yy, _ in c]
    fmin, fmax = float(min(all_y)), float(max(all_y))
    off = np.zeros(len(cells) + 1, np.uint32)
    off[1:] = np.cumsum([len(c) for c in cells])
    q = np.array([int(round(1 + (yy - fmin) / max(fmax - fmin, 1e-6) * 32766)) * 2 + b for c in cells for yy, b in c], np.uint16)
    return {'nx': nx, 'nz': nz, 'cell': cell, 'x0': float(lo[0]), 'z0': float(lo[2]), 'count': int(off[-1]), 'min': fmin, 'max': fmax}, off.tobytes() + q.tobytes()


def props_bytes(props, kinds):
    """Props [(kind, game-axes position, yaw, (sx, sy, sz), vis, ao)] sorted into chunks: their bytes and chunk table."""
    keyed = sorted(props, key=lambda p: (kinds.index(p[0]), math.floor(p[1][0] / PROP_CHUNK), math.floor(p[1][2] / PROP_CHUNK)))
    out = bytearray()
    chunks = []
    for i, (kind, pos, yaw, sc, vis, ao) in enumerate(keyed):
        key = (kinds.index(kind), math.floor(pos[0] / PROP_CHUNK), math.floor(pos[2] / PROP_CHUNK))
        if not chunks or chunks[-1]['key'] != key:
            chunks.append({'key': key, 'proto': key[0], 'first': i, 'count': 0, 'pts': []})
        chunks[-1]['count'] += 1
        chunks[-1]['pts'].append(pos)
        out += struct.pack('<7f2BH', *pos, yaw, *sc, int(round(vis * 255)), int(round(ao * 255)), 0)
    table = []
    for c in chunks:
        p = np.array(c['pts']); m = (p.min(0) + p.max(0)) / 2
        table.append({'proto': c['proto'], 'first': c['first'], 'count': c['count'], 'c': m.tolist(), 'r': float(np.linalg.norm(p - m, axis=1).max()) + 20})
    return bytes(out), table


def write(path, header, sections):
    """The file: gzipped header and 4-byte-aligned sections."""
    hj = json.dumps(header).encode()
    hj += b' ' * (-(8 + len(hj)) % 4)
    body = bytearray(b'SKYM' + struct.pack('<I', len(hj)) + hj)
    for s in sections:
        body += s + b'\0' * (-len(s) % 4)
    with open(path, 'wb') as f:
        f.write(gzip.compress(bytes(body), 9))
    return len(body)
