"""
Build a sky castle for the game: a layout of the kit (castle.py the cathedral, rotunda.py the rotunda) at three levels of detail, its interior, its props,
the light baked in, packed (skymodel_pack.py) into public/models/<name>.bin.
-----------------------------------------------------------------------------
    /Applications/Blender.app/Contents/MacOS/Blender -b --python scripts/sky-castle/build.py -- [name] [seed] [cathedral|rotunda]

Cycles bakes two shades into every corner: ambient occlusion (AO_DISTANCE: the creases, the shade of a colonnade)
and sky visibility (how much of the sky it sees: the interiors go dark, lit by their glowing glass and candles).
The props (instanced pieces) take both from rays cast from their middles.
"""

import bpy, sys, os, math, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.dirname(HERE)]
import kit, castle, rotunda
import skymodel_pack as pack
from mathutils.bvhtree import BVHTree

AO_DISTANCE = 12.0 * castle.SCALE
SKY_DISTANCE = 4000.0 * castle.SCALE
BAKE_SAMPLES = 48
SMOOTH_ANGLE = math.radians(40)
PROTO_KINDS = ['pinnacle', 'baluster']
# a level of detail of each part is drawn within its distance (m); the interior only near the castle
PART_NEAR = [None, 700.0 * castle.SCALE]
# the ivy's leaf cards and the pools' water are drawn within this distance (m)
NEAR_DETAIL = 1200.0 * castle.SCALE

args = sys.argv[sys.argv.index('--') + 1:] if '--' in sys.argv else []
name = args[0] if args else 'sky-castle'
seed = int(args[1]) if len(args) > 1 else 7
layout = {'cathedral': castle, 'rotunda': rotunda}[args[2] if len(args) > 2 else 'cathedral']
out_dir = os.path.join(os.path.dirname(os.path.dirname(HERE)), 'public', 'models')
t0 = time.time()
log = lambda *a: print(f'[{time.time() - t0:6.1f}s]', *a, flush=True)

bpy.ops.wm.read_factory_settings(use_empty=True)
sc = bpy.context.scene
mat = bpy.data.materials.new('m'); mat.use_nodes = True


def make(nm, geo):
    """A Blender object of a Geo: its faces' colours and materials as face attributes, its uvs, normals smoothed by angle."""
    v, f, c = geo.arrays()
    me = bpy.data.meshes.new(nm); me.from_pydata(v.tolist(), [], f)
    c = np.asarray(c, np.float32).reshape(-1, 5)
    me.attributes.new('col', 'FLOAT_COLOR', 'FACE').data.foreach_set('color', c[:, :4].ravel())
    me.attributes.new('mat', 'INT', 'FACE').data.foreach_set('value', c[:, 4].astype(np.int32))
    uv = geo.uvs()
    if np.any(uv):
        lv = np.empty(len(me.loops), np.int32); me.loops.foreach_get('vertex_index', lv)
        me.uv_layers.new(name='uv').data.foreach_set('uv', uv[lv].astype(np.float32).ravel())
    me.validate()
    me.materials.append(mat)
    o = bpy.data.objects.new(nm, me); sc.collection.objects.link(o)
    me.shade_smooth()
    for x in bpy.context.selected_objects: x.select_set(False)
    o.select_set(True); bpy.context.view_layer.objects.active = o
    bpy.ops.object.shade_smooth_by_angle(angle=SMOOTH_ANGLE)
    return o


def bake(objs, others=()):
    """Bake occlusion and sky visibility into each of objs (seen with the others and each other)."""
    for o in bpy.data.objects: o.hide_render = o not in objs and o not in others
    sc.render.engine = 'CYCLES'
    prefs = bpy.context.preferences.addons['cycles'].preferences
    try:
        prefs.compute_device_type = 'METAL'; prefs.get_devices()
        for d in prefs.devices: d.use = True
        sc.cycles.device = 'GPU'
    except Exception as e:
        log('bake on the CPU:', e)
    sc.cycles.samples = BAKE_SAMPLES
    sc.render.bake.target = 'VERTEX_COLORS'
    if sc.world is None: sc.world = bpy.data.worlds.new('w')
    for attr, dist in (('ao', AO_DISTANCE), ('vis', SKY_DISTANCE)):
        sc.world.light_settings.distance = dist
        for o in objs:
            me = o.data
            if attr not in me.color_attributes: me.color_attributes.new(attr, 'BYTE_COLOR', 'CORNER')
            me.color_attributes.active_color = me.color_attributes[attr]
            for x in bpy.context.selected_objects: x.select_set(False)
            o.select_set(True); bpy.context.view_layer.objects.active = o
            bpy.ops.object.bake(type='AO', target='VERTEX_COLORS')
            log('baked', attr, o.name)


# the castle at each level, the interior and the extras (recorded at the finest)
levels = []
for lod in range(3):
    g, gi, x = layout.build(lod, seed)
    levels.append((g, gi, x))
    log('lod', lod, 'tris', g.tris(), 'interior', gi.tris())
X = levels[0][2]
log('props', len(X.props), 'plants', len(X.plants), 'falls', len(X.falls), 'ivy cards', len(X.leaves.f), 'water tris', X.water.tris())
ext = [make(f'ext{l}', levels[l][0]) for l in range(3)]
inner = make('int0', levels[0][1])
leaves = make('leaves', X.leaves)
water = make('water', X.water)
protos = {k: [make(f'{k}{l}', kit.protos(l)[k]) for l in range(3 if k == 'pinnacle' else 1)] for k in PROTO_KINDS}
for k in protos:
    for o in protos[k]: o.hide_render = True

bake([ext[0], inner])
bake([leaves], others=[ext[0], inner])  # (the ivy shaded by the walls, without shading them)
bake([ext[1]]); bake([ext[2]])
for k in protos: bake(protos[k])   # (alone: their own creases)

# the props' shade: rays from each one's middle against the finest castle
P0 = np.concatenate([pack.mesh_corners(ext[0])[0], pack.mesh_corners(inner)[0]])
bvh = BVHTree.FromPolygons(P0.tolist(), np.arange(len(P0)).reshape(-1, 3).tolist())
dirs = []
for i in range(24):  # a cosine-weighted hemisphere, the same for every prop
    u, v = (i + 0.5) / 24, (i * 0.618034) % 1
    r = math.sqrt(u); a = math.tau * v
    dirs.append((r * math.cos(a), math.sqrt(1 - u), r * math.sin(a)))
packed_props = []
for kind, M in X.props:
    M = np.asarray(M)
    sx, sy, sz = (float(np.linalg.norm(M[:3, i])) for i in range(3))
    yaw = math.atan2(M[1, 0], M[0, 0])
    b = M[:3, 3]
    pos = (float(b[0]), float(b[2]), float(-b[1]))
    mid = (pos[0], pos[1] + 0.5 * sz * (kit.PINNACLE_H if kind == 'pinnacle' else 1.0), pos[2])
    vis = ao = 0
    for d in dirs:
        hit = bvh.ray_cast((mid[0] + d[0] * 0.6, mid[1] + d[1] * 0.6, mid[2] + d[2] * 0.6), d, SKY_DISTANCE)
        if hit[0] is None: vis += 1; ao += 1
        elif hit[3] > AO_DISTANCE: ao += 1
    packed_props.append((kind, pos, yaw, (sx, sz, sy), vis / len(dirs), ao / len(dirs)))
log('props shaded', len(packed_props))

# pack: the exterior (three levels), the interior, the ivy, the water; the prototypes; the props; the floors
sections, parts = [], []
UV = ((1.0, 1.0), (0.0, 0.0))
for objs, kind, near, uv in (([*ext], 'solid', PART_NEAR[0], None), ([inner], 'solid', PART_NEAR[1], None),
                             ([leaves], 'leaf', NEAR_DETAIL, UV), ([water], 'water', NEAR_DETAIL, UV)):
    lods = []
    for o in objs:
        vb, ib, info = pack.encode(*pack.mesh_corners(o, uv_xf=uv))
        sections += [vb, ib]; lods.append(info)
    parts.append({'kind': kind, 'near': near, 'lods': lods})
proto_hdr = []
for k in PROTO_KINDS:
    lods = []
    for o in protos[k]:
        vb, ib, info = pack.encode(*pack.mesh_corners(o))
        sections += [vb, ib]; lods.append(info)
    proto_hdr.append({'name': k, 'lods': lods})
pb, chunks = pack.props_bytes(packed_props, PROTO_KINDS)
sections.append(pb)
fh, fb = pack.floors(P0, cell=1.25)
sections.append(fb)
lo = np.min([l['lo'] for p in parts for l in p['lods']], 0); hi = np.max([l['hi'] for p in parts for l in p['lods']], 0)
# the plants and the falls, in game axes (x, z up -> y, -y -> z)
g3 = lambda p: [float(p[0]), float(p[2]), float(-p[1])]
plants = [{'species': s, 'p': g3(p), 'scale': float(sc), 'palette': pal} for s, p, sc, pal in X.plants]
falls = [{'lip': g3(p), 'dir': [float(d[0]), float(-d[1])], 'width': float(w), 'reach': float(r)} for p, d, w, r in X.falls]
header = {'version': 2, 'boxMin': lo.tolist(), 'boxMax': hi.tolist(), 'texture': False, 'parts': parts,
          'protos': proto_hdr, 'props': {'count': len(packed_props), 'chunks': chunks}, 'floor': fh,
          'plants': plants, 'falls': falls}
raw = pack.write(os.path.join(out_dir, name + '.bin'), header, sections)
log('wrote', name + '.bin', f'{raw / 1e6:.1f} MB raw,', f'{os.path.getsize(os.path.join(out_dir, name + ".bin")) / 1e6:.1f} MB gzipped')
log('parts', [[(l['verts'], l['tris']) for l in p['lods']] for p in parts], 'protos', [[(l['verts'], l['tris']) for l in p['lods']] for p in proto_hdr], 'chunks', len(chunks))
