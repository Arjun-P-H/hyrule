"""
Build the colossus for the game: Herod (a stone golem, Herod.glb: one unrigged 1.5 M-triangle mesh) decimated, scaled
to COLOSSUS height, given a skeleton and weights, packed into public/models/colossus.bin with its base colour as
public/models/colossus.jpg.
-----------------------------------------------------------------------------
    /Applications/Blender.app/Contents/MacOS/Blender -b --python scripts/colossus/build.py -- [Herod.glb] [--preview out.png]

The model has no rig and no clips: JOINTS places a skeleton on it by hand (read off front and side views of the
mesh in its own units, Blender space: Z up, facing −Y, its left +X), and the game moves it procedurally
(colossus/colossi.ts: walking by IK on the ground, the smash, the fall). Blender's bone heat weights the decimated
mesh to it (a vertex heat missed takes its nearest bone).

colossus.bin is horse.bin's format (horse/model.ts parseHorse, 'HRS1'), with no clips: the joints' names, parents
and rest places are what the game poses from; then a collision mesh (`coll`), skinned by the game on the CPU.
Everything is in the game's frame: metres, +Y up, facing +Z, its
right +X, the origin on the ground between its feet.
"""

import bpy, bmesh, sys, os, json, struct, math
import numpy as np
from mathutils import Vector

args = sys.argv[sys.argv.index('--') + 1:] if '--' in sys.argv else []
preview = args[args.index('--preview') + 1] if '--preview' in args else None
args = [a for i, a in enumerate(args) if a != '--preview' and (i == 0 or args[i - 1] != '--preview')]
SRC = args[0] if args else os.path.expanduser('~/Desktop/3D Characters/Mini Bosses/Herod.glb')
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'public', 'models')

HEIGHT = 30.0         # metres to the top of its tower (Argus-sized)
TRIS = 60000          # triangles after decimation
TEX = 2048            # the base colour's size (px)
COLL_TRIS = 10000     # triangles of the collision mesh (climbing, the hero's body, blows)
VERTEX_BYTES = 44

# The skeleton in the mesh's own units (0.957 tall). Left side (+X); mirrored to the right. Each joint's parent.
MID = {'hips': (0, 0.03, 0.42), 'chest': (0, 0.02, 0.60), 'head': (0, -0.08, 0.60), 'crown': (0, -0.14, 0.86)}
SIDE = {
    'clavicle': (0.07, 0.0, 0.68), 'shoulder': (0.27, 0.0, 0.56), 'elbow': (0.28, 0.0, 0.42), 'wrist': (0.30, -0.01, 0.22), 'hand': (0.30, -0.03, 0.02),
    'hip': (0.115, 0.04, 0.42), 'knee': (0.12, 0.0, 0.23), 'ankle': (0.12, 0.03, 0.07), 'toe': (0.12, -0.08, 0.01),
}
PARENT = {'chest': 'hips', 'head': 'chest', 'crown': 'head'}
for s in 'lr':
    PARENT.update({f'clavicle_{s}': 'chest', f'shoulder_{s}': f'clavicle_{s}', f'elbow_{s}': f'shoulder_{s}', f'wrist_{s}': f'elbow_{s}', f'hand_{s}': f'wrist_{s}',
                   f'hip_{s}': 'hips', f'knee_{s}': f'hip_{s}', f'ankle_{s}': f'knee_{s}', f'toe_{s}': f'ankle_{s}'})
JOINTS = dict(MID)
for k, (x, y, z) in SIDE.items():
    JOINTS[k + '_l'] = (x, y, z)
    JOINTS[k + '_r'] = (-x, y, z)
# the bone a joint's segment points to (for the weights) when it has several children
MAIN_CHILD = {'hips': 'chest', 'chest': 'head'}
ORDER = ['hips', 'chest', 'head', 'crown'] + [f'{k}_{s}' for s in 'lr' for k in SIDE]
LEAVES = {'crown', 'hand_l', 'hand_r', 'toe_l', 'toe_r'}
DEFORM = [n for n in ORDER if n not in LEAVES]

t0 = __import__('time').time()
log = lambda *a: print(f'[{__import__("time").time() - t0:6.1f}s]', *a, flush=True)

bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.import_scene.gltf(filepath=SRC)
mesh = next(o for o in bpy.context.scene.objects if o.type == 'MESH')
bpy.context.view_layer.objects.active = mesh
mesh.select_set(True)
bpy.ops.object.transform_apply(location=False, rotation=True, scale=True)
top = max(v.co.z for v in mesh.data.vertices)
SCALE = HEIGHT / top
J = {k: np.array(v) * SCALE for k, v in JOINTS.items()}
mesh.scale = (SCALE,) * 3
bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
bm = bmesh.new(); bm.from_mesh(mesh.data)
bmesh.ops.remove_doubles(bm, verts=bm.verts, dist=1e-4)
bm.to_mesh(mesh.data); bm.free()
d = mesh.modifiers.new('decimate', 'DECIMATE')
d.ratio = TRIS / len(mesh.data.polygons)
bpy.ops.object.modifier_apply(modifier='decimate')
# (smooth across the stones, their block edges kept sharp; the normal map gives the rest)
bpy.ops.object.shade_smooth_by_angle(angle=math.radians(50))
log('decimated to', len(mesh.data.polygons), 'faces')

arm = bpy.data.objects.new('rig', bpy.data.armatures.new('rig'))
bpy.context.scene.collection.objects.link(arm)
bpy.context.view_layer.objects.active = arm
bpy.ops.object.mode_set(mode='EDIT')
kids = {n: [c for c in ORDER if PARENT.get(c) == n] for n in ORDER}
tail = {n: MAIN_CHILD.get(n, kids[n][0]) for n in DEFORM}
eb = {}
for n in DEFORM:
    b = arm.data.edit_bones.new(n)
    b.head = Vector(J[n]); b.tail = Vector(J[tail[n]])
    eb[n] = b
for n in DEFORM:
    if PARENT.get(n) in eb: eb[n].parent = eb[PARENT[n]]
bpy.ops.object.mode_set(mode='OBJECT')
bpy.ops.object.select_all(action='DESELECT')
mesh.select_set(True); arm.select_set(True)
bpy.context.view_layer.objects.active = arm
bpy.ops.object.parent_set(type='ARMATURE_AUTO')

me = mesh.data
vg = {g.index: DEFORM.index(g.name) for g in mesh.vertex_groups}
segs = np.array([[J[n], J[tail[n]]] for n in DEFORM])
def nearest_bone(p):
    a, b = segs[:, 0], segs[:, 1]; ab = b - a
    u = np.clip(((p - a) * ab).sum(1) / (ab * ab).sum(1), 0, 1)
    return int(np.argmin(np.linalg.norm(a + ab * u[:, None] - p, axis=1)))
nv = len(me.vertices)
W = np.zeros((nv, 4)); I = np.zeros((nv, 4), np.int32)
missed = 0
for v in me.vertices:
    ws = sorted(((g.weight, vg[g.group]) for g in v.groups if g.weight > 1e-3), reverse=True)[:4]
    if not ws:
        missed += 1
        ws = [(1.0, nearest_bone(np.array(v.co)))]
    s = sum(w for w, _ in ws)
    for k, (w, j) in enumerate(ws): W[v.index, k] = w / s; I[v.index, k] = j
log('weighted;', missed, 'vertices took their nearest bone')

# the base colour at TEX (the model's normal map is all but flat: left out)
mat = mesh.material_slots[0].material
nodes = mat.node_tree.nodes
bsdf = next(n for n in nodes if n.type == 'BSDF_PRINCIPLED')
base = bsdf.inputs['Base Color'].links[0].from_node.image
ims = bpy.context.scene.render.image_settings
ims.file_format = 'JPEG'; ims.color_mode = 'RGB'; ims.quality = 82  # (save_render writes in the scene's format)
base.scale(TEX, TEX)
base.save_render(os.path.join(OUT, 'colossus.jpg'))

# Blender space → the game's: x' = −x, y' = z, z' = −y (det −1: the game's +X is the colossus's right)
C = np.array([[-1, 0, 0], [0, 0, 1], [0, -1, 0]], float)
me.calc_loop_triangles()
uv = me.uv_layers.active.data
cn = me.corner_normals
verts, index, seen = [], [], {}
for tri in me.loop_triangles:
    for li in (tri.loops[0], tri.loops[2], tri.loops[1]):  # (the frame's mirror flips the winding back)
        vi = me.loops[li].vertex_index
        n = cn[li].vector; t = uv[li].uv
        key = (vi, round(n.x, 3), round(n.y, 3), round(n.z, 3), round(t.x, 5), round(t.y, 5))
        k = seen.get(key)
        if k is None:
            k = seen[key] = len(verts)
            verts.append((vi, C @ np.array(me.vertices[vi].co), C @ np.array(n), (t.x, 1 - t.y)))
        index.append(k)
log(len(verts), 'GPU vertices,', len(index) // 3, 'triangles')

vb = bytearray()
for vi, p, n, t in verts:
    w = np.round(W[vi] * 255).astype(int); w[0] += 255 - w.sum()
    vb += struct.pack('<8f4B4B4B', *p, *n, *t, *I[vi], *w, 0, 0, 0, 0)
assert len(vb) == len(verts) * VERTEX_BYTES
ib = np.array(index, np.uint16 if len(verts) < 65536 else np.uint32).tobytes()
ib += b'\0' * (-len(ib) % 4)

lo = np.min([v[1] for v in verts], 0); hi = np.max([v[1] for v in verts], 0)
# The collision mesh: the bound mesh decimated again (its icicles and seams smoothed over), each vertex weighted as
# the drawn vertex nearest it, so the game skins it on the CPU with the very joints the GPU draws with.
from mathutils import kdtree
coll = mesh.copy(); coll.data = mesh.data.copy(); coll.modifiers.clear()
bpy.context.scene.collection.objects.link(coll)
bpy.ops.object.select_all(action='DESELECT')
coll.select_set(True); bpy.context.view_layer.objects.active = coll
d = coll.modifiers.new('decimate', 'DECIMATE')
d.ratio = COLL_TRIS / len(coll.data.polygons)
bpy.ops.object.modifier_apply(modifier='decimate')
kd = kdtree.KDTree(nv)
for v in me.vertices: kd.insert(v.co, v.index)
kd.balance()
cm = coll.data
cm.calc_loop_triangles()
cvb, cjw = bytearray(), bytearray()
for v in cm.vertices:
    _, near, _ = kd.find(v.co)
    w = np.round(W[near] * 255).astype(int); w[0] += 255 - w.sum()
    cvb += struct.pack('<3f', *(C @ np.array(v.co)))
    cjw += struct.pack('<4B4B', *I[near], *w)
cib = np.array([[t.vertices[0], t.vertices[2], t.vertices[1]] for t in cm.loop_triangles], np.uint16).tobytes()  # (the mirror: winding back)
cib += b'\0' * (-len(cib) % 4)
log('collision mesh:', len(cm.vertices), 'vertices,', len(cm.loop_triangles), 'triangles')
bpy.data.objects.remove(coll)

head = {
    'vertices': len(verts), 'indices': len(index), 'index32': len(verts) >= 65536, 'fps': 30,
    'joints': DEFORM, 'clips': [], 'bounds': [lo.tolist(), hi.tolist()], 'saddle': [0, 0, 0], 'saddleJoint': 0,
    'parents': [DEFORM.index(PARENT[n]) if n in PARENT else -1 for n in DEFORM],
    'rest': [[round(v, 4) for v in (C @ J[n]).tolist()] for n in DEFORM],
    # the leaves' places (the fists' and the toes' tips, the crown), for reach and IK
    'tips': {n: [round(v, 4) for v in (C @ J[n]).tolist()] for n in sorted(LEAVES)},
    # after the (absent) clips: positions f32×3, joints u8×4 + weights unorm8×4 a vertex, then u16 indices (padded to 4)
    'coll': {'vertices': len(cm.vertices), 'indices': len(cm.loop_triangles) * 3},
}
hj = json.dumps(head).encode()
hj += b' ' * (-(len(hj) + 8) % 4)
with open(os.path.join(OUT, 'colossus.bin'), 'wb') as f:
    f.write(b'HRS1' + struct.pack('<I', len(hj)) + hj + vb + ib + cvb + cjw + cib)
log('wrote colossus.bin', os.path.getsize(os.path.join(OUT, 'colossus.bin')) // 1024, 'KiB')

# --preview: the bound mesh skinned in a few poses (subtrees turned about their joints), each bone tinted
if preview:
    bpy.data.objects.remove(arm)
    mesh.hide_render = True
    base_p = np.array([v[1] for v in verts]); Ii = np.array([I[v[0]] for v in verts]); Wi = np.array([W[v[0]] for v in verts])
    tris = np.array(index).reshape(-1, 3)
    rest = {n: C @ J[n] for n in ORDER}
    def rx(a):
        c, s = math.cos(a), math.sin(a); return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
    def rz(a):
        c, s = math.cos(a), math.sin(a); return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    def pose(turns):
        """Each deforming joint's skinning matrix for local turns {joint: 3×3} (game frame, parents first)."""
        R, P = {}, {}
        for n in ORDER:
            p = PARENT.get(n)
            Rl = turns.get(n, np.eye(3))
            if p is None: R[n] = Rl; P[n] = rest[n]
            else: R[n] = R[p] @ Rl; P[n] = P[p] + R[p] @ (rest[n] - rest[p])
        return np.array([np.hstack([R[n], (P[n] - R[n] @ rest[n])[:, None]]) for n in DEFORM])
    # (in the game's frame: +X its right, pitch about X: negative raises a limb forward)
    shots = [
        {},
        {'shoulder_l': rx(-2.7), 'shoulder_r': rx(-2.7), 'elbow_l': rx(-0.5), 'elbow_r': rx(-0.5), 'chest': rx(0.25), 'clavicle_l': rz(0.3), 'clavicle_r': rz(-0.3)},
        {'hip_l': rx(-0.8), 'knee_l': rx(1.1), 'shoulder_r': rx(-0.5), 'shoulder_l': rx(0.4)},
        {'hips': rx(0.9), 'shoulder_l': rx(-1.2), 'shoulder_r': rx(-1.2), 'head': rz(0.3) @ rx(0.3)},
    ]
    tint = np.array([[(k * 0.37) % 1, ((k * 0.61) + 0.3) % 1, ((k * 0.83) + 0.6) % 1] for k in range(len(DEFORM))]) * 0.7 + 0.25
    for s, turns in enumerate(shots):
        M = pose(turns)
        p = np.zeros_like(base_p)
        for q in range(4):
            Mq = M[Ii[:, q]]
            p += Wi[:, q:q + 1] * (np.einsum('nij,nj->ni', Mq[:, :, :3], base_p) + Mq[:, :, 3])
        p = p @ C
        pm = bpy.data.meshes.new(f'p{s}'); pm.from_pydata(p.tolist(), [], tris[:, ::-1].tolist())
        o = bpy.data.objects.new(f'p{s}', pm); bpy.context.scene.collection.objects.link(o)
        o.location = (s * 40 - 60, 0, 0)
        ca = pm.color_attributes.new('bone', 'BYTE_COLOR', 'POINT')
        for i2 in range(len(p)): ca.data[i2].color = (*tint[Ii[i2, 0] % len(tint)], 1)
    sc = bpy.context.scene
    sc.render.engine = 'BLENDER_WORKBENCH'
    sc.display.shading.light = 'STUDIO'; sc.display.shading.color_type = 'VERTEX'
    sc.render.resolution_x, sc.render.resolution_y = 1800, 700
    cam = bpy.data.objects.new('cam', bpy.data.cameras.new('cam')); sc.collection.objects.link(cam); sc.camera = cam
    cam.data.type = 'ORTHO'; cam.data.ortho_scale = 170
    # (from its front)
    cam.location = (0, -150, 30); cam.rotation_euler = (math.radians(85), 0, 0)
    sc.render.filepath = preview
    bpy.ops.render.render(write_still=True)
    log('preview', preview)
