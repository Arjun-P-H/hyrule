"""
Build the horse for the game: the painted horse mesh (Horse.glb) bound to the skeleton and clips of Horse Rigs.glb,
packed into public/models/horse.bin with its base colour as public/models/horse.jpg.
-----------------------------------------------------------------------------
    /Applications/Blender.app/Contents/MacOS/Blender -b --python scripts/horse/build.py -- [source dir] [--preview out.png]

The rig's proportions are not the mesh's (its legs are a quarter longer, its neck higher), so the joints are not
taken from the rig: JOINTS places every one on the mesh (read off a side view at the game's scale, SCALE). The rig
gives what does not depend on the bone lengths: the hierarchy and every clip's joint rotations (all its clips key
rotations, and the hips' translation). The bind pose is the rig's Idle's first rotations (REST) with each joint's offset from
its parent re-expressed so that the joints land on JOINTS; a clip's pose is its rotations on those same offsets.
Blender's bone heat weights the decimated mesh to that skeleton. Each clip is baked at FPS into one skinning matrix
(3×4) a deforming joint a frame. Everything is written in the game's horse-local frame: metres, +Y up, facing +Z,
its right side +X (the hero's convention, rig.ts heroLocalToWorld), the origin on the ground under the girth.

horse.bin: 'HRS1', u32 JSON length, JSON (counts, joint names, parents and rest places, clips, saddle), padded to 4; then the vertices
(VERTEX_BYTES each: position f32×3, normal f32×3, uv f32×2, joints u8×4, weights unorm8×4, tags unorm8×4: x the mane
and tail (HAIR), the rest 0), the indices (u16 when
they fit, else u32, padded to 4), then every clip's frames (frames × joints × 12 f32, rows of the 3×4 matrix).
"""

import bpy, bmesh, sys, os, json, struct, math
import numpy as np
from mathutils import Vector, Matrix, Quaternion

args = sys.argv[sys.argv.index('--') + 1:] if '--' in sys.argv else []
preview = args[args.index('--preview') + 1] if '--preview' in args else None
args = [a for i, a in enumerate(args) if a != '--preview' and (i == 0 or args[i - 1] != '--preview')]
SRC = args[0] if args else os.path.expanduser('~/Desktop/3D Characters/Animals/Horse')
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'public', 'models')

SCALE = 2.42          # the mesh's units to metres: 1.55 m at the withers
TRIS = 14000          # triangles after decimation
TEX = 1024            # the base colour's size (px)
FPS = 30
VERTEX_BYTES = 44
# clips that loop (the others play once and hold their last frame)
LOOPS = {'Idle', 'Walk', 'Trot', 'Run', 'Eating', 'Sleep', 'Turn_Left', 'Turn_Right'}
SKIP = {'Rest_Pose'}
# the bind pose's rotations: Idle's first frame stands as the mesh does (Rest_Pose holds the tail out straight)
REST = 'Idle'

# Blender space (Z up, facing −Y, the horse's left +X), metres. Read off a side view at 437.5 px/m centred on
# (y 0, z 1): P(px, py, x). The left side's joints are mirrored to the right.
def P(px, py, x=0.0):
    return (x, (px - 700) / 437.5, 1.0 - (py - 525) / 437.5)

MID = {
    'hips': P(900, 440), 'spine_1': P(800, 430), 'spine_2': P(690, 425), 'spine_3': P(580, 415), 'spine_4': P(510, 400),
    'spine_5': P(420, 360), 'spine_6': P(340, 300), 'head': P(250, 235), 'nose': P(141, 359), 'head_leaf': P(105, 400),
    'mouth': P(150, 395), 'mouth_tip': P(130, 435),
    'ribcage_middle': P(590, 540), 'ribcage_leaf': P(600, 640), 'ribcage_front': P(420, 560), 'ribcage_front_leaf': P(395, 600),
    'tail_1': P(975, 405), 'tail_2': P(1030, 470), 'tail_3': P(1060, 600), 'tail_4': P(1100, 730), 'tail_leaf': P(1170, 830),
}
SIDE = {
    'ear_1': P(213, 200, 0.07), 'ear_2': P(200, 178, 0.09), 'ear_leaf': P(182, 152, 0.11),
    'front_scapula': P(560, 400, 0.12), 'front_humerus': P(420, 520, 0.17), 'front_leg_upper': P(500, 640, 0.16),
    'front_leg_lower': P(468, 765, 0.12), 'front_leg_ankle': P(478, 880, 0.125), 'front_leg_foot': P(455, 930, 0.135),
    'front_leg_leaf': P(430, 955, 0.14),
    'back_leg_pelvis': P(880, 430, 0.11), 'back_leg_upper': P(960, 520, 0.18), 'back_leg_lower': P(900, 640, 0.18),
    'back_leg_ankle': P(1015, 790, 0.11), 'back_leg_foot': P(1010, 895, 0.14), 'back_leg_toe': P(985, 935, 0.14),
    'back_leg_leaf': P(960, 958, 0.14),
}
JOINTS = dict(MID)
for k, (x, y, z) in SIDE.items():
    JOINTS[k + '_l'] = (x, y, z)
    JOINTS[k + '_r'] = (-x, y, z)
# the rider's seat: on the back behind the withers (Blender space), carried by SADDLE_JOINT
SADDLE = (0.0, -0.12, 1.36)
SADDLE_JOINT = 'spine_2'
# the bone a joint's segment points to (for the weights) when it has several children
MAIN_CHILD = {'hips': 'spine_1', 'spine_3': 'spine_4', 'spine_4': 'spine_5', 'head': 'nose', 'nose': 'head_leaf'}

t0 = __import__('time').time()
log = lambda *a: print(f'[{__import__("time").time() - t0:6.1f}s]', *a, flush=True)


# --- the rig, read straight from the glTF ------------------------------------------------------------------------

def read_glb(path):
    b = open(path, 'rb').read()
    n = struct.unpack_from('<I', b, 12)[0]
    j = json.loads(b[20:20 + n])
    binc = b[20 + n + 8:]
    def acc(i):
        a = j['accessors'][i]; v = j['bufferViews'][a['bufferView']]
        comps = {'SCALAR': 1, 'VEC2': 2, 'VEC3': 3, 'VEC4': 4, 'MAT4': 16}[a['type']]
        assert a['componentType'] == 5126 and not v.get('byteStride')
        off = v.get('byteOffset', 0) + a.get('byteOffset', 0)
        return np.frombuffer(binc, np.float32, a['count'] * comps, off).reshape(a['count'], comps).astype(np.float64)
    return j, acc

def q2m(q):  # glTF (x, y, z, w) → 3×3
    return np.array(Quaternion((q[3], q[0], q[1], q[2])).to_matrix())

def sample(times, vals, t, interp, rot):
    if t <= times[0] or len(times) == 1: return vals[0]
    if t >= times[-1]: return vals[-1]
    i = int(np.searchsorted(times, t)) - 1
    if interp == 'STEP': return vals[i]
    u = (t - times[i]) / (times[i + 1] - times[i])
    if rot:
        a = Quaternion((vals[i][3], *vals[i][:3])); b = Quaternion((vals[i + 1][3], *vals[i + 1][:3]))
        q = a.slerp(b, u)
        return np.array((q.x, q.y, q.z, q.w))
    return vals[i] * (1 - u) + vals[i + 1] * u

gj, acc = read_glb(os.path.join(SRC, 'Horse Rigs.glb'))
nodes = gj['nodes']
names = [n['name'] for n in nodes]
parent = {}
for i, n in enumerate(nodes):
    for c in n.get('children', []): parent[c] = i
# the joints below the root (the root only turns the rig to +Y up; Blender space is the root's)
order = [i for i in range(len(nodes)) if names[i] != 'root']
def depth(i):
    return 0 if i not in parent else 1 + depth(parent[i])
order.sort(key=depth)  # parents before children
assert set(names[i] for i in order) == set(JOINTS), set(names[i] for i in order) ^ set(JOINTS)
leaves = {i for i in order if not nodes[i].get('children')}
deform = [i for i in order if i not in leaves]
log(len(order), 'joints,', len(deform), 'deforming')

def clip_tracks(anim):
    tr = {}
    for ch in anim['channels']:
        s = anim['samplers'][ch['sampler']]
        tr[(ch['target']['node'], ch['target']['path'])] = (acc(s['input'])[:, 0], acc(s['output']), s.get('interpolation', 'LINEAR'))
    return tr

def pose_at(tr, t):
    """Every joint's local rotation (3×3) at t, and the hips' translation."""
    R = {}
    for i in order:
        k = tr.get((i, 'rotation'))
        R[i] = q2m(sample(*k[:2], t, k[2], True)) if k else q2m(nodes[i].get('rotation', [0, 0, 0, 1]))
    hips = names.index('hips')
    k = tr.get((hips, 'translation'))
    T = sample(*k[:2], t, k[2], False) if k else np.array(nodes[hips].get('translation', [0, 0, 0]))
    return R, T

anims = {a['name']: a for a in gj['animations']}
hips = names.index('hips')
refR, refT = pose_at(clip_tracks(anims[REST]), 0.0)
J = {i: np.array(JOINTS[names[i]]) for i in order}
# bind: the rest rotations, offsets re-expressed in each parent's rest frame so the joints land on J
bindRot, offset = {}, {}
for i in order:
    p = parent[i]
    bindRot[i] = refR[i] if p == names.index('root') else bindRot[p] @ refR[i]
    if p != names.index('root'): offset[i] = bindRot[p].T @ (J[i] - J[p])
# the hips' travel in a clip is the rig's, scaled to this horse's height
HIP_SCALE = J[hips][2] / refT[2]

def world_pose(R, T):
    rot, pos = {}, {}
    for i in order:
        p = parent[i]
        if i == hips:
            rot[i] = R[i]; pos[i] = J[hips] + (T - refT) * HIP_SCALE
        else:
            rot[i] = rot[p] @ R[i]; pos[i] = pos[p] + rot[p] @ offset[i]
    return rot, pos

# Blender space → the game's horse-local frame: x' = −x, y' = z, z' = −y (det −1: the game's +X is the horse's right)
C = np.array([[-1, 0, 0], [0, 0, 1], [0, -1, 0]], float)

def skin_frames(anim):
    tr = clip_tracks(anim)
    t0_, t1_ = min(v[0][0] for v in tr.values()), max(v[0][-1] for v in tr.values())
    n = int(round((t1_ - t0_) * FPS)) + 1
    out = np.zeros((n, len(deform), 12), np.float32)
    for f in range(n):
        rot, pos = world_pose(*pose_at(tr, t0_ + f / FPS))
        for k, i in enumerate(deform):
            Rm = rot[i] @ bindRot[i].T
            tv = pos[i] - Rm @ J[i]
            M = np.hstack([C @ Rm @ C.T, (C @ tv)[:, None]])
            out[f, k] = M.reshape(-1)
    return out


# --- the mesh -----------------------------------------------------------------------------------------------------

bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.import_scene.gltf(filepath=os.path.join(SRC, 'Horse.glb'))
mesh = next(o for o in bpy.context.scene.objects if o.type == 'MESH')
mesh.scale = (SCALE,) * 3
bpy.context.view_layer.objects.active = mesh
mesh.select_set(True)
bpy.ops.object.transform_apply(location=False, rotation=True, scale=True)
bm = bmesh.new(); bm.from_mesh(mesh.data)
bmesh.ops.remove_doubles(bm, verts=bm.verts, dist=1e-5)
bm.to_mesh(mesh.data); bm.free()
d = mesh.modifiers.new('decimate', 'DECIMATE')
d.ratio = TRIS / len(mesh.data.polygons)
bpy.ops.object.modifier_apply(modifier='decimate')
log('decimated to', len(mesh.data.polygons), 'faces')

# the armature at JOINTS, for the weights alone
arm = bpy.data.objects.new('rig', bpy.data.armatures.new('rig'))
bpy.context.scene.collection.objects.link(arm)
bpy.context.view_layer.objects.active = arm
bpy.ops.object.mode_set(mode='EDIT')
eb = {}
for i in deform:
    kids = nodes[i]['children']
    main = names.index(MAIN_CHILD[names[i]]) if names[i] in MAIN_CHILD else kids[0]
    b = arm.data.edit_bones.new(names[i])
    b.head = Vector(J[i]); b.tail = Vector(J[main])
    eb[i] = b
for i in deform:
    if parent[i] in eb: eb[i].parent = eb[parent[i]]
bpy.ops.object.mode_set(mode='OBJECT')
bpy.ops.object.select_all(action='DESELECT')
mesh.select_set(True); arm.select_set(True)
bpy.context.view_layer.objects.active = arm
bpy.ops.object.parent_set(type='ARMATURE_AUTO')

# the four strongest weights a vertex; a vertex bone heat missed takes its nearest bone
me = mesh.data
vg = {g.index: names.index(g.name) for g in mesh.vertex_groups}
slot = {i: k for k, i in enumerate(deform)}
segs = np.array([[J[i], J[names.index(MAIN_CHILD[names[i]]) if names[i] in MAIN_CHILD else nodes[i]['children'][0]]] for i in deform])
nv = len(me.vertices)
W = np.zeros((nv, 4)); I = np.zeros((nv, 4), np.int32)
missed = 0
for v in me.vertices:
    ws = sorted(((g.weight, slot[vg[g.group]]) for g in v.groups if g.weight > 1e-4), reverse=True)[:4]
    if not ws:
        missed += 1
        p = np.array(v.co); a, b = segs[:, 0], segs[:, 1]; ab = b - a
        u = np.clip(((p - a) * ab).sum(1) / (ab * ab).sum(1), 0, 1)
        ws = [(1.0, int(np.argmin(np.linalg.norm(a + ab * u[:, None] - p, axis=1))))]
    s = sum(w for w, _ in ws)
    for k, (w, j) in enumerate(ws): W[v.index, k] = w / s; I[v.index, k] = j
log('weighted;', missed, 'vertices took their nearest bone')

# the base colour, at TEX
mat = mesh.material_slots[0].material
bsdf = next(n for n in mat.node_tree.nodes if n.type == 'BSDF_PRINCIPLED')
img = bsdf.inputs['Base Color'].links[0].from_node.image
img.scale(TEX, TEX)
img.filepath_raw = os.path.join(OUT, 'horse.jpg'); img.file_format = 'JPEG'
bpy.context.scene.render.image_settings.quality = 88
img.save_render(os.path.join(OUT, 'horse.jpg'))

# corners → GPU vertices (a vertex for each distinct position, normal and uv), in the game's frame
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
# The hair (the game colours it apart from the coat): the tail (its joints carry it), and the mane and forelock —
# the painted cream hair on the neck and the poll, above the line from the withers to the poll.
TAIL = {i for i in deform if names[i].startswith('tail_')}
NECK = {i for i in deform if names[i] in ('spine_4', 'spine_5', 'spine_6', 'head')}
px = np.array(img.pixels[:], np.float32).reshape(TEX, TEX, 4)
na, nb = J[names.index('spine_4')][1:], J[names.index('head')][1:]
up = np.array([nb[1] - na[1], na[0] - nb[0]]); up /= np.linalg.norm(up)
def hair(vi, t):
    dom = deform[I[vi, 0]]
    if dom in TAIL: return 1.0
    if dom not in NECK: return 0.0
    r, g, b = px[min(TEX - 1, int(t[1] * TEX)) % TEX, min(TEX - 1, int(t[0] * TEX)) % TEX, :3]
    lum, sat = 0.2126 * r + 0.7152 * g + 0.0722 * b, max(r, g, b) - min(r, g, b)
    above = np.dot(np.array(me.vertices[vi].co)[1:] - na, up)
    # (the crest is mane however it is painted; lower down, the cream hair hanging over the neck)
    return 1.0 if above > 0.16 or (above > 0.02 and lum > 0.58 and sat < 0.32) else 0.0
for vi, p, n, t in verts:
    w = np.round(W[vi] * 255).astype(int); w[0] += 255 - w.sum()
    vb += struct.pack('<8f4B4B4B', *p, *n, *t, *I[vi], *w, int(255 * hair(vi, (t[0], 1 - t[1]))), 0, 0, 0)
assert len(vb) == len(verts) * VERTEX_BYTES
ib = np.array(index, np.uint16 if len(verts) < 65536 else np.uint32).tobytes()
ib += b'\0' * (-len(ib) % 4)

clips, cb = [], bytearray()
for a in gj['animations']:
    if a['name'] in SKIP: continue
    fr = skin_frames(a)
    clips.append({'name': a['name'], 'frames': len(fr), 'loop': a['name'] in LOOPS})
    cb += fr.tobytes()
log('clips:', ', '.join(f"{c['name']} {c['frames']}" for c in clips))

lo = np.min([v[1] for v in verts], 0); hi = np.max([v[1] for v in verts], 0)
head = {
    'vertices': len(verts), 'indices': len(index), 'index32': len(verts) >= 65536, 'fps': FPS,
    'joints': [names[i] for i in deform], 'clips': clips, 'bounds': [lo.tolist(), hi.tolist()],
    'saddle': (C @ np.array(SADDLE)).tolist(), 'saddleJoint': deform.index(names.index(SADDLE_JOINT)),
    # each joint's parent (−1: the hips) and place in the bind pose, for turns the game adds (the head looking round)
    'parents': [deform.index(parent[i]) if parent[i] in deform else -1 for i in deform],
    'rest': [[round(v, 5) for v in (C @ J[i]).tolist()] for i in deform],
}
hj = json.dumps(head).encode()
hj += b' ' * (-(len(hj) + 8) % 4)
with open(os.path.join(OUT, 'horse.bin'), 'wb') as f:
    f.write(b'HRS1' + struct.pack('<I', len(hj)) + hj + vb + ib + cb)
log('wrote horse.bin', os.path.getsize(os.path.join(OUT, 'horse.bin')) // 1024, 'KiB')

# --preview: the bound mesh posed from the baked frames (exactly what the game draws), a strip of clips
if preview:
    bpy.data.objects.remove(arm)
    base = np.array([v[1] for v in verts]); Ii = np.array([I[v[0]] for v in verts]); Wi = np.array([W[v[0]] for v in verts])
    shots = [('Idle', 0), ('Walk', 8), ('Trot', 5), ('Run', 4), ('Eating', 40), ('Rear', 40), ('Kick', 15), ('Sleep', 30)]
    frames = {c['name']: (k, c) for k, c in enumerate(clips)}
    starts = np.cumsum([0] + [c['frames'] for c in clips])
    allf = np.frombuffer(bytes(cb), np.float32).reshape(-1, len(deform), 3, 4)
    tris = np.array(index).reshape(-1, 3)
    for s, (cname, f) in enumerate(shots):
        k, c = frames[cname]
        M = allf[starts[k] + min(f, c['frames'] - 1)]
        p = np.zeros_like(base)
        for q in range(4):
            Mq = M[Ii[:, q]]
            p += Wi[:, q:q + 1] * (np.einsum('nij,nj->ni', Mq[:, :, :3], base) + Mq[:, :, 3])
        p = p @ C  # back to Blender space (C is its own inverse transposed)
        pm = bpy.data.meshes.new(cname); pm.from_pydata((p + [0, 0, 0]).tolist(), [], tris[:, ::-1].tolist())
        o = bpy.data.objects.new(cname, pm); bpy.context.scene.collection.objects.link(o)
        o.location = (0, (s % 4) * 3.2 - 4.8, -(s // 4) * 2.6)
        o.color = (0.8, 0.7, 0.55, 1)
        # (the hair tagged: red)
        hv = np.array([hair(v[0], (v[3][0], 1 - v[3][1])) for v in verts])
        ca = pm.color_attributes.new('hair', 'BYTE_COLOR', 'POINT')
        for i2, hh in enumerate(hv): ca.data[i2].color = (0.8 + 0.2 * hh, 0.7 * (1 - hh), 0.55 * (1 - hh), 1)
    mesh.hide_render = True
    sc = bpy.context.scene
    sc.render.engine = 'BLENDER_WORKBENCH'
    sc.display.shading.light = 'STUDIO'; sc.display.shading.color_type = 'VERTEX'
    sc.render.resolution_x, sc.render.resolution_y = 1600, 900
    cam = bpy.data.objects.new('cam', bpy.data.cameras.new('cam')); sc.collection.objects.link(cam); sc.camera = cam
    cam.data.type = 'ORTHO'; cam.data.ortho_scale = 13.5
    cam.location = (20, 0, -0.3); cam.rotation_euler = (math.radians(90), 0, math.radians(90))
    sc.render.filepath = preview
    bpy.ops.render.render(write_still=True)
    log('preview', preview)
