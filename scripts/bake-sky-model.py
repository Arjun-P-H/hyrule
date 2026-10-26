"""
Bake a sky-island model (a .glb) for the land's model isles (src/lib/skymodels.ts).
-----------------------------------------------------------------------------
    /Applications/Blender.app/Contents/MacOS/Blender -b --python scripts/bake-sky-model.py -- <in.glb> <name> [across_m=300]

Writes public/models/<name>.bin (skymodel_pack.py: one part, no props) and <name>.webp:
  - the model scaled to `across_m` over its wider side, its footprint's middle at x = z = 0 and its plaza (the
    commonest floor height) at y = 0;
  - three levels of detail (LOD_TRIS), decimated, on the model's uvs (its base colour texture at TEX_SIZE);
  - its floors.
Then add <name> to SKY_MODELS.names (src/lib/constants.ts).
"""

import bpy, bmesh, sys, math, os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import skymodel_pack as pack

LOD_TRIS = [400_000, 60_000, 8_000]
# vertices this close (share of the model's width) are welded before each level's decimation: the coarse
# levels need their fragments joined, or the collapse stalls on them
LOD_WELD = [0.0, 0.0005, 0.003]
TEX_SIZE = 4096

args = sys.argv[sys.argv.index('--') + 1:]
src, name = args[0], args[1]
across = float(args[2]) if len(args) > 2 else 300.0
out_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'public', 'models')
os.makedirs(out_dir, exist_ok=True)

bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.import_scene.gltf(filepath=src)
meshes = [o for o in bpy.context.scene.objects if o.type == 'MESH']
bpy.context.view_layer.objects.active = meshes[0]
for o in meshes: o.select_set(True)
if len(meshes) > 1: bpy.ops.object.join()
obj = bpy.context.view_layer.objects.active

# scale and centre: the wider side across `across` m, the footprint's middle at the origin
bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
co = np.empty(len(obj.data.vertices) * 3, np.float32); obj.data.vertices.foreach_get('co', co); co = co.reshape(-1, 3)
lo, hi = co.min(0), co.max(0)
k = across / max(hi[0] - lo[0], hi[1] - lo[1])
obj.scale = (k, k, k)
obj.location = (-(lo[0] + hi[0]) / 2 * k, -(lo[1] + hi[1]) / 2 * k, 0)
bpy.ops.object.transform_apply(location=True, rotation=False, scale=True)

# the base colour's uv transform (gltfpack's quantised uvs come with one: the importer leaves it in a Mapping node)
mat = obj.active_material
mapping = next((n for n in mat.node_tree.nodes if n.type == 'MAPPING'), None)
uv_xf = (np.array(mapping.inputs['Scale'].default_value[:2] if mapping else (1, 1), np.float64),
         np.array(mapping.inputs['Location'].default_value[:2] if mapping else (0, 0), np.float64))
bsdf = next(n for n in mat.node_tree.nodes if n.type == 'BSDF_PRINCIPLED')
base_img = bsdf.inputs['Base Color'].links[0].from_node.image

# weld the seams gltfpack split, so the decimation keeps the surface whole
bm = bmesh.new(); bm.from_mesh(obj.data)
bmesh.ops.remove_doubles(bm, verts=bm.verts, dist=1e-5 * across)
bm.to_mesh(obj.data); bm.free()
print('welded', len(obj.data.vertices), 'verts', flush=True)


def decimated(source, tris, weld):
    o = source.copy(); o.data = source.data.copy(); bpy.context.scene.collection.objects.link(o)
    if weld:
        bm = bmesh.new(); bm.from_mesh(o.data)
        bmesh.ops.remove_doubles(bm, verts=bm.verts, dist=weld * max(o.dimensions))
        bm.to_mesh(o.data); bm.free()
    have = sum(len(p.vertices) - 2 for p in o.data.polygons)
    if tris < have:
        m = o.modifiers.new('dec', 'DECIMATE'); m.ratio = tris / have; m.use_collapse_triangulate = True
        bpy.context.view_layer.objects.active = o
        bpy.ops.object.modifier_apply(modifier=m.name)
    o.data.shade_smooth()
    bpy.context.view_layer.objects.active = o
    for x in bpy.context.selected_objects: x.select_set(False)
    o.select_set(True)
    bpy.ops.object.shade_smooth_by_angle(angle=math.radians(50))
    return o


lods, prev = [], obj
for t, weld in zip(LOD_TRIS, LOD_WELD):
    prev = decimated(prev, t, weld)
    lods.append(pack.mesh_corners(prev, uv_xf=uv_xf))
    print('lod', len(lods) - 1, len(lods[-1][0]) // 3, 'tris', flush=True)

# the floors, then the plaza (the commonest height of the top floor, 1 m bins) made y = 0
fh, fb = pack.floors(lods[0][0])
off = np.frombuffer(fb, np.uint32, fh['nx'] * fh['nz'] + 1)
q = np.frombuffer(fb, np.uint16, offset=off.nbytes)
F = q[off[:-1][off[1:] > off[:-1]]]  # (each cell's top layer)
ys = fh['min'] + ((F >> 1).astype(np.float64) - 1) / 32766 * (fh['max'] - fh['min'])
hist, edges = np.histogram(ys, bins=max(int(ys.max() - ys.min()), 1))
plaza = float(edges[hist.argmax()] + 0.5)
fh['min'] -= plaza; fh['max'] -= plaza

sections, infos = [], []
for P, N, UV, C, AO, VIS, MAT in lods:
    vb, ib, info = pack.encode(P - np.array([0, plaza, 0], np.float32), N, UV, C, AO, VIS, MAT)
    sections += [vb, ib]; infos.append(info)
sections += [b'', fb]
header = {'version': 2, 'boxMin': infos[0]['lo'], 'boxMax': infos[0]['hi'], 'texture': True,
          'parts': [{'kind': 'solid', 'near': None, 'lods': infos}], 'protos': [], 'props': {'count': 0, 'chunks': []}, 'floor': fh,
          'plants': [], 'falls': []}
raw = pack.write(os.path.join(out_dir, name + '.bin'), header, sections)
print('wrote', name, f'{raw / 1e6:.1f} MB raw', 'lods', [(i['verts'], i['tris']) for i in infos], flush=True)

base_img.scale(TEX_SIZE, TEX_SIZE)
sc = bpy.context.scene.render.image_settings
sc.file_format = 'WEBP'; sc.quality = 88
base_img.save_render(os.path.join(out_dir, name + '.webp'))
print('done', flush=True)
