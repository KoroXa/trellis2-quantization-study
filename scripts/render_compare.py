"""Blender 4.5 background script: 2x3 comparison grid (settings x quants).

Rows (shelves, Z offset): standard (baseline postprocess) on top,
optimized (combo) at bottom. Columns (X): bf16 | int8 | nvfp4.
Renders textured, then swaps to a shared clay material and renders again.

Usage:
  blender.exe --background --python render_compare.py -- <out_textured> <out_clay>
"""
import bpy
import math
import os
import sys

argv = sys.argv[sys.argv.index("--") + 1:]
OUT_TEX, OUT_CLAY = argv[0], argv[1]

MODELS_DIR = r"E:\AI\ComfyUI_windows_portable\ComfyUI\output\3d"
CELLS = [
    # (filename, col_x, row_z, label)
    ("ComfyUI_00011.glb", -2.6, +1.5, "bf16_std"),
    ("ComfyUI_00010.glb", 0.0, +1.5, "int8_std"),
    ("ComfyUI_00009.glb", +2.6, +1.5, "nvfp4_std"),
    ("ComfyUI_00014.glb", -2.6, -1.5, "bf16_opt"),
    ("ComfyUI_00007.glb", 0.0, -1.5, "int8_opt"),
    ("ComfyUI_00015.glb", +2.6, -1.5, "nvfp4_opt"),
]

# --- clean scene ---
bpy.ops.object.select_all(action="SELECT")
bpy.ops.object.delete(use_global=False)
for coll in (bpy.data.meshes, bpy.data.materials, bpy.data.images,
             bpy.data.lights, bpy.data.cameras):
    for x in list(coll):
        coll.remove(x)

# --- import + place + upright via PCA ---
import numpy as np
from mathutils import Vector, Quaternion

for fname, cx, cz, label in CELLS:
    before = set(bpy.data.objects)
    bpy.ops.import_scene.gltf(filepath=os.path.join(MODELS_DIR, fname))
    new_objs = [o for o in bpy.data.objects if o not in before]
    meshes = [o for o in new_objs if o.type == "MESH"]
    assert len(meshes) == 1, f"{label}: {len(meshes)} meshes"
    mo = meshes[0]
    # longest principal axis of vertices (object space)
    n = len(mo.data.vertices)
    co = np.empty(n * 3, dtype=np.float64)
    mo.data.vertices.foreach_get("co", co)
    co = co.reshape(n, 3)
    _, _, vt = np.linalg.svd(co - co.mean(axis=0), full_matrices=False)
    ax = vt[0] / np.linalg.norm(vt[0])
    # head (wider end) must point +Z: compare radial spread above/below median
    t = (co - co.mean(axis=0)) @ ax
    med = np.median(t)
    perp = co - np.outer(t, ax)
    spread_hi = np.linalg.norm(perp[t > med] - perp[t > med].mean(axis=0), axis=1).mean()
    spread_lo = np.linalg.norm(perp[t <= med] - perp[t <= med].mean(axis=0), axis=1).mean()
    if spread_hi < spread_lo:
        ax = -ax
    empty = bpy.data.objects.new(label, None)
    bpy.context.collection.objects.link(empty)
    empty.location = (cx, 0.0, cz)
    q = Vector(ax).rotation_difference(Vector((0, 0, 1)))
    qyaw = Quaternion((0, 0, 1), math.radians(65.0))
    empty.rotation_mode = "QUATERNION"
    empty.rotation_quaternion = (qyaw @ q).normalized()
    q = qyaw @ q
    for o in new_objs:
        if o.parent is None and o != empty:
            o.parent = empty
    # center geometry on empty origin: transform the LOCAL bbox center
    # (a single point transforms exactly; corner min/max would inflate)
    bpy.context.view_layer.update()
    M = empty.matrix_world
    xs = [c[0] for c in mo.bound_box]
    ys = [c[1] for c in mo.bound_box]
    zs = [c[2] for c in mo.bound_box]
    cx = (min(xs) + max(xs)) / 2.0
    cy = (min(ys) + max(ys)) / 2.0
    cz = (min(zs) + max(zs)) / 2.0
    lc = Vector((cx, cy, cz))
    c = M @ lc
    mo.location -= (M.to_3x3().inverted() @ (c - empty.location))
    print(f"cell {label}: upright, head-up={spread_hi >= spread_lo}", flush=True)

# --- world / background ---
world = bpy.data.worlds.new("Bg")
world.use_nodes = True
world.node_tree.nodes["Background"].inputs["Color"].default_value = (0.16, 0.16, 0.17, 1.0)
world.node_tree.nodes["Background"].inputs["Strength"].default_value = 1.0
bpy.context.scene.world = world

# --- lights ---
def area_light(name, loc, energy, size=3.0):
    bpy.ops.object.light_add(type="AREA", location=loc)
    l = bpy.context.active_object
    l.name = name
    l.data.energy = energy
    l.data.size = size
    l.data.shape = "RECTANGLE"
    return l

key = area_light("Key", (4.5, -6.0, 7.0), 900.0)
fill = area_light("Fill", (-5.0, -4.0, 3.0), 280.0)
rim = area_light("Rim", (0.0, 6.0, 5.0), 450.0)

# --- ground shadow catcher ---
bpy.ops.mesh.primitive_plane_add(size=40, location=(0, 0, -2.15))
ground = bpy.context.active_object
ground.name = "Ground"
ground.is_shadow_catcher = True

# --- camera ---
bpy.ops.object.camera_add(location=(0.0, -8.0, 0.5), rotation=(math.radians(87.0), 0, 0))
cam = bpy.context.active_object
cam.name = "Camera"
cam.data.lens = 40
cam.data.clip_start = 0.1
cam.data.clip_end = 100.0
bpy.context.scene.camera = cam

# --- render settings ---
scene = bpy.context.scene
scene.render.engine = "BLENDER_EEVEE_NEXT"
scene.render.resolution_x = 1920
scene.render.resolution_y = 1280
scene.render.resolution_percentage = 100
scene.render.film_transparent = False
scene.render.image_settings.file_format = "PNG"
if hasattr(scene.eevee, "taa_render_samples"):
    scene.eevee.taa_render_samples = 128

def render(path):
    scene.render.filepath = path
    bpy.ops.render.render(write_still=True)
    print("rendered", path, flush=True)

render(OUT_TEX)

# --- clay override ---
clay = bpy.data.materials.new("Clay")
clay.use_nodes = True
bsdf = clay.node_tree.nodes["Principled BSDF"]
bsdf.inputs["Base Color"].default_value = (0.62, 0.62, 0.63, 1.0)
bsdf.inputs["Roughness"].default_value = 0.85
bsdf.inputs["Metallic"].default_value = 0.0
for o in bpy.data.objects:
    if o.type == "MESH":
        o.data.materials.clear()
        o.data.materials.append(clay)

render(OUT_CLAY)
print("DONE", flush=True)
