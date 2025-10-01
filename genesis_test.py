# genesis_vanilla_tree_occlusion.py
# Runs on macOS/Apple Silicon with Genesis 0.3.x using the Rasterizer backend (no LuisaRenderPy).
import os, json, numpy as np
import imageio.v2 as iio
import genesis as gs

OUT = "output"
for d in ["rgb", "depth", "seg_modal", "seg_amodal"]:
    os.makedirs(os.path.join(OUT, d), exist_ok=True)

def add_branch_segment(scene, pos, size, color=(0.4, 0.3, 0.2)):
    ent = scene.add_entity(
        gs.morphs.Box(pos=pos, size=size),
        surface=gs.surfaces.Default(color=color)
    )
    return ent, {"pos": tuple(pos), "size": tuple(size)}

def make_toy_tree(scene):
    segs, spec = [], []
    s, sp = add_branch_segment(scene, (0.0, 0.0, 0.50), (0.07, 0.07, 1.00)); segs.append(s); spec.append(sp)  # trunk
    s, sp = add_branch_segment(scene, (0.20, 0.0, 1.10), (0.06, 0.40, 0.06)); segs.append(s); spec.append(sp)
    s, sp = add_branch_segment(scene, (-0.22, 0.0, 1.25), (0.06, 0.35, 0.06)); segs.append(s); spec.append(sp)
    s, sp = add_branch_segment(scene, (0.0, 0.22, 1.35), (0.06, 0.06, 0.35)); segs.append(s); spec.append(sp)
    return segs, spec

def sprinkle_leaves(scene, n=120, spread=0.6, z0=1.1, seed=123):
    rng = np.random.default_rng(seed)
    for _ in range(n):
        x = rng.uniform(-spread, spread); y = rng.uniform(-spread, spread); z = z0 + rng.uniform(-0.25, 0.25)
        s = rng.uniform(0.03, 0.07)
        scene.add_entity(
            gs.morphs.Box(pos=(x, y, z), size=(s*1.6, s, s*0.3)),
            surface=gs.surfaces.Default(color=(0.2+rng.uniform(0,0.3), 0.4+rng.uniform(0,0.4), 0.2))
        )

def make_ground(scene):
    scene.add_entity(gs.morphs.Box(pos=(0,0,-0.25), size=(5,5,0.5), fixed=True),
                     surface=gs.surfaces.Default(color=(0.5,0.5,0.5)))

# ---------- Modal scene (Rasterizer; works on Metal) ----------
gs.init(backend=gs.metal)  # you’re on M3 Max per your log
scene = gs.Scene(
    show_viewer=False,
    vis_options=gs.options.VisOptions(
        show_world_frame=False,
        show_cameras=False,
        segmentation_level='entity'  # per-entity instance IDs
    ),
    renderer=gs.renderers.Rasterizer()  # <-- key: use Rasterizer, not RayTracer
)
make_ground(scene)
segments, seg_specs = make_toy_tree(scene)
sprinkle_leaves(scene)

cam = scene.add_camera(res=(1024, 768), pos=(2.4, 1.2, 1.7), lookat=(0.0, 0.0, 1.2), fov=40)
scene.build()

rgb, depth, seg_modal, _ = cam.render(rgb=True, depth=True, segmentation=True, normal=False)

# Save outputs
iio.imwrite(f"{OUT}/rgb/frame0.png", np.clip(rgb, 0, 255).astype(np.uint8))

# Save depth data - PNG doesn't support float32, so we provide multiple options:
# 1. Save exact floating point values as .npy
np.save(f"{OUT}/depth/frame0.npy", depth)

# 2. Convert to 16-bit PNG for visualization (scale to 0-65535 range)
depth_min, depth_max = depth.min(), depth.max()
depth_normalized = (depth - depth_min) / (depth_max - depth_min) * 65535
iio.imwrite(f"{OUT}/depth/frame0.png", depth_normalized.astype(np.uint16))

print(f"Depth range: {depth_min:.3f} to {depth_max:.3f}")
# map background -1 -> 0, shift entities +1
seg_png = (seg_modal.astype(np.int32) + 1).astype(np.uint16)
iio.imwrite(f"{OUT}/seg_modal/frame0.png", seg_png)

# ---------- Amodal pass per segment (new tiny scene each time) ----------
def render_amodal_mask(spec):
    amodal = gs.Scene(
        show_viewer=False,
        vis_options=gs.options.VisOptions(segmentation_level='entity', show_world_frame=False, show_cameras=False),
        renderer=gs.renderers.Rasterizer()
    )
    make_ground(amodal)
    amodal.add_entity(gs.morphs.Box(pos=spec["pos"], size=spec["size"]),
                      surface=gs.surfaces.Default(color=(0.4,0.3,0.2)))
    am_cam = amodal.add_camera(res=cam.res, pos=cam.pos, lookat=cam.lookat, fov=cam.fov)
    amodal.build()
    _, _, seg_amodal, _ = am_cam.render(segmentation=True)
    return (seg_amodal == 0)  # single entity -> ID 0

report = []
for i, spec in enumerate(seg_specs):
    amodal_mask = render_amodal_mask(spec)
    modal_visible = (seg_modal == i)
    occ_region = amodal_mask & (~modal_visible)
    A = int(amodal_mask.sum()); O = int(occ_region.sum())
    ratio = float(O) / float(A) if A > 0 else 0.0
    report.append({"segment": i, "amodal_area": A, "occluded_area": O, "occlusion_ratio": ratio})
    iio.imwrite(f"{OUT}/seg_amodal/segment_{i:03d}.png", (amodal_mask.astype(np.uint8))*255)

with open(f"{OUT}/occlusion_report.json", "w") as f:
    json.dump(report, f, indent=2)

print("Saved to:", os.path.abspath(OUT))

