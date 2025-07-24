import numpy as np
import open3d as o3d
from superprimitive_fusion.scanner import virtual_rgbd_scan
from superprimitive_fusion.utils import bake_uv_to_vertex_colours, polar2cartesian
from superprimitive_fusion.mesh_fusion import fuse_meshes

mesh = o3d.io.read_triangle_mesh("data/mustard-bottle/textured.obj", enable_post_processing=True)

bake_uv_to_vertex_colours(mesh)

mesh.compute_vertex_normals()

bb = mesh.get_minimal_oriented_bounding_box()
scale = np.mean(bb.get_max_bound())
scans = []
for i in range(6):
    mesh_scan, pcd_scan = virtual_rgbd_scan(
        mesh,
        cam_centre=polar2cartesian(r=0.3, lat=60, long=0+60*i),
        look_dir=(0, 0, 0),
        width_px=180,
        height_px=120,
        fov=70,
        dropout_rate=0,
        depth_error_std=0.003*scale,
        translation_error_std=0,#0.02*scale,
        rotation_error_std_degs=0,
        dist_thresh=0.25*scale,
    )
    scans.append(mesh_scan)

j = 2
fused_mesh = scans[0]
for i in range(1,j):
    fused_mesh = fuse_meshes(
        fused_mesh,
        scans[i],
        h_alpha=5,
        trilat_iters=2,
        shift_all=False,
        fill_holes=False,
    )

o3d.visualization.draw_geometries(
    [fused_mesh],
    window_name="Virtual scan",
    front=[0.3, 1, 0],
    lookat=[0, 0, 0],
    up=[0, 0, 1],
    zoom=0.7,
)