import open3d as o3d  # type: ignore
import numpy as np

###########################################################
# Geometry helpers                                        #
###########################################################

def _quad_to_tris(idx: tuple[int, int, int, int],
                  verts: np.ndarray) -> tuple[list[int], list[int]]:
    """Split a quadrilateral into two triangles.

    The shortest diagonal is used for convex quads, while concave quads are
    triangulated so both resulting triangles are oriented counter‑clockwise
    (positive signed area).
    """
    def signed_area(a, b, c):
        return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])

    # Convexity check – all cross products have the same sign
    crosses: list[float] = []
    for i in range(4):
        a, b, c = verts[i], verts[(i + 1) % 4], verts[(i + 2) % 4]
        ab, bc = b - a, c - b
        crosses.append(ab[0] * bc[1] - ab[1] * bc[0])
    is_convex = all(cp > 0 for cp in crosses)

    if is_convex:
        # pick the shorter diagonal
        d0 = np.sum((verts[0] - verts[2]) ** 2)
        d1 = np.sum((verts[1] - verts[3]) ** 2)
        if d0 <= d1:
            return [idx[0], idx[1], idx[2]], [idx[0], idx[2], idx[3]]
        return [idx[0], idx[1], idx[3]], [idx[1], idx[2], idx[3]]

    # Concave: test both diagonals for valid (positive‑area) triangles
    sa = [signed_area(*verts[[0, 1, 2]]), signed_area(*verts[[0, 2, 3]]),
          signed_area(*verts[[0, 1, 3]]), signed_area(*verts[[1, 2, 3]])]
    split1_ok = sa[0] > 0 and sa[1] > 0  # (0,1,2) & (0,2,3)
    split2_ok = sa[2] > 0 and sa[3] > 0  # (0,1,3) & (1,2,3)
    if split1_ok or not split2_ok:
        return [idx[0], idx[1], idx[2]], [idx[0], idx[2], idx[3]]
    return [idx[0], idx[1], idx[3]], [idx[1], idx[2], idx[3]]

###########################################################
# Colour helpers                                          #
###########################################################

def _interpolate_vertex_colors(mesh: o3d.geometry.TriangleMesh,
                               primitive_ids: np.ndarray,
                               bary_uv: np.ndarray) -> np.ndarray:
    """Interpolate per‑vertex colours of *mesh* at the hit points."""
    vcols = np.asarray(mesh.vertex_colors)
    tris = np.asarray(mesh.triangles, dtype=np.int32)

    tri = tris[primitive_ids]
    c0, c1, c2 = vcols[tri[:, :, 0]], vcols[tri[:, :, 1]], vcols[tri[:, :, 2]]

    u, v = bary_uv[:, :, 0], bary_uv[:, :, 1]
    w = 1.0 - u - v
    vcols = w[:, :, None] * c0 + u[:, :, None] * c1 + v[:, :, None] * c2

    return vcols

###########################################################
# Main entry point                                        #
###########################################################

def virtual_rgbd_scan(
    mesh: o3d.geometry.TriangleMesh,
    cam_centre=(80.0, 80.0, 0.0),
    look_dir=(0.0, 0.0, 0.0),
    *,
    dropout_rate: float = 0.10,
    depth_error_std: float = 0.0,
    translation_error_std: float = 0.0,
    rotation_error_std_degs: float = 0.0,
    width_px: int = 640,
    height_px: int = 480,
    fov: float = 90.0,
    dist_thresh: float = 10.0,
):
    """Generate a *virtual* depth scan of *mesh* and return a coloured mesh."""
    ###############
    # Ray casting #
    ###############
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))

    rays = o3d.t.geometry.RaycastingScene.create_rays_pinhole(
        fov_deg=fov,
        center=list(look_dir),
        eye=list(cam_centre),
        up=[0, 0, 1],
        width_px=width_px,
        height_px=height_px,
    )

    ans = scene.cast_rays(rays)
    t_hit = ans["t_hit"].numpy()

    # Intersection metadata (triangle id + barycentric uv)
    prim_ids = ans["primitive_ids"].numpy().astype(np.int32)
    bary_uv = ans.get("primitive_uvs", None).numpy()

    ######################
    # Dropout / validity #
    ######################
    valid = np.isfinite(t_hit).reshape(-1)
    # random dropout mask
    n_dropout = int(dropout_rate * valid.size)
    if n_dropout:
        dropout_idx = np.random.choice(valid.size, n_dropout, replace=False)
        valid[dropout_idx] = False

    ########################
    # Generate 3‑D vertices#
    ########################
    rays_np = rays.numpy()  # (H*W,6)
    origins = rays_np[..., :3]
    dirs = rays_np[..., 3:]
    noise = (depth_error_std * np.random.randn(*t_hit.shape)).astype(np.float32)
    t_noisy = t_hit + noise
    verts = origins + dirs * t_noisy[..., None]
    verts = verts.reshape(-1, 3)

    ######################################
    # Optional colour interpolation step #
    ######################################
    vcols: np.ndarray | None = None
    if mesh.has_vertex_colors():
        if bary_uv is not None:
            vcols = _interpolate_vertex_colors(mesh, prim_ids, bary_uv)
        else:  # fall back to the first vertex of the hit triangle
            source_cols = np.asarray(mesh.vertex_colors)
            tri_first = np.asarray(mesh.triangles, dtype=np.int32)[prim_ids, 0]
            vcols = source_cols[tri_first]
        # invalid hits – set to black
        assert vcols is not None  # narrow type for mypy
        vcols.reshape(-1,3)[~valid] = np.full(3, 0.0)

    ########################
    # Re‑construct surface #
    ########################
    H, W = height_px, width_px
    tris: list[list] = []
    for v in range(H - 1):
        for u in range(W - 1):
            tl = v * W + u
            tr = tl + 1
            bl = tl + W
            br = bl + 1
            quad = (bl, tl, tr, br)
            if not (valid[quad[0]] and valid[quad[1]] and valid[quad[2]] and valid[quad[3]]):
                continue

            quad_verts = verts[list(quad)]
            
            # large edge ‑> depth discontinuity
            if any(np.linalg.norm(quad_verts[i] - quad_verts[(i + 1) % 4]) > dist_thresh for i in range(4)):
                continue

            tris.extend(_quad_to_tris(quad, quad_verts))

    mesh_out = o3d.geometry.TriangleMesh()
    mesh_out.vertices = o3d.utility.Vector3dVector(verts)
    mesh_out.triangles = o3d.utility.Vector3iVector(np.asarray(tris, dtype=np.int32)[:, [0,2,1]])

    if vcols is not None:
        mesh_out.vertex_colors = o3d.utility.Vector3dVector(vcols.reshape(-1,3))

    #############################
    # Clean‑up / post‑processing#
    #############################
    mesh_out.remove_unreferenced_vertices()
    mesh_out.remove_degenerate_triangles()
    mesh_out.remove_duplicated_triangles()
    mesh_out.remove_non_manifold_edges()
    mesh_out.compute_vertex_normals()

    ######################
    # Registration error #
    ######################
    mesh_out.translate(tuple(np.random.randn(3) * translation_error_std))
    R = mesh.get_rotation_matrix_from_xyz(
        tuple(np.random.randn(3) * np.deg2rad(rotation_error_std_degs))
    )
    mesh_out.rotate(R, center=mesh_out.get_center())

    #######################
    # Noised point cloud  #
    #######################
    pcd = o3d.t.geometry.PointCloud(o3d.core.Tensor(verts, dtype=o3d.core.Dtype.Float32))
    pcd = pcd.translate(tuple(np.random.randn(3) * translation_error_std))
    R = mesh.get_rotation_matrix_from_xyz(
        tuple(np.random.randn(3) * np.deg2rad(rotation_error_std_degs))
    )
    pcd = pcd.rotate(R, center=pcd.get_center())

    return mesh_out, pcd


if __name__ == '__main__':
    import open3d as o3d
    from superprimitive_fusion.utils import bake_uv_to_vertex_colours, polar2cartesian

    mesh = o3d.io.read_triangle_mesh("data/mustard-bottle/textured.obj", enable_post_processing=True)

    bake_uv_to_vertex_colours(mesh)

    mesh.compute_vertex_normals()

    mesh_scan, pcd_scan = virtual_rgbd_scan(
        mesh,
        cam_centre=polar2cartesian(r=0.3, lat=90, long=90),
        look_dir=(0, 0, 0),
        width_px=360,
        height_px=240,
        fov=70,
        dropout_rate=0,
        depth_error_std=0.0001,
        dist_thresh=10,
    )
    
    o3d.visualization.draw_geometries([mesh_scan], window_name="Virtual scan (mesh)")