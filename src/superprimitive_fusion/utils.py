import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import open3d as o3d  # type: ignore
from trimesh import Trimesh
from mpl_toolkits.axes_grid1 import make_axes_locatable # type: ignore


def get_integer_segments(sp_regions):
    integer_segments = np.zeros([sp_regions.shape[1], sp_regions.shape[2]])
    for i in range(sp_regions.shape[0]):
        integer_segments[sp_regions[i,:,:]] = i + 1
    return integer_segments


def plot_region_numbers(integer_segments: np.ndarray,
                        text_size: int = 8,
                        figsize: tuple = (10, 8),
                        title: str = 'Labelled Segments'):
    
    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(integer_segments, cmap='nipy_spectral')

    n_regions = int(np.max(integer_segments))
    for region_id in range(1, n_regions + 1):
        region_mask = (integer_segments == region_id)
        if np.any(region_mask):
            y, x = np.where(region_mask)
            ax.text(x.mean(), y.mean(), str(region_id),
                    color='white', fontweight='bold', fontsize=text_size,
                    ha='center', va='center',
                    bbox=dict(boxstyle='round,pad=0.2',
                              facecolor='black', alpha=0.7))

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = fig.colorbar(im, cax=cax)
    cbar.set_label('Region ID')

    ax.set_title(title)
    ax.axis('off')
    plt.show()


def quadto2tris(idx: tuple, verts: np.ndarray) -> tuple[list[int], list[int]]:
    """Decides which two triangles to construct from the quadrilateral provided"""
    
    diag1_len_sq = np.sum((verts[0] - verts[2])**2)
    diag2_len_sq = np.sum((verts[1] - verts[3])**2)
    
    if diag1_len_sq <= diag2_len_sq:
        tri1 = [idx[0], idx[2], idx[1]]
        tri2 = [idx[0], idx[3], idx[2]]
    else:
        tri1 = [idx[0], idx[3], idx[1]]
        tri2 = [idx[1], idx[3], idx[2]]
    
    return (tri1, tri2)


def triangulate_segments(verts, integer_segments):
    tris = [[] for _ in range(int(np.max(integer_segments))+1)]
    H, W = integer_segments.shape[0], integer_segments.shape[1]
    for v in range(H - 1):
        for u in range(W - 1):
            id_tl = v * W + u
            id_tr = id_tl + 1
            id_bl = id_tl + W
            id_br = id_bl + 1

            quad_idx = (id_bl, id_tl, id_tr, id_br)
            quad_verts = np.array([verts[id] for id in quad_idx])

            sps = [int(integer_segments.flatten()[id]) for id in quad_idx]
            sp_counts = {}
            for sp in sps:
                sp_counts[sp] = 1 + sp_counts.get(sp, 0)

            if len(sp_counts) == 1 and sps[0] != 0: # All verts from one sp
                tri1, tri2 = quadto2tris(quad_idx, quad_verts)
                tris[sps[0]-1].extend([tri1, tri2])
            elif len(sp_counts) == 2 and len(set(sp_counts.values())) == 2: # Three verts from one sp, one from another
                aclock_quad_idx = list(quad_idx)
                for i in range(4):
                    if sp_counts[sps[i]] == 1:
                        del aclock_quad_idx[i]
                sample_sp = int(integer_segments.flatten()[aclock_quad_idx[0]])
                if sample_sp == 0:
                    continue # Skip this triangle if it is not in a superprimitive
                aclock_quad_idx.reverse()
                tris[sample_sp-1].append(aclock_quad_idx)
    return tris


def o3d_to_trimesh(mesh: o3d.geometry.TriangleMesh, copy: bool = False) -> Trimesh:
    """Convert Open3D TriangleMesh -> Trimesh, preserving vertex colours.

    Parameters
    ----------
    mesh : o3d.geometry.TriangleMesh
        The source mesh.
    copy : bool, default False
        If True, numpy arrays are deep-copied; otherwise views are used.

    Returns
    -------
    Trimesh
        A Trimesh with vertices, faces and (if present) vertex colours.
    """
    verts = np.asarray(mesh.vertices)
    faces = np.asarray(mesh.triangles)

    if copy:
        verts = verts.copy()
        faces = faces.copy()

    tm = Trimesh(vertices=verts, faces=faces, process=False)

    # Copy across vertex colours
    if mesh.has_vertex_colors():
        # Open3D stores floats in [0,1]; Trimesh expects uint8 in [0,255]
        cols = np.asarray(mesh.vertex_colors)
        if copy:
            cols = cols.copy()

        if cols.max() <= 1.0:
            cols = (cols * 255).astype(np.uint8)

        # Ensure Trimesh's 4-channel RGBA
        if cols.shape[1] == 3:
            alpha = np.full((cols.shape[0], 1), 255, dtype=np.uint8)
            cols = np.hstack([cols, alpha])

        tm.visual.vertex_colors = cols

    return tm


def trimesh_to_o3d(mesh: Trimesh, copy: bool = False) -> o3d.geometry.TriangleMesh:
    """Convert Trimesh -> Open3D TriangleMesh, preserving vertex colours.

    Parameters
    ----------
    mesh : Trimesh
        The source mesh.
    copy : bool, default False
        If True, numpy arrays are deep-copied before passing to Open3D.

    Returns
    -------
    o3d.geometry.TriangleMesh
        An Open3D mesh with vertices, faces and (if present) vertex colours.
    """
    verts = mesh.vertices.copy() if copy else mesh.vertices
    faces = mesh.faces.copy()     if copy else mesh.faces

    o3d_mesh = o3d.geometry.TriangleMesh()
    o3d_mesh.vertices  = o3d.utility.Vector3dVector(verts)
    o3d_mesh.triangles = o3d.utility.Vector3iVector(faces)

    # Copy across vertex colours
    if mesh.visual.kind == 'vertex' and mesh.visual.vertex_colors.size:
        cols = mesh.visual.vertex_colors
        if copy:
            cols = cols.copy()

        # Trimesh stores uint8 [0,255]; Open3D wants float [0,1] and RGB only
        if cols.dtype == np.uint8:
            cols = cols.astype(np.float64) / 255.0
        if cols.shape[1] == 4:        # drop alpha channel for Open3D
            cols = cols[:, :3]

        o3d_mesh.vertex_colors = o3d.utility.Vector3dVector(cols)

    return o3d_mesh


def get_o3d_colours_from_trimesh(colours:np.ndarray) -> o3d.utility.Vector3dVector:
    if colours.shape[1] == 4:
        no_alpha = colours[:,:3]
    else:
        no_alpha = colours
    
    return o3d.utility.Vector3dVector((no_alpha / 255).astype('float64'))


def smooth_mask(
    mask: np.ndarray,
    radius_erode: int = 2,
    radius_dilate: int | None = 2,
) -> np.ndarray:
    """Smooths a binary mask by conservative morphological erosion.

    Args:
        mask (np.ndarray): Boolean or 0/1 binary array where `True`/1 denotes
            the foreground region to be smoothed.
        radius_erode (int): Radius (in pixels) of the circular structuring
            element used for erosion. Larger values remove larger “teeth” and
            thin the mask more.
        radius_dilate (int | None): If an integer, the eroded mask is dilated
            with this (usually smaller) radius and then intersected with the
            original mask to regain a little thickness while **guaranteeing**
            that no new foreground pixels are added. If ``None`` (default),
            only erosion is applied.

    Returns:
        np.ndarray: A boolean array of the same shape as ``mask``, containing
            the smoothed (and still conservative) mask.
    """
    if mask.dtype != np.bool_ and mask.dtype != np.uint8:
        raise TypeError("`mask` must be of dtype bool or uint8/0-1")

    # Ensure uint8 for OpenCV, but keep a bool copy for the final AND.
    m_uint8: np.ndarray = mask.astype(np.uint8, copy=False)

    # --- build circular structuring element --------------------------------
    def _disk(r: int) -> np.ndarray:
        """Return a (2r+1)×(2r+1) uint8 disk for cv::morphologyEx kernels."""
        if r < 1:
            raise ValueError("radius must be ≥ 1")
        return cv.getStructuringElement(cv.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))

    # --- 1. Erosion ---------------------------------------------------------
    eroded = cv.erode(m_uint8, _disk(radius_erode))

    if radius_dilate is None:
        return eroded.astype(bool, copy=False)

    # --- 2. Optional dilation, then clip to original ------------------------
    reopened = cv.dilate(eroded, _disk(radius_dilate))
    conservative = np.logical_and(reopened.astype(bool, copy=False), mask.astype(bool, copy=False))

    return conservative


def fill_ring_holes(
    mask: np.ndarray,
    radius: int = 3,
    max_iterations: int = 10,
) -> np.ndarray:
    """Fills background pixels whose surrounding ring is entirely foreground.

    Args:
        mask (np.ndarray): Binary image where `True` (or 1) denotes the
            foreground region.
        radius (int): Radius (in pixels) of the hollow circle that must be
            completely inside the foreground for the centre pixel to be filled.
        max_iterations (int): The operation is applied iteratively because
            filling one pixel may enable its neighbour on the next pass.  Set
            this to a small number (e.g. 10) or -1 for “until convergence”.

    Returns:
        np.ndarray: A boolean array where all pixels that satisfied the ring
            condition (possibly over several passes) have been converted to
            foreground.
    """
    if radius < 1:
        raise ValueError("'radius' must be >= 1")
    if mask.dtype != np.bool_ and mask.dtype != np.uint8:
        raise TypeError("'mask' must be of dtype bool or uint8/0-1")

    # Build the hit-or-miss kernel
    side = 2 * radius + 1
    kernel = np.zeros((side, side), dtype=np.int8)

    yx = np.indices(kernel.shape) - radius
    dist = np.hypot(yx[0], yx[1])
    kernel[(dist >= radius - 0.5) & (dist <= radius + 0.5)] = 1   # ring = "should be 1"
    kernel[radius, radius] = -1                                   # centre = "should be 0"
    # 0  "don't care"

    # Iteratively apply hit-or-miss until no more pixels are filled
    work = mask.astype(bool, copy=True)
    it = 0
    while True:
        hit = cv.morphologyEx(work.astype(np.uint8), cv.MORPH_HITMISS, kernel)
        if not hit.any() or (max_iterations != -1 and it >= max_iterations):
            break
        work |= hit.astype(bool)
        it += 1

    return work


def crop_by_SP(segment_id:int,
               integer_segments:np.ndarray,
               image:np.ndarray, points:np.ndarray,
               border=0,
               make_binary=True) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Crops the segmentation, colour and, 3D point arrays to the bounding rectangle around a particular segment.

    Args:
        segment_id (int): the integer related to the superprimitive to be cropped around
        image (np.ndarray): original colour image to be correspondingly cropped
        integer_segments (np.ndarray): segmentation masks in integer format (each segment made of a different integer)
        points (np.ndarray): 3D points in (H,W,3) format

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray]: cropped segmentation, image, and points arrays
    """
    if integer_segments.ndim != 2:
        raise ValueError("integer_segments must be 2-D (H, W).")
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("image must be shape (H, W, 3)")
    if points.ndim != 3 or points.shape[2] != 3:
        raise ValueError("points must be shape (H, W, 3)")
    if points.shape != image.shape:
        raise ValueError("points and image shapes must match")
    if points.shape[:2] != integer_segments.shape:
        raise ValueError("integer_segments shape must match image and points shapes")

    temp_segs = integer_segments.copy()
    
    # Locate all pixels in the desired segment
    mask = temp_segs == segment_id
    if not mask.any():
        raise ValueError(f"segment_id {segment_id} not found in integer_segments.")
    
    if make_binary:
        temp_segs[~mask] = 0
        temp_segs[mask] = 1
        temp_segs = temp_segs.astype(bool)
    
    # Bounding rows and columns
    rows = np.where(mask.any(axis=1))[0]
    cols = np.where(mask.any(axis=0))[0]
    r_min, r_max = rows[0], rows[-1]
    c_min, c_max = cols[0], cols[-1]

    # Slice both arrays
    seg_crop = temp_segs[r_min - border : r_max + border + 1, c_min - border : c_max + border + 1]
    img_crop =     image[r_min - border : r_max + border + 1, c_min - border : c_max + border + 1]
    pts_crop =    points[r_min - border : r_max + border + 1, c_min - border : c_max + border + 1, :]

    return seg_crop, img_crop, pts_crop


def show_anns(anns, borders=True):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)

    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    img[:, :, 3] = 0
    for ann in sorted_anns:
        m = ann['segmentation']
        color_mask = np.concatenate([np.random.random(3), [0.5]])
        img[m] = color_mask 
        if borders:
            import cv2
            contours, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) 
            # Try to smooth contours
            contours = [cv2.approxPolyDP(contour, epsilon=0.01, closed=True) for contour in contours]
            cv2.drawContours(img, contours, -1, (0, 0, 1, 0.4), thickness=1) 

    ax.imshow(img)