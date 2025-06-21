import numpy as np
import matplotlib.pyplot as plt

def get_integer_segments(sp_regions):
    integer_segments = np.zeros([sp_regions.shape[1], sp_regions.shape[2]])
    for i in range(sp_regions.shape[0]):
        integer_segments[sp_regions[i,:,:]] = i + 1
    return integer_segments

def plot_region_numbers(integer_segments:np.ndarray,text_size=8):
    plt.figure(figsize=(10, 8))
    plt.imshow(integer_segments, cmap='nipy_spectral')

    # Calculate center of mass for each region and add text labels
    for region_id in range(1, int(np.max(integer_segments)) + 1):
        region_mask = (integer_segments == region_id)
        
        # Calculate center of mass
        if np.any(region_mask):  # Check if region exists
            y_coords, x_coords = np.where(region_mask)
            com_y = float(np.mean(y_coords))
            com_x = float(np.mean(x_coords))
            
            # Add text at center of mass
            plt.text(com_x, com_y, str(region_id), 
                    color='white', fontweight='bold', fontsize=text_size,
                    ha='center', va='center',
                    bbox=dict(boxstyle='round,pad=0.2', facecolor='black', alpha=0.7))

    plt.colorbar(label='Region ID')
    plt.title('Segmented Regions with Labels')
    plt.axis('off')
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
    H, W = 240, 320
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