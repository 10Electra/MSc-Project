import torch
from torch import nn

# from frontend.segment.mask_generation import invalid_borders_mask

def normalise_coordinates(x_pixel, dims):
    inv = 1.0 / (torch.as_tensor(dims, dtype=torch.float32, device=x_pixel.device) - 1)

    x_norm = 2 * x_pixel * inv - 1
    return x_norm

def denormalise_coordinates(x_norm, dims, return_int=True):
    dims = torch.as_tensor(dims, dtype=torch.float32, device=x_norm.device)
    x_pixel = 0.5 * (dims - 1) * ((x_norm) + 1 )

    
    if return_int:
        return x_pixel.round().long()
    else:
        return x_pixel


def invalid_borders_mask(H, W, device='cuda:0'):
    y, x = torch.meshgrid(torch.arange(H, device=device), 
                          torch.arange(W, device=device), indexing='ij')
    coords = torch.stack((y, x), dim=-1).reshape(-1, 2)

    grid = normalise_coordinates(coords, (H, W))
    valid_grid = grid.abs().max(dim=-1).values < 0.99  

    return valid_grid.reshape(H, W)


def spatial_size(pointmap):
    # pointmap is [..., 3]
    assert(pointmap.shape[-1] == 3)
    return pointmap.shape[:-1]

def create_opt_params(kf, deformable, dtype=torch.float32, pose_dtype=torch.float32):
    global_logdepth_init = 0.0
    # _, H, W = kf.image.shape
    device = kf.image.device
    
    num_segments = kf.num_segments()

    global_logscale =  torch.tensor([global_logdepth_init], device=device, dtype=dtype)
    segment_logscales =  torch.zeros(num_segments, device=device, dtype=dtype)

 
    kf_opts = {'sp_logscales': segment_logscales,
               'global_logscale': global_logscale}

    kf_deforms = None
    if deformable:
        kf_deforms = torch.eye(4, device=device,
                               dtype=pose_dtype).unsqueeze(0).expand(num_segments,
                                                                                          -1, -1)
        
    return kf_opts, kf_deforms  


def infer_raymap(pointmap, eps=1e-7):
    # pointmaps is [..., 3]
    assert(pointmap.shape[-1] == 3)
    raydist = torch.norm(pointmap, dim=-1, keepdim=True)

    raymap = pointmap / (raydist + eps)

    return raymap, raydist.squeeze(-1)

class PointmapKF(nn.Module):
    def __init__(self, 
                 image, 
                 pointmap, 
                 sp_regions, 
                 region_corr=None,
                 keypoints=None,
                 K=None,
                 original_image=None,
                 **kwargs):
        super().__init__()

        with torch.no_grad():
            raymap, raydist = infer_raymap(pointmap)

        self.register_buffer('image_raw', image)
        self.register_buffer('raymap', raymap)
        self.register_buffer('raydist', raydist)
        self.register_buffer('log_raydist', torch.log(raydist))
        self.register_buffer('pointmap', pointmap)
        self.register_buffer('sp_regions', sp_regions)

        if K is not None:
            self.register_buffer('K', K)
        if keypoints is not None:
            self.register_buffer('keypoints', keypoints)
        if region_corr is None:
            self.register_buffer('region_corr', region_corr)

        if original_image is not None:
            self.register_buffer('original_image', original_image)

        self._remove_empty_segments()
        # self._original_image = original_image

    @property
    def original_image(self):
        if self._original_image is not None:
            return self._original_image
        return self.image_raw
    
    @property
    def image(self):
        if self.image_raw.shape[0] > 3:
            return self.image_raw[:3]
        return self.image_raw

    def has_normals(self):
        return self.image_raw.shape[0] > 3
    
    @property
    def image_normals(self):
        if self.image_raw.shape[0] > 3:
            return self.image_raw[3:]
        return None
    
    @property
    def normals(self):
        if self.image_raw.shape[0] > 3:
            return self.image_raw[3:]
        return None
    
    def num_segments(self):
        return self.sp_regions.shape[0]
    
    def spatial_size(self):
        return spatial_size(self.raymap)


    def _remove_empty_segments(self):
        H, W = self.spatial_size()
        valid_grid = invalid_borders_mask(H, W, device=self.sp_regions.device)
        masks_padd = (self.sp_regions & valid_grid[None, ...])

        num_valid_pix = (masks_padd).sum(dim=(1, 2))
        good_masks = num_valid_pix > 0

        self.sp_regions = self.sp_regions[good_masks]
        return 
