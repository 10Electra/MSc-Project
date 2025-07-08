# Notes on the Project

## To do

- [ ] Implement a trilateral filter with projected normal distance, perpendicular radius, colour
- [ ] Apply mean-shift clustering
- [ ] Implement some triangulation

- [ ] Implement mesh.merge_close_vertices(eps)?

## Approach

- Currently building segmented meshes from superprimitives; an alternative approach would be to cookie-cut (convex hull type method) superprimitives from complete environment mesh?
- Density-adaptivity might not be so important in this case as the meshes come from images with uniform resolution
- We can use colour-based comparison instead
- Illumination change MLP for lighting-related colour changes

- Big debate - go for point-shifting or stitching? Compute budget sufficient for point-shifting?
  
### SurfelMeshing / Point-based fusion

- Fusion process involves projecting surfels onto the image plane and comparing with new depth estimates
- (obvs already requires a pose estimate for the current depth image)

## Useful libraries

- pykdtree is possibly fastest for knn, but doesn't have fixed radius search

## Implementation steps

1. Get local sampling spacing using 25-nn
2. Find neighbours within CYND defined by local sampling spacing
3. Find the overlapping regions of the pointcloud using CYND neighbourhoods
4. Use CYND to calculate normal shift for each point

## Mesh completion effort

### [BridgeShape: Latent Diffusion Schrödinger Bridge for 3D Shape Completion](https://arxiv.org/pdf/2506.23205v1)
- Very fast
- Requires fine-tuning of VQ-VAE

## Future directions of extension

### Semantic labelling?
- use a model such as yolo to identify objects and label them
- use the identification bounding boxes to aid segmentation?

### Integration of uncertainty somehow?
- model this in ithe fusion process to improve robustness etc
- can run MoGE multiple times and estimate per-pixel uncertainty (https://arxiv.org/abs/1506.02142)
- can back-project MoGE points into neighbouring frames and measure the reprojection error

### Integration of learning techniques
- e.g. self-supervised mesh refinement
- GNNs to improve/guide explicit mesh fusion

### Multi-scale / hierarchical fusion