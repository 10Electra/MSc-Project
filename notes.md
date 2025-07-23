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
...

## Final implementation goals
- Implement a colour comparison in the trilateral filter
  - Possibly only comparing Hue in HSV?

- Averaging: weighting all the averages by 'age'

- Fast nearest neighbour search using pixel correspondences

- Adaptive merging
  - Reduce vertex density where not needed based on curvature
  - Use before or after fusion?
  - Use only for the fusion zone?

- Experiment with more complex objects
  - Scanner which trims more intelligently
  - Fusion that works with 'islands'

- Make faster
  - Tensor-based approach?
  - Stitching approach?

- ~~Retexturing post fusion~~
  - Wouldn't work on real data

## Overall algorithm

1. Extract attributes from input mesh (points, colour, ...)
2. Compute and smooth vertex normals
3. Calculate local spacing and density throughout the meshes
4. Precompute neighbours within cylindrical neighbourhoods
5. Find overlapping regions
6. Shift overlapping vertices along individual normal directions
7. Merge nearby clusters in overlapping region
8. Book-keeping: create a mapping between old vertices and new merged vertices (and their attributes)
9. Triangulate the overlap region
10. Trim the overlap mesh to the overlap boundary
11. Concatenate the overlap mesh with the retained non-overlapping meshes
12. Fill holes and clean mesh

## Implementation efforts

### Tried to use the original triangulations in the overlap region
- Would have been nice to ...

### Tried many ways to retriangulate the overlap region
- Many ways to project the pointcloud
  - Projecting onto the new mesh's camera view
    - The overlap region is larger than the overlapping region of the new mesh because of the safety margin
    - The new areas of the overlap region are not constrained to be visible from the new mesh's camera pose (they might fold during projection, causing spurious connections in 3d)
  - Projecting the overlap region adaptively
    - Many methods including local tangent planes triangulation, local PCA, laplacian magic didn't produce a representative mapping
    - My own method involving imagining an ant crawling between points and projecting them locally relative to the starting point worked for some regions but not well overall
  - Meshing directly
    - Ball Pivoting Algorithm (best so far; has holes)
    - Poisson (uses different vertices, thus hard to concat with old mesh)
    - Alpha shapes (hard to choose a robust alpha)
    - Investigated using 