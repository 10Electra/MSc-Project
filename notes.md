# Notes on the Project

## To do

- [ ] Implement a trilateral filter with projected normal distance, perpendicular radius, colour
- [ ] Apply mean-shift clustering
- [ ] Implement some triangulation

## Approach

- Currently building segmented meshes from superprimitives; an alternative approach would be to cookie-cut (convex hull type method) superprimitives from complete environment mesh?
- Density-adaptivity might not be so important in this case as the meshes come from images with uniform resolution
- We can use colour-based comparison instead
- Illumination change MLP for lighting-related colour changes
  
### SurfelMeshing / Point-based fusion

- Fusion process involves projecting surfels onto the image plane and comparing with new depth estimates
- (obvs already requires a pose estimate for the current depth image)

## Useful libraries

- pykdtree is possibly fastest for knn, but doesn't have fixed radius search