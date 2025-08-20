# Project Story

Thinking about how to present the story of my project and its outcomes in both writing and slides.

## Backbone

1. Goal - trying to map 3D surroundings from monocular RGB
2. Context I - SuperPrimitive VO; specific goal: merge SPs
3. Context II - existing mesh fusion methods; underexplored direction
4. Overview of my approach
5. Preliminary results
6. Limitations
7. Conclusions/future works

## Presentation Structure

1. Ideal for Mapping
2. SuperPrimitive and its strengths and weaknesses
3. This project's goal: mesh fusion
4. Ways to fuse meshes - related works - focus on point-shifting and Bayesian approaches
5. Approach and contributions
6. Pipeline
7. Results
8. Limitations and future works
9. Conclusions

## Suggested Report Structure

[Title page, Abstract, Acknowledgements, Contents pages]
1. Introduction
2. Literature review and related work
3. Body of report
4. Evaluation
5. Conclusions and future work
6.  References
7.  Declarations
8.  Appendices

## My Report Structure

1. Introduction
   1. Motivation
      - Importance of mapping
      - Importance of mesh
      - SuperPrimitive VO
      - Merging mesh: SuperPrimitive VO ---> SLAM
   3. Contributions
      - [Aim of this project]
      - Implemented such a system based on a paper
      - Developed/improved the point-shifting step
      - Built a Bayesian uncertainty framework around it
   4. Report structure overview
2. Background and related work
   1. Application context (more detail / reminder on SuperPrimitive and its shortcomings)
   2. Problem specification
   3. Review of existing solutions through the lens of the application
   4. Trilateral shift paper (check whether they've implemented weights)
   5. Modifications of trilateral shift paper for MoGe; proposed solutions
3. Implementation
   1. Early testing in 2D/3D?
   2. [Step through the final pipeline, explaining algorithms]
4. Results and evaluation
   1. Qualitative results on simulated data
   2. Evaluation of uncertainty awareness
   3. Comparison with TSDF fusion
   4. Robustness against noise and differing scales
   5. Tests on real data
5. Limitations and future work
6. Conclusions

## Background Report structure

Introduction
    Project Summary
    Motivation and Setting
Technical Background
    SuperPrimitive and its Context
    Geometry Representation and Estimation
    Explicit Representations
    Implicit Representations
Related Work
    Stanford zippering
    A probabilistic perspective
    Point-based approaches
    Other techniques of interest
Experimentation
    Vertex Grouping Approach
    Zippering approach
Project plan