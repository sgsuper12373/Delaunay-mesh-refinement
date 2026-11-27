# Delaunay Mesh Refinement on the GPU

GPU Programming course project: CUDA refinement of a 2D Delaunay mesh until no angle is below 30°. See [src/EXPLANATION.MD](src/EXPLANATION.MD) for summary on how it works.

## How to run

```sh
cd src
make ARCH=sm_75     # your GPU's architecture: sm_75 for a T4, sm_80 (the default) for an A100
./dmr 50 50 4 1     # grid of nx x ny cells, each w x h; writes before.obj and after.obj
```

CPU baseline (Triangle): `cd baselines && make test`.

## Current caveats

- The GPU code in `src/` assumes its input is already a Delaunay triangulation (for now, the grid that `main.cu` builds). It refines that mesh; it does not triangulate points itself.
- At 30°, refinement is not proven to terminate.
- Refinement stops after `MAX_ROUNDS` (1000, set in `src/kernels.cuh`), even if bad triangles remain. Large inputs may not converge in time; the default `./dmr` (200 x 200) reaches the limit with a few bad triangles left. Raise `MAX_ROUNDS` for such inputs.
