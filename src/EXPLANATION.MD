# DMR on the GPU (using CUDA)

We follows Chew's refinement algorithm in the form used by the Galois / LonestarGPU DMR benchmark.

## Reference

R. Nasre, M. Burtscher, K. Pingali. *Morph Algorithms on GPUs.* PPoPP 2013.
GPU code of the original benchmark: https://github.com/IntelligentSoftwareSystems/Galois,
`lonestar/scientific/gpu/delaunayrefinement/`.

## Algorithm

A triangle is bad if its smallest angle is below `MIN_ANGLE` (30°). Mesh-boundary edges (`nbr == -1`) are the segments. One round:

1. **`find_bad`**: each thread tests one triangle.
2. **`build_cavities`**: for each bad triangle, find the point to insert and the triangles it replaces.
   - If the triangle is obtuse, walk across the edge opposite the obtuse corner until reaching a non-obtuse triangle, so the circumcenter `c` lies inside the mesh.
   - The cavity is all triangles whose circumcircle contains `c` (`dist² <= r²`), grown by BFS.
   - If `c` encroaches on a segment (lies in its diametral circle), split that segment at its midpoint instead.
   - The cavity is kept only if every boundary edge `(a, b)` has `orient(a, b, c) > 0` and it has at most `MAX_CAVITY` triangles.
3. **`claim`**: overlapping cavities conflict, so each triangle and each outside neighbour, goes to the lowest cavity id (`atomicMin`).
4. **`retriangulate`**: a cavity that won all its triangles is replaced by a fan of triangles from the new point to its boundary edges.

`refine()` repeats rounds until no bad triangle is left, a round inserts nothing  or `MAX_ROUNDS` is reached.
