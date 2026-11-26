# Baseline Notes (chronological)

- Baseline choice
  - CPU: Triangle (Shewchuk), handles constrained Delaunay + Ruppert refinement, reads `.poly`
  - GPU: gDel2D for triangulation, plus a naive CUDA refinement kernel (target A100, `sm_80`) - not set up yet
- Input format (`data/inputs/input/`, files named `.txt` but are **binary**)
  - `*_vertex.txt`: int32 `n`, then `n` x (float32 x, float32 y)
  - `*_constraint.txt`: int32 `m`, then `m` x (int32 a, int32 b), 0-based vertex ids
- Source downloaded from `https://www.netlib.org/voronoi/triangle.zip` into `baselines/triangle/`
- GCC 16 defaults to C23 and rejects Triangle's K&R C -> build with `-std=gnu89`
- Triangle `.poly` needs a holes count (`0`) after the segments, otherwise "Unexpected end of file"
- `bin2poly.c` converts the binary dataset files to `.poly`
- Layout
  - `bin/` binaries, `tests/` test files (`A.poly`), `output/` generated meshes, `triangle/` source, `docs/` notes
  - `bin/` and `output/` are git-ignored
- Commands (run from `baselines/`)
  - Build + sanity test: `make test`
  - Run a dataset: `./run_cpu_baseline.sh 1.2M [min_angle=20]`
  - Timing log: `../logs/cpu_baseline.log`
  - Consistency check: `bin/triangle -pq20 -C output/1.2M.poly`
  - Angle histogram / stats: `bin/triangle -pq20 -V output/1.2M.poly`
  - Clean: `make clean`
- Result, 1.2M dataset, `-pq20`
  - 7.4-7.9 s, ~320 MB RSS
  - 1,177,332 input vertices -> 1,035,154 mesh vertices, 1,621,101 triangles
  - Smallest angle 2.76 deg, only 18 triangles below 20 deg (expected at sharp constraint corners)
- Viewing results
  - `showme.c` (X11) only suits small meshes; build with `gcc -O2 -std=gnu89 -w -o bin/showme triangle/showme.c -lX11`
  - For big meshes use a Python script (see `todo.md`)
- Git
  - Local `main` diverged from `origin/main` (PR #1 added `src/mesh.cuh`); resolved with a merge, no file conflicts
  - `baselines/Makefile`, `bin2poly.c`, `run_cpu_baseline.sh`, `docs/` are not committed yet





# PSLG Format
- <n>: Number of vertices.
- 2: Dimension (always 2D).
- 0: Number of per-vertex attributes 
- 0: Boundary markers (0 = omitted, 1 = included).
- 
## section 1 
  <n> 2 0 0
  0 <x0> <y0>
  1 <x1> <y1>
  ...


- <m>: Number of constraint segments.
- 0: Boundary markers (0 = omitted).
- Body lines (<id> <a> <b>): Segment index (0 to m-1) followed by the two endpoint vertex indices a and b.

## section 2 
  <m> 0
  0 <a0> <b0>
  1 <a1> <b1>
  ...

- hole list 

## section 3