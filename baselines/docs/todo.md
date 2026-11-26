# Baseline TODO

## CPU baseline (Triangle)
- [x] Download Triangle source into `baselines/triangle/`
- [x] Binary dataset -> `.poly` converter (`bin2poly.c`)
- [x] Makefile (binaries in `bin/`) and `run_cpu_baseline.sh`
- [x] Verify on `A.poly` and the 1.2M dataset (`-C` consistency check, `-V` angle histogram)
- [ ] Run the remaining datasets (3.2M, 4.3M, 5.6M, 8.4M, 9.4M) and record timings
- [ ] Add optional `make showme` rule (needs X11 headers; keep out of `make all`)
- [ ] Write Python visualizer `scripts/plot_mesh.py`
  - [ ] Read `.node` / `.ele` (Triangle output)
  - [ ] Zoomed-window mesh plot saved as PNG (no display needed, works on HPC)
  - [ ] Minimum-angle histogram
- [ ] Write Python checker: triangle count, min angle, Delaunay violations (same code for Triangle and our output)

## GPU baseline (A100, HPC)
- [ ] Confirm CUDA module/version on the HPC
- [ ] Set up gDel2D (build with `-arch=sm_80`)
- [ ] Time gDel2D on the initial triangulation
- [ ] Write naive GPU refinement kernel (one thread per bad triangle, global locks) as second baseline
- [ ] `run_gpu_baseline.sh` writing timings to `logs/`

## Comparison
- [ ] Compare our algorithm vs Triangle: time, triangle count, min angle
- [ ] Compare our algorithm vs GPU baselines
