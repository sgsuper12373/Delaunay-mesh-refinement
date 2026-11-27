#pragma once
#include <cuda_runtime.h>
#include <climits>
#include "mesh.cuh"

constexpr double MIN_ANGLE = 30.0;
constexpr int MAX_CAVITY = 32;
constexpr int MAX_EDGES = MAX_CAVITY + 2;
constexpr int MAX_WALK = 256; // obtuse walk
constexpr int MAX_BATCH = 1 << 16;
constexpr int MAX_ROUNDS = 1000;

// It can be a triangle or a segment
struct Element
{
    int tri;
    int edge; // NOTE: -1 for triangle
    __host__ __device__ bool is_segment() const { return edge >= 0; }
};

struct Cavity
{
    Element split;
    Point center; // insertion point
    int tris[MAX_CAVITY];
    int num_tris;
    int edge_a[MAX_EDGES];
    int edge_b[MAX_EDGES];
    int edge_out[MAX_EDGES];
    int num_edges;
    bool valid;
};

// Helpers

__device__ inline Point vertex(const MeshView &m, int t, int i)
{
    return m.points[m.tris[t].v[i]];
}

// Kernel 1: find bad triangles

__global__ void find_bad(MeshView m, int *bad, int *num_bad)
{
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= *m.num_tris)
        return;

    double angle = min_angle_deg(vertex(m, t, 0), vertex(m, t, 1), vertex(m, t, 2));
    if (angle >= MIN_ANGLE)
        return;

    int slot = atomicAdd(num_bad, 1);
    if (slot < MAX_BATCH)
        bad[slot] = t;
}