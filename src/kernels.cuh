#pragma once
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
    Point center;         // insertion point
    int tris[MAX_CAVITY];
    int num_tris;
    int edge_a[MAX_EDGES];
    int edge_b[MAX_EDGES];
    int edge_out[MAX_EDGES];
    int num_edges;
    bool valid;
};