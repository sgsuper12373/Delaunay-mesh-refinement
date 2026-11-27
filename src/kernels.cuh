#pragma once
#include "mesh.cuh"
#include <climits>
#include <cuda_runtime.h>

constexpr double MIN_ANGLE = 30.0;
constexpr int MAX_CAVITY = 32;
constexpr int MAX_EDGES = MAX_CAVITY + 2;
constexpr int MAX_WALK = 256; // obtuse walk
constexpr int MAX_BATCH = 1 << 16;
constexpr int MAX_ROUNDS = 1000;

// It can be a triangle or a segment
struct Element {
  int tri;
  int edge; // NOTE: -1 for triangle
  __host__ __device__ bool is_segment() const { return edge >= 0; }
};

struct Cavity {
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

__device__ inline Point vertex(const MeshView &m, int t, int i) {
  return m.points[m.tris[t].v[i]];
}

__device__ inline bool contains(const int *list, int n, int x) {
  for (int i = 0; i < n; ++i)
    if (list[i] == x)
      return true;
  return false;
}

// circle(e) of the pseudocode: its center and squared radius.
__device__ inline void circle(const MeshView &m, Element e, Point *center,
                              double *r2) {
  if (e.is_segment()) {
    Point a = vertex(m, e.tri, (e.edge + 1) % 3),
          b = vertex(m, e.tri, (e.edge + 2) % 3);
    *center = Point{(a.x + b.x) / 2, (a.y + b.y) / 2};
    *r2 = dist2(a, b) / 4;
  } else {
    *center = circumcenter(vertex(m, e.tri, 0), vertex(m, e.tri, 1),
                           vertex(m, e.tri, 2));
    *r2 = dist2(*center, vertex(m, e.tri, 0));
  }
}

__device__ inline bool inside(const MeshView &m, Point p, Element e) {
  Point center;
  double r2;
  circle(m, e, &center, &r2);
  return dist2(p, center) <= r2;
}

// Kernel: find bad triangles

__global__ void find_bad(MeshView m, int *bad, int *num_bad) {
  int t = blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= *m.num_tris)
    return;

  double angle =
      min_angle_deg(vertex(m, t, 0), vertex(m, t, 1), vertex(m, t, 2));
  if (angle >= MIN_ANGLE)
    return;

  int slot = atomicAdd(num_bad, 1);
  if (slot < MAX_BATCH)
    bad[slot] = t;
}

// Kernel: reset owners

__global__ void reset_owners(int *owner, int n) {
  int t = blockIdx.x * blockDim.x + threadIdx.x;
  if (t < n)
    owner[t] = INT_MAX;
}

// Kernel: claim triangles (lowest cavity id wins each one)

__global__ void claim(const Cavity *cavities, int num, int *owner) {
  int id = blockIdx.x * blockDim.x + threadIdx.x;
  if (id >= num || !cavities[id].valid)
    return;

  const Cavity &c = cavities[id];
  for (int q = 0; q < c.num_tris; ++q)
    atomicMin(&owner[c.tris[q]], id);
  for (int j = 0; j < c.num_edges; ++j)
    if (c.edge_out[j] >= 0)
      atomicMin(&owner[c.edge_out[j]], id);
}

// Walk across the edge opposite each obtuse corner until the triangle has none
// or we hit a boundary edge (a segment)
__device__ inline bool pick_element(const MeshView &m, int t, Element *e) {
  *e = Element{t, -1};
  for (int step = 0; step < MAX_WALK; ++step) {
    int k = -1;
    for (int i = 0; i < 3; ++i) {
      Point p = vertex(m, e->tri, i);
      Point a = vertex(m, e->tri, (i + 1) % 3);
      Point b = vertex(m, e->tri, (i + 2) % 3);
      if ((a.x - p.x) * (b.x - p.x) + (a.y - p.y) * (b.y - p.y) < 0) {
        k = i;
        break;
      }
    }
    if (k < 0)
      return true;

    int next = m.tris[e->tri].nbr[k];
    if (next < 0) {
      e->edge = k;
      return true;
    }
    e->tri = next;
  }
  return false;
}

// Returns false if the center encroaches on a segment; *e becomes that segment
__device__ bool grow(const MeshView &m, Cavity &c, Element *e, bool *too_big) {
  double r2;
  circle(m, *e, &c.center, &r2);
  c.split = *e;
  c.tris[0] = e->tri;
  c.num_tris = 1;

  for (int q = 0; q < c.num_tris; ++q) {
    const Triangle &tri = m.tris[c.tris[q]];
    for (int i = 0; i < 3; ++i) {
      int n = tri.nbr[i];

      if (n < 0) {
        Element seg{c.tris[q], i};
        if (!e->is_segment() && inside(m, c.center, seg)) {
          *e = seg;
          return false;
        }
        continue;
      }

      if (contains(c.tris, c.num_tris, n) ||
          !inside(m, c.center, Element{n, -1}))
        continue;

      if (c.num_tris == MAX_CAVITY) {
        *too_big = true;
        return true;
      }
      c.tris[c.num_tris++] = n;
    }
  }
  return true;
}

__global__ void build_cavities(MeshView m, const int *bad, int num_bad,
                               Cavity *cavities) {
  int id = blockIdx.x * blockDim.x + threadIdx.x;
  if (id >= num_bad)
    return;

  Cavity &c = cavities[id];
  c.valid = false;

  Element e;
  if (!pick_element(m, bad[id], &e))
    return;

  // One retry is allowed: growing can switch the target from a triangle to a
  // segment
  bool too_big = false;
  if (!grow(m, c, &e, &too_big) && !grow(m, c, &e, &too_big))
    return;
  if (too_big)
    return;

  // Collect the edges around the cavity
  c.num_edges = 0;
  for (int q = 0; q < c.num_tris; ++q) {
    const Triangle &tri = m.tris[c.tris[q]];
    for (int i = 0; i < 3; ++i) {
      bool inner_edge =
          tri.nbr[i] >= 0 && contains(c.tris, c.num_tris, tri.nbr[i]);
      bool split_edge = e.is_segment() && c.tris[q] == e.tri && i == e.edge;
      if (inner_edge || split_edge)
        continue;

      int a = tri.v[(i + 1) % 3];
      int b = tri.v[(i + 2) % 3];
      if (orient(m.points[a], m.points[b], c.center) <= 0)
        return;
      if (c.num_edges == MAX_EDGES)
        return;

      c.edge_a[c.num_edges] = a;
      c.edge_b[c.num_edges] = b;
      c.edge_out[c.num_edges] = tri.nbr[i];
      ++c.num_edges;
    }
  }

  // No vertex may start two boundary edges
  for (int j = 0; j < c.num_edges; ++j)
    for (int k = j + 1; k < c.num_edges; ++k)
      if (c.edge_a[j] == c.edge_a[k])
        return;

  int expected = c.num_tris + (e.is_segment() ? 1 : 2);
  c.valid = (c.num_edges == expected);
}

// retriangulate the cavities that won
__global__ void retriangulate(MeshView m, const Cavity *cavities, int num,
                              const int *owner, int *inserted) {
  int id = blockIdx.x * blockDim.x + threadIdx.x;
  if (id >= num || !cavities[id].valid)
    return;

  const Cavity &c = cavities[id];

  // Only go on if this cavity won every triangle it claimed
  for (int q = 0; q < c.num_tris; ++q)
    if (owner[c.tris[q]] != id)
      return;
  for (int j = 0; j < c.num_edges; ++j)
    if (c.edge_out[j] >= 0 && owner[c.edge_out[j]] != id)
      return;

  int p = atomicAdd(m.num_points, 1);
  int extra = atomicAdd(m.num_tris, c.num_edges - c.num_tris);
  if (p >= m.max_points || extra + c.num_edges - c.num_tris > m.max_tris)
    return;
  m.points[p] = c.center;

  int slot[MAX_EDGES];
  for (int j = 0; j < c.num_edges; ++j)
    slot[j] = (j < c.num_tris) ? c.tris[j] : extra + (j - c.num_tris);

  // One new triangle per boundary edge, fanned around p
  for (int j = 0; j < c.num_edges; ++j) {
    int a = c.edge_a[j];
    int b = c.edge_b[j];

    Triangle t;
    t.v[0] = a;
    t.v[1] = b;
    t.v[2] = p;
    t.nbr[2] = c.edge_out[j];
    t.nbr[0] = t.nbr[1] = -1;

    for (int k = 0; k < c.num_edges; ++k) {
      if (c.edge_a[k] == b)
        t.nbr[0] = slot[k];
      if (c.edge_b[k] == a)
        t.nbr[1] = slot[k];
    }

    m.tris[slot[j]] = t;

    // Point the outside triangle back at the new one
    int out = c.edge_out[j];
    if (out >= 0) {
      Triangle &o = m.tris[out];
      for (int i = 0; i < 3; ++i)
        if (o.v[(i + 1) % 3] == b && o.v[(i + 2) % 3] == a)
          o.nbr[i] = slot[j];
    }
  }

  atomicAdd(inserted, 1);
}