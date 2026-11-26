#include "geometry.cuh"

struct Triangle {
  int v[3];
  int nbr[3];
};

struct MeshView {
  Point *points;
  Triangle *tris;
  int *num_points;
  int *num_tris;
  int max_points;
  int max_tris;
};