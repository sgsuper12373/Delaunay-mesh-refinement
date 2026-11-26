#pragma once
#include "geometry.cuh"
#include <map>
#include <vector>

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

struct HostMesh {
  std::vector<Point> points;
  std::vector<Triangle> tris;

  void link_neighbours() {
    std::map<std::pair<int, int>, std::pair<int, int>>
        edges; // (from, to) -> (triangle, slot)
    for (int t = 0; t < (int)tris.size(); ++t)
      for (int i = 0; i < 3; ++i) {
        tris[t].nbr[i] = -1;
        edges[{tris[t].v[(i + 1) % 3], tris[t].v[(i + 2) % 3]}] = {t, i};
      }
    for (int t = 0; t < (int)tris.size(); ++t)
      for (int i = 0; i < 3; ++i) {
        auto twin =
            edges.find({tris[t].v[(i + 2) % 3],
                        tris[t].v[(i + 1) % 3]}); // same edge, other way round
        if (twin != edges.end())
          tris[t].nbr[i] = twin->second.first;
      }
  }

  static HostMesh grid(int nx, int ny, double w, double h) {
    HostMesh m;
    for (int j = 0; j <= ny; ++j)
      for (int i = 0; i <= nx; ++i)
        m.points.push_back(Point{i * w, j * h});
    auto id = [nx](int i, int j) { return j * (nx + 1) + i; };
    for (int j = 0; j < ny; ++j)
      for (int i = 0; i < nx; ++i) {
        int p00 = id(i, j), p10 = id(i + 1, j), p11 = id(i + 1, j + 1),
            p01 = id(i, j + 1);
        m.tris.push_back(Triangle{{p00, p10, p11}, {-1, -1, -1}});
        m.tris.push_back(Triangle{{p00, p11, p01}, {-1, -1, -1}});
      }
    m.link_neighbours();
    return m;
  }

  double min_angle() const {
    double smallest = 180.0;
    for (const Triangle &t : tris)
      smallest = fmin(smallest, min_angle_deg(points[t.v[0]], points[t.v[1]],
                                              points[t.v[2]]));
    return smallest;
  }

  int count_bad(double angle_deg) const {
    int bad = 0;
    for (const Triangle &t : tris)
      bad += min_angle_deg(points[t.v[0]], points[t.v[1]], points[t.v[2]]) <
             angle_deg;
    return bad;
  }
};