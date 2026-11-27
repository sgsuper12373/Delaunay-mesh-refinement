#include <cstdio>
#include <cstdlib>

#include "refine.cuh"

int main(int argc, char **argv)
{
    int nx = argc > 1 ? atoi(argv[1]) : 200;
    int ny = argc > 2 ? atoi(argv[2]) : 200;
    double w = argc > 3 ? atof(argv[3]) : 4.0;
    double h = argc > 4 ? atof(argv[4]) : 1.0;
    if (nx <= 0 || ny <= 0 || w <= 0 || h <= 0)
    {
        fprintf(stderr, "usage: %s [nx ny w h]   (all values must be > 0)\n", argv[0]);
        return 1;
    }

    HostMesh mesh = HostMesh::grid(nx, ny, w, h);
    printf("start: %zu points, %zu triangles, smallest angle %.2f deg, %d below %.1f deg\n",
           mesh.points.size(), mesh.tris.size(), mesh.min_angle(), mesh.count_bad(MIN_ANGLE), MIN_ANGLE);
    mesh.save_obj("before.obj");

    // Room to grow (size_t first: int * 20 overflows for large meshes)
    size_t mp = mesh.points.size() * 20, mt = mesh.tris.size() * 20;
    if (mp > INT_MAX || mt > INT_MAX)
    {
        fprintf(stderr, "mesh too large for 20x capacity (int indices)\n");
        return 1;
    }
    int max_points = (int)mp, max_tris = (int)mt;

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start));
    RefineStats stats = refine(mesh, max_points, max_tris);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float ms = 0;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    CUDA_CHECK(cudaGetLastError());
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    printf("done:  %zu points (+%d), %zu triangles, %d rounds, %.1f ms\n",
           mesh.points.size(), stats.points_added, mesh.tris.size(), stats.rounds, ms);
    printf("       smallest angle %.2f deg, %d triangles still below %.1f deg%s\n",
           mesh.min_angle(), mesh.count_bad(MIN_ANGLE), MIN_ANGLE,
           stats.out_of_room ? " (stopped: out of room, mesh may be incomplete)" : "");
    printf("       (at 30 degrees refinement is not proven to end: some bad triangles may remain)\n");
    mesh.save_obj("after.obj");
    return 0;
}