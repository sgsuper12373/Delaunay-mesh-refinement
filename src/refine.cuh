#pragma once
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include "kernels.cuh"

#define CUDA_CHECK(call)                                        \
    do                                                          \
    {                                                           \
        cudaError_t err = (call);                               \
        if (err != cudaSuccess)                                 \
        {                                                       \
            fprintf(stderr, "CUDA error at line %d: %s\n",      \
                    __LINE__, cudaGetErrorString(err));         \
            exit(1);                                            \
        }                                                       \
    } while (0)

constexpr int THREADS = 256;

struct RefineStats
{
    int rounds = 0;
    int points_added = 0;
    int bad_left = 0;
    bool out_of_room = false;
};

inline RefineStats refine(HostMesh &mesh, int max_points, int max_tris)
{
    int num_points = mesh.points.size();
    int num_tris = mesh.tris.size();

    if (num_points > max_points || num_tris > max_tris)
    {
        fprintf(stderr, "Input mesh exceeds capacity\n");
        exit(1);
    }

    // Allocate
    MeshView m{};
    m.max_points = max_points;
    m.max_tris = max_tris;
    int *bad, *owner, *counter;
    Cavity *cavities;
    CUDA_CHECK(cudaMalloc(&m.points, sizeof(Point) * max_points));
    CUDA_CHECK(cudaMalloc(&m.tris, sizeof(Triangle) * max_tris));
    CUDA_CHECK(cudaMalloc(&m.num_points, sizeof(int)));
    CUDA_CHECK(cudaMalloc(&m.num_tris, sizeof(int)));
    CUDA_CHECK(cudaMalloc(&bad, sizeof(int) * MAX_BATCH));
    CUDA_CHECK(cudaMalloc(&owner, sizeof(int) * max_tris));
    CUDA_CHECK(cudaMalloc(&counter, sizeof(int)));
    CUDA_CHECK(cudaMalloc(&cavities, sizeof(Cavity) * MAX_BATCH));

    // Copy initial mesh to device
    CUDA_CHECK(cudaMemcpy(m.points, mesh.points.data(), sizeof(Point) * num_points, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(m.tris, mesh.tris.data(), sizeof(Triangle) * num_tris, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(m.num_points, &num_points, sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(m.num_tris, &num_tris, sizeof(int), cudaMemcpyHostToDevice));

    RefineStats stats;

    while (true)
    {
        int tri_blocks = (num_tris + THREADS - 1) / THREADS;

        // Find bad triangles
        CUDA_CHECK(cudaMemset(counter, 0, sizeof(int)));
        find_bad<<<tri_blocks, THREADS>>>(m, bad, counter);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaMemcpy(&stats.bad_left, counter, sizeof(int), cudaMemcpyDeviceToHost));
        if (stats.bad_left == 0)
            break;

        // Fix a batch
        int batch = std::min(stats.bad_left, MAX_BATCH);
        int batch_blocks = (batch + THREADS - 1) / THREADS;

        CUDA_CHECK(cudaMemset(counter, 0, sizeof(int)));
        build_cavities<<<batch_blocks, THREADS>>>(m, bad, batch, cavities);
        reset_owners<<<tri_blocks, THREADS>>>(owner, num_tris);
        claim<<<batch_blocks, THREADS>>>(cavities, batch, owner);
        retriangulate<<<batch_blocks, THREADS>>>(m, cavities, batch, owner, counter);
        CUDA_CHECK(cudaGetLastError());

        // Read results
        int added;
        CUDA_CHECK(cudaMemcpy(&added, counter, sizeof(int), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(&num_points, m.num_points, sizeof(int), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(&num_tris, m.num_tris, sizeof(int), cudaMemcpyDeviceToHost));

        stats.rounds++;
        stats.points_added += added;
        printf("round %3d: %7d bad, %6d inserted, %8d triangles\n",
               stats.rounds, stats.bad_left, added, num_tris);

        stats.out_of_room = num_points > max_points || num_tris > max_tris;
        if (stats.out_of_room || added == 0 || stats.rounds == MAX_ROUNDS)
            break;
    }

    // Copy back to host
    mesh.points.resize(std::min(num_points, max_points));
    mesh.tris.resize(std::min(num_tris, max_tris));
    CUDA_CHECK(cudaMemcpy(mesh.points.data(), m.points, sizeof(Point) * mesh.points.size(), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(mesh.tris.data(), m.tris, sizeof(Triangle) * mesh.tris.size(), cudaMemcpyDeviceToHost));

    // Free
    cudaFree(m.points);
    cudaFree(m.tris);
    cudaFree(m.num_points);
    cudaFree(m.num_tris);
    cudaFree(bad);
    cudaFree(owner);
    cudaFree(counter);
    cudaFree(cavities);

    return stats;
}