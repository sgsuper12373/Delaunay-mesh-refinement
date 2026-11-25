// Points and geometric predicates
#pragma once
#include <cmath>

// NOTE: nvcc defines these but plain C++ compilers don't
#ifndef __CUDACC__
#define __host__
#define __device__
#endif

constexpr double PI = 3.14159265358979323846;

struct Point
{
    double x;
    double y;
};

__host__ __device__ inline double dist2(Point a, Point b)
{
    double dx = a.x - b.x;
    double dy = a.y - b.y;
    return dx * dx + dy * dy;
}

// NOTE: > 0 if a, b, c turn counter-clockwise; < 0 clockwise; 0 collinear
__host__ __device__ inline double orient(Point a, Point b, Point c)
{
    return (b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x);
}

// NOTE: > 0 if d is strictly inside the circumcircle of counter-clockwise (a, b, c).
__host__ __device__ inline double incircle(Point a, Point b, Point c, Point d)
{
    // NOTE: Shift to make d the origin (determinant computation is easier)
    double ax = a.x - d.x, ay = a.y - d.y;
    double bx = b.x - d.x, by = b.y - d.y;
    double cx = c.x - d.x, cy = c.y - d.y;

    double a2 = ax * ax + ay * ay;
    double b2 = bx * bx + by * by;
    double c2 = cx * cx + cy * cy;

    return a2 * (bx * cy - cx * by) - b2 * (ax * cy - cx * ay) + c2 * (ax * by - bx * ay);
}

// NOTE: Division by zero if they are collinear
__host__ __device__ inline Point circumcenter(Point a, Point b, Point c)
{
    double d = 2.0 * (a.x * (b.y - c.y) + b.x * (c.y - a.y) + c.x * (a.y - b.y));

    double a2 = a.x * a.x + a.y * a.y;
    double b2 = b.x * b.x + b.y * b.y;
    double c2 = c.x * c.x + c.y * c.y;

    return Point{
        (a2 * (b.y - c.y) + b2 * (c.y - a.y) + c2 * (a.y - b.y)) / d,
        (a2 * (c.x - b.x) + b2 * (a.x - c.x) + c2 * (b.x - a.x)) / d};
}

__host__ __device__ inline double min_angle_deg(Point a, Point b, Point c)
{
    double ab = dist2(a, b);
    double bc = dist2(b, c);
    double ca = dist2(c, a);

    // Smallest angle sits opposite the shortest "s"
    double s, p, q;
    if (ab <= bc && ab <= ca)
    {
        s = ab;
        p = bc;
        q = ca;
    }
    else if (bc <= ab && bc <= ca)
    {
        s = bc;
        p = ab;
        q = ca;
    }
    else
    {
        s = ca;
        p = ab;
        q = bc;
    }

    // Law of cosines
    double cosine = (p + q - s) / (2.0 * sqrt(p * q));
    cosine = fmin(1.0, fmax(-1.0, cosine)); // NOTE: clamp as cosine might be slightly incorrect

    return acos(cosine) * 180.0 / PI;
}