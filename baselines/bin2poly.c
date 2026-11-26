// Convert binary <name>_vertex.txt / <name>_constraint.txt to Triangle .poly
// vertex file:     int32 n, then n * (float32 x, float32 y)
// constraint file: int32 m, then m * (int32 a, int32 b)  (0-based vertex ids)


/*
    
*/
#include <stdio.h>
#include <stdlib.h>
int main(int argc, char **argv)
{
    //verify args
    if (argc != 4)
    {
        fprintf(stderr, "usage: %s vertex constraint out.poly\n", argv[0]);
        return 1;
    }

    //open file 
    FILE *fv = fopen(argv[1], "rb"), *fc = fopen(argv[2], "rb"), *fo = fopen(argv[3], "w");
    if (!fv || !fc || !fo)
    {
        perror("open");
        return 1;
    }

    // read the total number of vertices(n) and constrained segments(m) and validate 
    int n, m;
    if (fread(&n, 4, 1, fv) != 1 || fread(&m, 4, 1, fc) != 1)
    {
        fprintf(stderr, "bad header\n");
        return 1;
    }

    // Allocate heap buffers and bulk-read payloads
    float *xy = malloc(sizeof(float) * 2 * (size_t)n);
    int *ab = malloc(sizeof(int) * 2 * (size_t)m);


    // read values into heap buffers 
    if (fread(xy, 8, n, fv) != (size_t)n || fread(ab, 8, m, fc) != (size_t)m)
    {
        fprintf(stderr, "truncated file\n");
        return 1;
    }

    fprintf(fo, "%d 2 0 0\n", n);
    for (int i = 0; i < n; i++)
        fprintf(fo, "%d %.9g %.9g\n", i, xy[2 * i], xy[2 * i + 1]); // 0-based

    
    fprintf(fo, "%d 0\n", m);
    for (int i = 0; i < m; i++)
        fprintf(fo, "%d %d %d\n", i, ab[2 * i], ab[2 * i + 1]);

    
    fprintf(fo, "0\n"); // number of holes
    fclose(fo);
    fprintf(stderr, "wrote %d vertices, %d segments\n", n, m);
    return 0;
}
