#include <malloc.h>
#include <stdio.h>
#include <stdlib.h>

#define real float

void sincos_(int* nx, int* ny, int* nz, real* x, real* y, real* xy1, real* xy2);

int main(int argc, char* argv[])
{
	if (argc != 4)
	{
		printf("Usage: %s <nx> <ny> <nz>\n", argv[0]);
		return 0;
	}

	int nx = atoi(argv[1]), ny = atoi(argv[2]), nz = atoi(argv[3]);
	real* x = (real*)malloc(sizeof(real) * nx * ny * nz);
	real* y = (real*)malloc(sizeof(real) * nx * ny * nz);
	real* xy1 = (real*)malloc(sizeof(real) * nx * ny * nz);
	real* xy2 = (real*)malloc(sizeof(real) * nx * ny * nz);
	double invrmax = 1.0 / RAND_MAX;
	for (int i = 0; i < nx * ny * nz; i++)
	{
		x[i] = rand() * invrmax;
		y[i] = rand() * invrmax;
	}
	
	sincos_(&nx, &ny, &nz, x, y, xy1, xy2);

	free(x);
	free(y);
	free(xy1);
	free(xy2);
	
	return 0;
}

