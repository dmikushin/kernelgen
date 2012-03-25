#include <malloc.h>
#include <stdio.h>
#include <stdlib.h>

#define real float

void jacobi_(int* m, int* n, int* nit,
	real* a, real* b, real* w0, real* w1, real* w2);

int main(int argc, char* argv[])
{
	if (argc != 3)
	{
		printf("Usage: %s <m> <n>\n", argv[0]);
		return 0;
	}

	int m = atoi(argv[1]), n = atoi(argv[2]);
	real* a = (real*)malloc(sizeof(real) * m * n);
	real* b = (real*)malloc(sizeof(real) * m * n);
	double invrmax = 1.0 / RAND_MAX;
	for (int i = 0; i < m * n; i++)
	{
		a[i] = rand() * invrmax;
		b[i] = rand() * invrmax;
	}

	real w0 = 0.5;
	real w1 = 0.3;
	real w2 = 0.2;

	int nit = 5;

	jacobi_(&m, &n, &nit, a, b, &w0, &w1, &w2);

	return 0;
}

