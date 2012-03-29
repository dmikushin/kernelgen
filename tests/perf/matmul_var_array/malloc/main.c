#include <malloc.h>
#include <stdio.h>
#include <stdlib.h>

#define real float

void matmul(int n, float a[n][n], float b[n][n], float c[n][n]);

int main(int argc, char* argv[])
{
	if (argc != 2)
	{
		printf("Usage: %s <n>\n", argv[0]);
		return 0;
	}

	int n = atoi(argv[1]);
	real* a = (real*)malloc(sizeof(real) * n * n);
	real* b = (real*)malloc(sizeof(real) * n * n);
	real* c = (real*)malloc(sizeof(real) * n * n);
	double invrmax = 1.0 / RAND_MAX;
	for (int i = 0; i < n * n; i++)
	{
		a[i] = rand() * invrmax;
		b[i] = rand() * invrmax;
		c[i] = rand() * invrmax;
	}

	matmul(n, (float(*)[n])a, (float(*)[n])b, (float(*)[n])c);

	free(a);
	free(b);
	free(c);

	return 0;
}

