#include <stdio.h>
#include <stdlib.h>

void usage(const char* filename)
{
	printf("Usage: %s <n>\n", filename);
	printf("n must be positive\n");
}

void generate(int n, float a[][n][n], float b[][n][n], float c[][n][n])
{
	float finvrmax = 1.0 / RAND_MAX;
	for (int k = 0; k < n; k++)
		for (int j = 0; j < n; j++)
			for (int i = 0; i < n; i++)
                        {
				a[k][j][i] = rand() * finvrmax;
				b[k][j][i] = rand() * finvrmax;
				c[k][j][i] = rand() * finvrmax;
			}
}

void sum(int n, float a[][n][n], float b[][n][n], float c[][n][n])
{
	for (int k = 0; k < n; k++)
		for (int j = 0; j < n; j++)
			for (int i = 0; i < n; i++)
				c[k][i][j] += a[k][i][j] + b[k][i][j];
}

void check(int n, float a[][n][n], float b[][n][n], float c[][n][n])
{
	float asum = 0, amin = a[0][0][0], amax = a[0][0][0];
	float bsum = 0, bmin = b[0][0][0], bmax = b[0][0][0];
	float csum = 0, cmin = c[0][0][0], cmax = c[0][0][0];

	for (int k = 0; k < n; k++)
		for (int j = 0; j < n; j++)
			for (int i = 0; i < n; i++)
			{
				float aval = a[k][j][i];
				asum += aval;
				amin = amin > aval ? aval : amin;
				amax = amax < aval ? aval : amax;

				float bval = b[k][j][i];
				bsum += bval;
				bmin = bmin > bval ? bval : bmin;
				bmax = bmax < bval ? bval : bmax;

				float cval = c[k][j][i];
				csum += cval;
				cmin = cmin > cval ? cval : cmin;
				cmax = cmax < cval ? cval : cmax;
			}

	printf("asum = %f, amin = %f, amax = %f\n", asum, amin, amax);
	printf("bsum = %f, bmin = %f, bmax = %f\n", bsum, bmin, bmax);
	printf("csum = %f, cmin = %f, cmax = %f\n", csum, cmin, cmax);
}

int main(int argc, char* argv[])
{
	if (argc != 2)
	{
		usage(argv[0]);
		return 0;
	}
	int n = atoi(argv[1]);
	if (n <= 0)
	{
		usage(argv[0]);
		return 0;
	}

	float* a = (float*)malloc(n * n * n * sizeof(float));
	float* b = (float*)malloc(n * n * n * sizeof(float));
	float* c = (float*)malloc(n * n * n * sizeof(float));

	generate	(n, (float(*)[n][n])a, (float(*)[n][n])b, (float(*)[n][n])c);
	sum		(n, (float(*)[n][n])a, (float(*)[n][n])b, (float(*)[n][n])c);
	check		(n, (float(*)[n][n])a, (float(*)[n][n])b, (float(*)[n][n])c);

	generate	(n, (float(*)[n][n])a, (float(*)[n][n])b, (float(*)[n][n])c);
	sum		(n, (float(*)[n][n])a, (float(*)[n][n])b, (float(*)[n][n])c);
	check		(n, (float(*)[n][n])a, (float(*)[n][n])b, (float(*)[n][n])c);
	
	free(a);
	free(b);
	free(c);

	return 0;
}

