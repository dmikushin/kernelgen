/*
 * KGen - the LLVM-based compiler with GPU kernels generation over C backend.
 *
 * Copyright (c) 2011 Dmitry Mikushin
 *
 * This software is provided 'as-is', without any express or implied warranty.
 * In no event will the authors be held liable for any damages arising 
 * from the use of this software.
 * Permission is granted to anyone to use this software for any purpose, 
 * including commercial applications, and to alter it and redistribute it freely,
 * subject to the following restrictions:
 *
 * 1. The origin of this software must not be misrepresented;
 * you must not claim that you wrote the original software.
 * If you use this software in a product, an acknowledgment
 * in the product documentation would be appreciated but is not required.
 * 2. Altered source versions must be plainly marked as such,
 * and must not be misrepresented as being the original software.
 * 3. This notice may not be removed or altered from any source distribution.
 */

#include <kernelgen.h>
#include <malloc.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#define real float

void sincos_(int* nx, int* ny, int* nz, real* x, real* y, real* xy);

int main(int argc, char* argv[])
{
	if (argc != 4)
	{
		printf("Usage: %s <nx> <ny> <ns>\n", argv[0]);
		return 0;
	}
	
	int nx = atoi(argv[1]);
	int ny = atoi(argv[2]);
	int nz = atoi(argv[3]);
	real* x1 = (real*)malloc(sizeof(real) * nx * ny * nz);
	real* x2 = (real*)malloc(sizeof(real) * nx * ny * nz);
	real* y1 = (real*)malloc(sizeof(real) * nx * ny * nz);
	real* y2 = (real*)malloc(sizeof(real) * nx * ny * nz);
	real* xy1 = (real*)malloc(sizeof(real) * nx * ny * nz);
	real* xy2 = (real*)malloc(sizeof(real) * nx * ny * nz);
	double invrmax = 1.0 / RAND_MAX;
	for (int i = 0; i < nx * ny * nz; i++)
	{
		x1[i] = rand() * invrmax; x2[i] = x1[i];
		y1[i] = rand() * invrmax; y2[i] = y1[i];
	}
	
	// Measure time of version ported onto accelerator.
	kernelgen_time_t start, end;
	kernelgen_get_time(&start);
	{
		sincos_(&nx, &ny, &nz, x1, y1, xy1);
	}
	kernelgen_get_time(&end);
	printf("kernelgen time = %f sec\n",
		kernelgen_get_time_diff(&start, &end));
	
	// Measure time of regular CPU version.
	kernelgen_get_time(&start);
	{
		sincos_serial_(&nx, &ny, &nz, x2, y2, xy2);
	}
	kernelgen_get_time(&end);
	printf("regular time = %f sec\n",
		kernelgen_get_time_diff(&start, &end));
	
	// Compare results.
	real maxabsdiff = fabs(xy1[0] - xy2[0]);
	int imaxabsdiff = 0;
	for (int i = 0; i < nx * ny * nz; i++)
	{
		real absdiff = fabs(xy1[i] - xy2[i]);
		if (absdiff > maxabsdiff)
		{
			maxabsdiff = absdiff;
			imaxabsdiff = i;
		}
	}
	printf("max diff = %e @ %d\n", maxabsdiff, imaxabsdiff);

	free(x1); free(x2);
	free(y1); free(y2);
	free(xy1); free(xy2);

	kernelgen_status_t status = kernelgen_get_last_error();
	if (status.value != kernelgen_success)
	{
		printf("%s\n", kernelgen_get_error_string(status));
		return 1;
	}
	
	return 0;
}

