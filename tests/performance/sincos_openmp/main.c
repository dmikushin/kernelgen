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
#include <string.h>

#define real float

void sincos_(int* nx, int* ny, int* nz, real* x, real* y, real* xy);

int main(int argc, char* argv[])
{
	if (argc != 6)
	{
		printf("Usage: %s <nx> <ny> <ns> <nthreads> <ntests>\n", argv[0]);
		return 0;
	}
	
	int nx = atoi(argv[1]);
	int ny = atoi(argv[2]);
	int nz = atoi(argv[3]);
	int nthreads = atoi(argv[4]);
	int ntests = atoi(argv[5]);
	
	if ((nx <= 0) || (ny <= 0) || (nz <= 0) || (nthreads <= 0) || (ntests <= 0))
	{
		fprintf(stderr, "Invalid input arguments\n");
		return 1;
	}

	size_t size = nx * ny * nz;	
	real* x = (real*)malloc(sizeof(real) * size * (nthreads + 1));
	real* y = (real*)malloc(sizeof(real) * size * (nthreads + 1));
	real* xy = (real*)malloc(sizeof(real) * size * (nthreads + 1));
	double invrmax = 1.0 / RAND_MAX;
	for (int i = 0; i < size; i++)
	{
		x[i] = rand() * invrmax;
		y[i] = rand() * invrmax;
	}
	for (int ithread = 0; ithread < nthreads; ithread++)
	{
		memcpy(x + (ithread + 1) * size, x, size);
		memcpy(y + (ithread + 1) * size, y, size);
	}
	
	// Perform specified number of tests.
	for (int itest = 0; itest < ntests; itest++)
	{
		// Measure time of version ported onto accelerator.
		kernelgen_time_t start, end;
		kernelgen_get_time(&start);
		{
			// Run the specified number of OpenMP threads.
			#pragma omp parallel for
			for (int ithread = 0; ithread < nthreads; ithread++)
			{
				// Assign work device to the current thread.
				kernelgen_set_device(0, ithread);
			
				sincos_(&nx, &ny, &nz,
					x + (ithread + 1) * size, y + (ithread + 1) * size,
					xy + (ithread + 1) * size);
			}
		}
		kernelgen_get_time(&end);
		printf("kernelgen time = %f sec\n",
			kernelgen_get_time_diff(&start, &end));
	
		// Measure time of regular CPU version.
		kernelgen_get_time(&start);
		{
			for (int i = 0; i < nx * ny * nz; i++)
			{
				xy[i] = sin(x[i]) + cos(y[i]);
			}
		}
		kernelgen_get_time(&end);
		printf("regular time = %f sec\n",
			kernelgen_get_time_diff(&start, &end));
	
		// Compare results.
		real maxabsdiff = 0.0;
		for (int ithread = 0; ithread < nthreads; ithread++)
		{
			for (int i = 0; i < nx * ny * nz; i++)
			{
				real absdiff = fabs(xy[i] - xy[i + size * (ithread + 1)]);
				if (absdiff > maxabsdiff) maxabsdiff = absdiff;
			}
		}
		printf("max diff = %f\n", maxabsdiff);
	}

	free(x);
	free(y);
	free(xy);

	kernelgen_status_t status = kernelgen_get_last_error();
	if (status.value != kernelgen_success)
	{
		fprintf(stderr, "%s\n", kernelgen_get_error_string(status));
		return 1;
	}
	
	return 0;
}

