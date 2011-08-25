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
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

#define real float

void sincos_(int* nx, int* ny, int* nz, real* x, real* y, real* xy);

int main(int argc, char* argv[])
{
	if (argc != 5)
	{
		printf("Usage: %s <nx> <ny> <ns> <ntests>\n", argv[0]);
		return 0;
	}
	
	int nx = atoi(argv[1]);
	int ny = atoi(argv[2]);
	int nz = atoi(argv[3]);
	int ntests = atoi(argv[4]);
	
	if ((nx <= 0) || (ny <= 0) || (nz <= 0) || (ntests <= 0))
	{
		fprintf(stderr, "Invalid input arguments\n");
		return 1;
	}
	
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
	
	int mpi_status = MPI_Init(&argc, &argv);
	if (mpi_status != MPI_SUCCESS)
	{
		fprintf(stderr, "Cannot initialize MPI, status = %d\n",
			mpi_status);
		return 1;
	}
	
	int rank = 0;
	mpi_status = MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	if (mpi_status != MPI_SUCCESS)
	{
		fprintf(stderr, "Cannot get MPI process rank, status = %d\n",
			mpi_status);
		return 1;
	}
	
	// Assign each MPI process with its own device.
	kernelgen_status_t status = kernelgen_set_device(0, rank + 1);
	if (status.value != kernelgen_success)
	{
		fprintf(stderr, "Cannot assign kernelgen device to MPI process: %s\n",
			kernelgen_get_error_string(status));
		return 1;
	}
	
	// Perform specified number of tests.
	for (int itest = 0; itest < ntests; itest++)
	{
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
			for (int i = 0; i < nx * ny * nz; i++)
			{
				xy2[i] = sin(x2[i]) + cos(y2[i]);
			}
		}
		kernelgen_get_time(&end);
		printf("regular time = %f sec\n",
			kernelgen_get_time_diff(&start, &end));
	
		// Compare results.
		real maxabsdiff = 0.0;
		for (int i = 0; i < nx * ny * nz; i++)
		{
			real absdiff = fabs(xy1[i] - xy2[i]);
			if (absdiff > maxabsdiff) maxabsdiff = absdiff;
		}
		printf("max diff = %f\n", maxabsdiff);
	}

	free(x1); free(x2);
	free(y1); free(y2);
	free(xy1); free(xy2);

	status = kernelgen_get_last_error();
	if (status.value != kernelgen_success)
	{
		fprintf(stderr, "%s\n", kernelgen_get_error_string(status));
		return 1;
	}

	mpi_status = MPI_Finalize();
	if (mpi_status != MPI_SUCCESS)
	{
		fprintf(stderr, "Cannot finalize MPI, status = %d\n",
			mpi_status);
		return 1;
	}
	
	return 0;
}

