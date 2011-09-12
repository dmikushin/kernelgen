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

#include <math.h>

#define real float

void sincos_cpu(
	int nx, int ny, int nz, real* x, real* y, real* xy)
{
	for (int k = 0; k < nz; k++)
		for (int j = 0; j < ny; j++)
			for (int i = 0; i < nx; i++)
			{
				int index = k * nx * ny + j * nx + i;
				xy[index] = sin(x[index]) + cos(y[index]);
			}
}

__global__ void sincos_gpu(
	int nx, int ny, int nz, real* x, real* y, real* xy)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;
	int k = blockIdx.z * blockDim.z + threadIdx.z;

	int index = k * nx * ny + j * nx + i;

	xy[index] = sin(x[index]) + cos(y[index]);
}

#include <kernelgen.h>
#include <malloc.h>
#include <stdlib.h>
#include <stdio.h>

int main(int argc, char* argv[])
{
	if (argc != 8)
	{
		printf("Usage: %s <bx> <by> <bs> <tx> <ty> <ts> <full_copy>\n", argv[0]);
		return 0;
	}
	
	int bx = atoi(argv[1]), by = atoi(argv[2]), bz = atoi(argv[3]);
	int tx = atoi(argv[4]), ty = atoi(argv[5]), tz = atoi(argv[6]);
	int full_copy = atoi(argv[7]);
	int nx = bx * tx, ny = by * ty, nz = bz * tz;
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
	dim3 blocks(bx, by, bz);
	dim3 threads(tx, ty, tz);
	kernelgen_time_t start, end;
	kernelgen_get_time(&start);
	{
		cudaError_t status = cudaSuccess;
		float *x1_dev, *y1_dev, *xy1_dev;

		status = cudaMalloc((void**)&x1_dev, sizeof(real) * nx * ny * nz);
		if (status != cudaSuccess)
		{
			fprintf(stderr, "Cannot allocate memory for x1: %s\n",
				cudaGetErrorString(status));
			return 1;
		}

		status = cudaMalloc((void**)&y1_dev, sizeof(real) * nx * ny * nz);
		if (status != cudaSuccess)
		{
			fprintf(stderr, "Cannot allocate memory for y1: %s\n",
				cudaGetErrorString(status));
			return 1;
		}
		
		status = cudaMalloc((void**)&xy1_dev, sizeof(real) * nx * ny * nz); 
		if (status != cudaSuccess)
		{
			fprintf(stderr, "Cannot allocate memory for xy1: %s\n",
				cudaGetErrorString(status));
			return 1;
		}

		status = cudaMemcpy(x1_dev, x1, sizeof(real) * nx * ny * nz,
			cudaMemcpyHostToDevice);
		if (status != cudaSuccess)
		{
			fprintf(stderr, "Cannot copy data to x1: %s\n",
				cudaGetErrorString(status));
			return 1;
		}

		status = cudaMemcpy(y1_dev, y1, sizeof(real) * nx * ny * nz,
			cudaMemcpyHostToDevice);
		if (status != cudaSuccess)
		{
			fprintf(stderr, "Cannot copy data to y1: %s\n",
				cudaGetErrorString(status));
			return 1;
		}

		// Optionally skip output-only data copy-in.
		if (full_copy)
		{
			status = cudaMemcpy(xy1_dev, xy1, sizeof(real) * nx * ny * nz,
				cudaMemcpyHostToDevice);
			if (status != cudaSuccess)
			{
				fprintf(stderr, "Cannot copy data to xy1: %s\n",
					cudaGetErrorString(status));
				return 1;
			}
		}

		sincos_gpu<<<blocks, threads>>>(nx, ny, nz, x1_dev, y1_dev, xy1_dev);
		status = cudaGetLastError();
		if (status != cudaSuccess)
		{
			fprintf(stderr, "Cannot launch kernel: %s\n",
				cudaGetErrorString(status));
			return 1;
		}

		status = cudaThreadSynchronize();
		if (status != cudaSuccess)
		{
			fprintf(stderr, "Cannot synchronize kernel: %s\n",
				cudaGetErrorString(status));
			return 1;
		}

		// Optionally skip input-only data copy-out.
		if (full_copy)
		{
			status = cudaMemcpy(x1, x1_dev, sizeof(real) * nx * ny * nz,
				cudaMemcpyDeviceToHost);
			if (status != cudaSuccess)
			{
				fprintf(stderr, "Cannot copy data from x1: %s\n",
					cudaGetErrorString(status));
				return 1;
			}

			status = cudaMemcpy(y1, y1_dev, sizeof(real) * nx * ny * nz,
				cudaMemcpyDeviceToHost);
			if (status != cudaSuccess)
			{
				fprintf(stderr, "Cannot copy data from y1: %s\n",
					cudaGetErrorString(status));
				return 1;
			}
		}

		status = cudaMemcpy(xy1, xy1_dev, sizeof(real) * nx * ny * nz,
			cudaMemcpyDeviceToHost);
		if (status != cudaSuccess)
		{
			fprintf(stderr, "Cannot copy data from xy1: %s\n",
				cudaGetErrorString(status));
			return 1;
		}

		status = cudaFree(x1_dev);
		if (status != cudaSuccess)
		{
			fprintf(stderr, "Cannot deallocate memory for x1: %s\n",
				cudaGetErrorString(status));
			return 1;
		}

		status = cudaFree(y1_dev);
		if (status != cudaSuccess)
		{
			fprintf(stderr, "Cannot deallocate memory for y1: %s\n",
				cudaGetErrorString(status));
			return 1;
		}

		status = cudaFree(xy1_dev);
		if (status != cudaSuccess)
		{
			fprintf(stderr, "Cannot deallocate memory for xy1: %s\n",
				cudaGetErrorString(status));
			return 1;
		}
	}
	kernelgen_get_time(&end);
	printf("gpu time = %f sec\n",
		kernelgen_get_time_diff(&start, &end));
	
	// Measure time of regular CPU version.
	kernelgen_get_time(&start);
	{
		sincos_cpu(nx, ny, nz, x2, y2, xy2);
	}
	kernelgen_get_time(&end);
	printf("cpu time = %f sec\n",
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

