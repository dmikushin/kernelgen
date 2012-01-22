/*
 * KernelGen - the LLVM-based compiler with GPU kernels generation over C backend.
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

#include <cublas_v2.h>
#include <malloc.h>
#include <stdio.h>
#include <stdlib.h>
#include "timing.h"

__global__ void gpu_kernel(int npasses, int* lock, int* finish)
{
	*finish = 0;
#ifdef VERBOSE
	printf("gpu kernel starting\n");
#endif
	for (int ipass = 0; ipass < npasses - 1; ipass++)
	{
		// Thread runs when lock = 0 and gets blocked
		// on lock = 1.
		
		// Lock thread.
		atomicCAS(lock, 0, 1);
#ifdef VERBOSE
		printf("gpu kernel acquires lock\n");
#endif
		// Wait for unlock.
		while (atomicCAS(lock, 0, 0)) continue;
	}

	// Lock thread.
	atomicCAS(lock, 0, 1);
#ifdef VERBOSE	
	printf("gpu kernel finishing\n");
#endif
	*finish = 1;
}

__global__ void gpu_monitor(int* lock)
{
#ifdef VERBOSE
	printf("gpu monitor starting\n");
#endif
	// Unlock blocked gpu kernel associated
	// with lock. It simply waits for lock
	// to be dropped to zero.
	atomicCAS(lock, 1, 0);
#ifdef VERBOSE
	printf("gpu monitor releases lock\n");
#endif
	// Wait for lock to be set.
	// When lock is set this thread exits,
	// and CPU monitor thread gets notified
	// by synchronization.
	while (!atomicCAS(lock, 1, 1)) continue;
#ifdef VERBOSE
	printf("gpu monitor finishing\n");
#endif
}

struct params_t
{
	cudaStream_t stream;
	int* lock;
	int* finish;
}
cpu, gpu;

int main(int argc, char* argv[])
{
	if (argc != 3)
	{
		printf("%s <npasses> <n>\n", argv[0]);
		printf("\t<npasses> - the number of iterations\n");
		printf("\t<n> - the dimension of matrix in dgemm\n");
		return 0;
	}

	int count = 0;
	cudaError_t custat = cudaGetDeviceCount(&count);
	if (custat != cudaSuccess)
	{
		fprintf(stderr, "Cannot get CUDA device count: %s\n",
			cudaGetErrorString(custat));
		return 1;
	}
	if (!count)
	{
		fprintf(stderr, "No CUDA devices found\n");
		return 1;
	}

	int npasses = atoi(argv[1]);
	int n = atoi(argv[2]);

	// Generate random matrix.
	size_t szmatrix = sizeof(double) * n * n;
	double* matrix = (double*)malloc(szmatrix);
	double dinvrandmax = 1.0 / RAND_MAX;
	for (int i = 0; i < n * n; i++)
		matrix[i] = rand() * dinvrandmax;
	
	// Create copies of random matrix in GPU memory.
	double *a = NULL, *b = NULL, *c = NULL;
	custat = cudaMalloc(&a, szmatrix);
	if (custat != cudaSuccess)
	{
		fprintf(stderr, "Cannot create GPU matrix A: %s\n",
			cudaGetErrorString(custat));
		return 1;
	}
	custat = cudaMemcpy(a, matrix, szmatrix, cudaMemcpyHostToDevice);
	if (custat != cudaSuccess)
	{
		fprintf(stderr, "Cannot initialize GPU matrix A: %s\n",
			cudaGetErrorString(custat));
		return 1;
	}
	custat = cudaMalloc(&b, szmatrix);
	if (custat != cudaSuccess)
	{
		fprintf(stderr, "Cannot create GPU matrix B: %s\n",
			cudaGetErrorString(custat));
		return 1;
	}
	custat = cudaMemcpy(b, matrix, szmatrix, cudaMemcpyHostToDevice);
	if (custat != cudaSuccess)
	{
		fprintf(stderr, "Cannot initialize GPU matrix B: %s\n",
			cudaGetErrorString(custat));
		return 1;
	}
	custat = cudaMalloc(&c, szmatrix);
	if (custat != cudaSuccess)
	{
		fprintf(stderr, "Cannot create GPU matrix C: %s\n",
			cudaGetErrorString(custat));
		return 1;
	}
	custat = cudaMemcpy(c, matrix, szmatrix, cudaMemcpyHostToDevice);
	if (custat != cudaSuccess)
	{
		fprintf(stderr, "Cannot initialize GPU matrix C: %s\n",
			cudaGetErrorString(custat));
		return 1;
	}

	// Initialize thread locker variable.
	// Initial state is "locked". It will be dropped
	// by gpu side monitor that must be started *before*
	// target GPU kernel.
	custat = cudaMalloc((void**)&gpu.lock, sizeof(int));
	if (custat != cudaSuccess)
	{
		fprintf(stderr, "Cannot create GPU lock buffer: %s\n",
			cudaGetErrorString(custat));
		return 1;
	}
	int one = 1;
	custat = cudaMemcpy(gpu.lock, &one, sizeof(int),
		cudaMemcpyHostToDevice);
	if (custat != cudaSuccess)
	{
		fprintf(stderr, "Cannot initialize GPU lock buffer: %s\n",
			cudaGetErrorString(custat));
		return 1;
	}

	// Initialize finish indicator.
	custat = cudaMallocHost((void**)&gpu.finish, sizeof(int));
	if (custat != cudaSuccess)
	{
		fprintf(stderr, "Cannot create host pinned buffer: %s\n",
			cudaGetErrorString(custat));
		return 1;
	}
	*gpu.finish = 0;

	// Create streams where monitoring and target kernels
	// will be executed.
	custat = cudaStreamCreate(&gpu.stream);
	if (custat != cudaSuccess)
	{
		fprintf(stderr, "Cannot create monitoring stream: %s\n",
			cudaGetErrorString(custat));
		return 1;
	}
	custat = cudaStreamCreate(&cpu.stream);
	if (custat != cudaSuccess)
	{
		fprintf(stderr, "Cannot create monitoring stream: %s\n",
			cudaGetErrorString(custat));
		return 1;
	}

	// Create CUBLAS handle.
	cublasHandle_t handle;
	cublasStatus_t cberr = cublasCreate_v2(&handle);
	if (cberr != CUBLAS_STATUS_SUCCESS)
	{
		fprintf(stderr, "Cannot create cublas handle: %d\n", cberr);
		return 1;
	}
	
	// Bind CUBLAS handle to gpu monitor stream.
	cberr = cublasSetStream(handle, gpu.stream);
	if (cberr != CUBLAS_STATUS_SUCCESS)
	{
		fprintf(stderr, "Cannot bind cublas handle to gpu monitor stream: %d\n", cberr);
		return 1;
	}

	double avg_without = 0.0;
	printf("Testing cublasDgemm perf WITHOUT concurrent kernel running:\n");
	for (int ipass = 0; ipass < npasses; ipass++)
	{
		// Measure time of dgemm execution.
		util_time_t start;
		util_get_time(&start);

		// Perform CUBLAS dgemm in gpu stream.
		double alpha = 1.0, beta = 0.0;
		cberr = cublasDgemm_v2(
			handle, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n,
			&alpha, a, n, b, n, &beta, c, n);
		if (cberr != CUBLAS_STATUS_SUCCESS)
		{
			fprintf(stderr, "Error launching cublasSgemm_v2: %d\n", cberr);
			return 1;
		}

		// Synchronize with monitoring kernel.
		custat = cudaStreamSynchronize(gpu.stream);
		if (custat != cudaSuccess)
		{
			fprintf(stderr, "Cannot synchronize GPU monitor kernel: %s\n",
				cudaGetErrorString(custat));
			return 1;
		}
		
		// Measure time of dgemm execution.
		util_time_t stop;
		util_get_time(&stop);
		double time = util_get_time_diff(&start, &stop);
		printf("cublasDgemm time: %f sec\n", time);
		avg_without += time;
	}
	avg_without /= npasses;
	printf("avg = %f\n\n", avg_without);

	double avg_with = 0.0;
	printf("Testing cublasDgemm perf WITH concurrent kernel running:\n");

	// Launch GPU monitoring kernel.
	gpu_monitor<<<1, 1, 1, gpu.stream>>>(gpu.lock);
	custat = cudaGetLastError();
	if (custat != cudaSuccess)
	{
		fprintf(stderr, "Cannot launch monitoring GPU kernel: %s\n",
			cudaGetErrorString(custat));
		return 1;
	}
	
	// Execute target GPU kernel.
	gpu_kernel<<<1, 1, 1, cpu.stream>>>(npasses, gpu.lock, gpu.finish);
	custat = cudaGetLastError();
	if (custat != cudaSuccess)
	{
		fprintf(stderr, "Cannot launch target GPU kernel: %s\n",
			cudaGetErrorString(custat));
		return 1;
	}
#ifdef VERBOSE
	int istep = 0;
#endif
	while (1)
	{
		// Synchronize with monitoring kernel.
		custat = cudaStreamSynchronize(gpu.stream);
		if (custat != cudaSuccess)
		{
			fprintf(stderr, "Cannot synchronize GPU monitor kernel: %s\n",
				cudaGetErrorString(custat));
			return 1;
		}

		// Measure time of dgemm execution.
		util_time_t start;
		util_get_time(&start);

		// Perform CUBLAS dgemm in gpu stream.
		double alpha = 1.0, beta = 0.0;
		cberr = cublasDgemm_v2(
			handle, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n,
			&alpha, a, n, b, n, &beta, c, n);
		if (cberr != CUBLAS_STATUS_SUCCESS)
		{
			fprintf(stderr, "Error launching cublasSgemm_v2: %d\n", cberr);
			return 1;
		}

		// Synchronize with monitoring kernel.
		custat = cudaStreamSynchronize(gpu.stream);
		if (custat != cudaSuccess)
		{
			fprintf(stderr, "Cannot synchronize GPU monitor kernel: %s\n",
				cudaGetErrorString(custat));
			return 1;
		}
		
		// Measure time of dgemm execution.
		util_time_t stop;
		util_get_time(&stop);
		double time = util_get_time_diff(&start, &stop);
		printf("cublasDgemm time: %f sec\n", time);
		avg_with += time;
		
                // Check if target GPU kernel has finished.
                if (*gpu.finish == 1) break;
	
		// Again, launch GPU monitoring kernel.
		gpu_monitor<<<1, 1, 1, gpu.stream>>>(gpu.lock);
		custat = cudaGetLastError();
		if (custat != cudaSuccess)
		{
			fprintf(stderr, "Cannot launch monitoring GPU kernel: %s\n",
				cudaGetErrorString(custat));
			return 1;
		}
#ifdef VERBOSE
		istep++;
		printf("step %d\n", istep);
#endif
	}
	avg_with /= npasses;
	printf("avg = %f\n\n", avg_with);
	printf("with : without diff is %f%\n", avg_with / avg_without * 100 - 100);

	// Synchronize with target kernel.
	custat = cudaStreamSynchronize(cpu.stream);
	if (custat != cudaSuccess)
	{
		fprintf(stderr, "Cannot synchronize GPU monitor kernel: %s\n",
			cudaGetErrorString(custat));
		return 1;
	}
	
	// Destroy CUBLAS handle.
	cberr = cublasDestroy_v2(handle);
	if (cberr != CUBLAS_STATUS_SUCCESS)
	{
		fprintf(stderr, "Cannot destroy cublas handle: %d\n", cberr);
		return 1;
	}

	// Cleanups.	
	custat = cudaFree(a);
	if (custat != cudaSuccess)
	{
		fprintf(stderr, "Cannot release GPU matrix A: %s\n",
			cudaGetErrorString(custat));
		return 1;
	}
	custat = cudaFree(b);
	if (custat != cudaSuccess)
	{
		fprintf(stderr, "Cannot release GPU matrix B: %s\n",
			cudaGetErrorString(custat));
		return 1;
	}
	custat = cudaFree(c);
	if (custat != cudaSuccess)
	{
		fprintf(stderr, "Cannot release GPU matrix C: %s\n",
			cudaGetErrorString(custat));
		return 1;
	}
	custat = cudaFree(gpu.lock);
	if (custat != cudaSuccess)
	{
		fprintf(stderr, "Cannot release GPU lock buffer: %s\n",
			cudaGetErrorString(custat));
		return 1;
	}

	custat = cudaStreamDestroy(gpu.stream);
	if (custat != cudaSuccess)
	{
		fprintf(stderr, "Cannot create monitoring stream: %s\n",
			cudaGetErrorString(custat));
		return 1;
	}
	custat = cudaStreamDestroy(cpu.stream);
	if (custat != cudaSuccess)
	{
		fprintf(stderr, "Cannot create monitoring stream: %s\n",
			cudaGetErrorString(custat));
		return 1;
	}

	return 0;
}

