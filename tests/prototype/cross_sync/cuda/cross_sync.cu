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

#include <cuda_runtime.h>
#include <malloc.h>
#include <stdio.h>
#include <stdlib.h>

__global__ void gpu_kernel(float* data, size_t size, int npasses,
	int* lock, int* finish, int* pmaxidx, float* pmaxval)
{
	*finish = 0;
#ifdef VERBOSE
	printf("gpu kernel starting\n");
#endif
	for (int ipass = 0; ipass < npasses; ipass++)
	{
		// Run some time-consuming work.
		for (int i = size - 1; i >= 0; i--)
			data[i] = data[i - 1];
		data[0] = data[size - 1];
	
		int maxidx = 0;
		float maxval = data[0];
		for (int i = 1; i < size; i++)
			if (data[i] >= maxval)
			{
				maxval = data[i];
				maxidx = i;
			}
		*pmaxidx = maxidx;
		*pmaxval = maxval;

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
	float* data;
	int* maxidx;
	float* maxval;
	int* lock;
	int* finish;
}
cpu, gpu;

int main(int argc, char* argv[])
{
	if (argc != 3)
	{
		printf("%s <size> <npasses>\n", argv[0]);
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

	size_t size = atoi(argv[1]);
	int npasses = atoi(argv[2]);

	cpu.data = (float*)malloc(sizeof(float) * size);
	double dinvrandmax = (double)1.0 / RAND_MAX;
	for (int i = 0; i < size; i++)
		cpu.data[i] = rand() * dinvrandmax;

	gpu.data = NULL;
	custat = cudaMalloc((void**)&gpu.data, sizeof(float) * size);
	if (custat != cudaSuccess)
	{
		fprintf(stderr, "Cannot create GPU data buffer: %s\n",
			cudaGetErrorString(custat));
		return 1;
	}
	custat = cudaMemcpy(gpu.data, cpu.data, sizeof(float) * size,
		cudaMemcpyHostToDevice);
	if (custat != cudaSuccess)
	{
		fprintf(stderr, "Cannot fill GPU data buffer: %s\n",
			cudaGetErrorString(custat));
		return 1;
	}
	free(cpu.data);
	
	custat = cudaMalloc((void**)&gpu.maxidx, sizeof(int));
	if (custat != cudaSuccess)
	{
		fprintf(stderr, "Cannot create GPU maxidx buffer: %s\n",
			cudaGetErrorString(custat));
		return 1;
	}
	
	custat = cudaMalloc((void**)&gpu.maxval, sizeof(float));
	if (custat != cudaSuccess)
	{
		fprintf(stderr, "Cannot create GPU maxval buffer: %s\n",
			cudaGetErrorString(custat));
		return 1;
	}
	
	custat = cudaMallocHost((void**)&gpu.finish, sizeof(int));
	if (custat != cudaSuccess)
	{
		fprintf(stderr, "Cannot create GPU finish buffer: %s\n",
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
	gpu_kernel<<<1, 1, 1, cpu.stream>>>(
		gpu.data, size, npasses, gpu.lock,
		gpu.finish, gpu.maxidx, gpu.maxval);
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

		// Do something with GPU data describing the current
		// running kernel state from host.
		int maxidx = 0;
		custat = cudaMemcpyAsync(&maxidx, gpu.maxidx, sizeof(int),
			cudaMemcpyDeviceToHost, gpu.stream);
		if (custat != cudaSuccess)
		{
			fprintf(stderr, "Cannot get GPU maxidx value: %s\n",
				cudaGetErrorString(custat));
			return 1;
		}
		float maxval = 0.0;
		custat = cudaMemcpyAsync(&maxval, gpu.maxval, sizeof(float),
			cudaMemcpyDeviceToHost, gpu.stream);
		if (custat != cudaSuccess)
		{
			fprintf(stderr, "Cannot get GPU maxval value: %s\n",
				cudaGetErrorString(custat));
			return 1;
		}
		custat = cudaStreamSynchronize(gpu.stream);
		if (custat != cudaSuccess)
		{
			fprintf(stderr, "Cannot synchronize GPU monitor kernel: %s\n",
				cudaGetErrorString(custat));
			return 1;
		}
		printf("max value = %f @ index = %d\n", maxval, maxidx);
		
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

	// Synchronize with target kernel.
	custat = cudaStreamSynchronize(cpu.stream);
	if (custat != cudaSuccess)
	{
		fprintf(stderr, "Cannot synchronize GPU monitor kernel: %s\n",
			cudaGetErrorString(custat));
		return 1;
	}
	
	custat = cudaFree(gpu.data);
	if (custat != cudaSuccess)
	{
		fprintf(stderr, "Cannot release GPU data buffer: %s\n",
			cudaGetErrorString(custat));
		return 1;
	}
	custat = cudaFree(gpu.maxidx);
	if (custat != cudaSuccess)
	{
		fprintf(stderr, "Cannot release GPU maxidx buffer: %s\n",
			cudaGetErrorString(custat));
		return 1;
	}
	custat = cudaFree(gpu.maxval);
	if (custat != cudaSuccess)
	{
		fprintf(stderr, "Cannot release GPU maxval buffer: %s\n",
			cudaGetErrorString(custat));
		return 1;
	}
	custat = cudaFreeHost(gpu.finish);
	if (custat != cudaSuccess)
	{
		fprintf(stderr, "Cannot release GPU finish buffer: %s\n",
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

