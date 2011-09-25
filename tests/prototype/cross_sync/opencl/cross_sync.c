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

#include <malloc.h>
#include <stdio.h>
#include <stdlib.h>

#include "builder.h"
#include "error.h"

struct params_t
{
	float* data;
}
cpu;

struct gpu_params_t
{
	cl_mem data;
	cl_mem maxidx;
	cl_mem maxval;
	cl_mem lock;
	cl_mem finish;
}
gpu;

int main(int argc, char* argv[])
{
	if (argc != 3)
	{
		printf("%s <size> <npasses>\n", argv[0]);
		return 0;
	}

	builder_config_t* config = builder_init("cross_sync.cl", 2);
	if (!config) return 1;

	size_t size = atoi(argv[1]);
	int npasses = atoi(argv[2]);

	cpu.data = (float*)malloc(sizeof(float) * size);
	double dinvrandmax = (double)1.0 / RAND_MAX;
	for (int i = 0; i < size; i++)
		cpu.data[i] = rand() * dinvrandmax;

	cl_int clstat = CL_SUCCESS;
	gpu.data = clCreateBuffer(config->context,
		CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
		sizeof(float) * size, cpu.data, &clstat);
	if (clstat != CL_SUCCESS)
	{
		fprintf(stderr, "Cannot create GPU data buffer: %s\n",
			get_error_string(clstat));
		return 1;
	}
	free(cpu.data);

	gpu.maxidx = clCreateBuffer(config->context,
		CL_MEM_READ_WRITE, sizeof(int), NULL, &clstat);
	if (clstat != CL_SUCCESS)
	{
		fprintf(stderr, "Cannot create GPU maxidx buffer: %s\n",
			get_error_string(clstat));
		return 1;
	}

	gpu.maxval = clCreateBuffer(config->context,
		CL_MEM_READ_WRITE, sizeof(float), NULL, &clstat);
	if (clstat != CL_SUCCESS)
	{
		fprintf(stderr, "Cannot create GPU maxval buffer: %s\n",
			get_error_string(clstat));
		return 1;
	}

	gpu.finish = clCreateBuffer(config->context,
		CL_MEM_READ_WRITE, sizeof(int), NULL, &clstat);	
	if (clstat != CL_SUCCESS)
	{
		fprintf(stderr, "Cannot create GPU finish buffer: %s\n",
			get_error_string(clstat));
		return 1;
	}

	// Initialize thread locker variable.
	// Initial state is "locked". It will be dropped
	// by gpu side monitor that must be started *before*
	// target GPU kernel.
	int one = 1;
	gpu.lock = clCreateBuffer(config->context,
		CL_MEM_READ_WRITE, sizeof(int), &one, &clstat);
	if (clstat != CL_SUCCESS)
	{
		fprintf(stderr, "Cannot create GPU lock buffer: %s\n",
			get_error_string(clstat));
		return 1;
	}
	
	// Launch GPU monitoring kernel.
	gpu_monitor<<<1, 1, 1, gpu.stream>>>(gpu.lock);
	custat = cudaGetLastError();
	if (clstat != CL_SUCCESS)
	{
		fprintf(stderr, "Cannot launch monitoring GPU kernel: %s\n",
			get_error_string(clstat));
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
			get_error_string(clstat));
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
			fprintf(stderr, "Cannot synchronize GPU kernel: %s\n",
				get_error_string(clstat));
			return 1;
		}

		/*// Do something with GPU data.
		int maxidx = 0;
		custat = cudaMemcpy(&maxidx, params.maxidx, sizeof(int),
			cudaMemcpyDeviceToHost);
		if (custat != cudaSuccess)
		{
			fprintf(stderr, "Cannot get GPU maxidx value: %s\n",
				get_error_string(clstat));
			return 1;
		}
		float maxval = 0.0;
		custat = cudaMemcpy(&maxval, params.maxval, sizeof(float),
			cudaMemcpyDeviceToHost);
		if (custat != cudaSuccess)
		{
			fprintf(stderr, "Cannot get GPU maxval value: %s\n",
				get_error_string(clstat));
			return 1;
		}
		printf("max value = %f @ index = %d\n", maxval, maxidx);*/
		
                // Check if target GPU kernel has finished.
                if (*gpu.finish == 1) break;
	
		// Again, launch GPU monitoring kernel.
		gpu_monitor<<<1, 1, 1, gpu.stream>>>(gpu.lock);
		custat = cudaGetLastError();
		if (custat != cudaSuccess)
		{
			fprintf(stderr, "Cannot launch monitoring GPU kernel: %s\n",
				get_error_string(clstat));
			return 1;
		}
#ifdef VERBOSE
		istep++;
		printf("step %d\n", istep);
#endif
	}

	clstat = clReleaseMemObject(gpu.data);	
	if (clstat != CL_SUCCESS)
	{
		fprintf(stderr, "Cannot release GPU data buffer: %s\n",
			get_error_string(clstat));
		return 1;
	}
	clstat = clReleaseMemObject(gpu.maxidx);
	if (clstat != CL_SUCCESS)
	{
		fprintf(stderr, "Cannot release GPU maxidx buffer: %s\n",
			get_error_string(clstat));
		return 1;
	}
	clstat = clReleaseMemObject(gpu.maxval);
	if (clstat != CL_SUCCESS)
	{
		fprintf(stderr, "Cannot release GPU maxval buffer: %s\n",
			get_error_string(clstat));
		return 1;
	}
	clstat = clReleaseMemObject(gpu.finish);
	if (clstat != CL_SUCCESS)
	{
		fprintf(stderr, "Cannot release GPU finish buffer: %s\n",
			get_error_string(clstat));
		return 1;
	}
	clstat = clReleaseMemObject(gpu.lock);
	if (clstat != CL_SUCCESS)
	{
		fprintf(stderr, "Cannot release GPU lock buffer: %s\n",
			get_error_string(clstat));
		return 1;
	}

	return 0;
}

