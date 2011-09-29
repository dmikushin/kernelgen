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

struct cpu_params_t
{
	float* data;
}
cpu;

struct monitor_params_t
{
	cl_mem lock;
}
monitor;

struct target_params_t
{
	cl_mem data;
	cl_mem maxidx;
	cl_mem maxval;
	cl_mem lock;
	cl_mem finish;
}
target;

int main(int argc, char* argv[])
{
	if (argc != 3)
	{
		printf("%s <size> <npasses>\n", argv[0]);
		return 0;
	}

#ifdef VERBOSE
	builder_config_t* config = builder_init("cross_sync.cl", "-DVERBOSE", 2);
#else
	builder_config_t* config = builder_init("cross_sync.cl", NULL, 2);
#endif
	if (!config) return 1;

	int size = atoi(argv[1]);
	int npasses = atoi(argv[2]);

	cpu.data = (float*)malloc(sizeof(float) * size);
	double dinvrandmax = (double)1.0 / RAND_MAX;
	for (int i = 0; i < size; i++)
		cpu.data[i] = rand() * dinvrandmax;

	cl_int clstat = CL_SUCCESS;
	target.data = clCreateBuffer(config->cpu.context,
		CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
		sizeof(float) * size, cpu.data, &clstat);
	if (clstat != CL_SUCCESS)
	{
		fprintf(stderr, "Cannot create GPU data buffer: %s\n",
			get_error_string(clstat));
		return 1;
	}
	free(cpu.data);

	target.maxidx = clCreateBuffer(config->cpu.context,
		CL_MEM_READ_WRITE, sizeof(int), NULL, &clstat);
	if (clstat != CL_SUCCESS)
	{
		fprintf(stderr, "Cannot create GPU maxidx buffer: %s\n",
			get_error_string(clstat));
		return 1;
	}

	target.maxval = clCreateBuffer(config->cpu.context,
		CL_MEM_READ_WRITE, sizeof(float), NULL, &clstat);
	if (clstat != CL_SUCCESS)
	{
		fprintf(stderr, "Cannot create GPU maxval buffer: %s\n",
			get_error_string(clstat));
		return 1;
	}

	int finish = 0;
	target.finish = clCreateBuffer(config->cpu.context,
		CL_MEM_USE_HOST_PTR,
		sizeof(int), &finish, &clstat);	
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
	int lock = 1;
	monitor.lock = clCreateBuffer(config->gpu.context,
		CL_MEM_USE_HOST_PTR,
		sizeof(int), &lock, &clstat);
	if (clstat != CL_SUCCESS)
	{
		fprintf(stderr, "Cannot create GPU lock buffer: %s\n",
			get_error_string(clstat));
		return 1;
	}
	target.lock = monitor.lock;/* clCreateBuffer(config->cpu.context,
		CL_MEM_ALLOC_HOST_PTR, sizeof(int), NULL, &clstat);
	if (clstat != CL_SUCCESS)
	{
		fprintf(stderr, "Cannot create GPU lock buffer: %s\n",
			get_error_string(clstat));
		return 1;
	}*/

	// Create command queues.
	cl_command_queue monitor_queue = clCreateCommandQueue(
		config->gpu.context, config->gpu.device, 0, &clstat);
	if (clstat != CL_SUCCESS)
	{
		fprintf(stderr, "Cannot create queue for monitor GPU kernel: %s\n",
			get_error_string(clstat));
		return 1;
	}
	cl_command_queue target_queue = clCreateCommandQueue(
		config->cpu.context, config->cpu.device, 0, &clstat);
	if (clstat != CL_SUCCESS)
	{
		fprintf(stderr, "Cannot create queue for target GPU kernel: %s\n",
			get_error_string(clstat));
		return 1;
	}
	
	// Launch GPU monitoring kernel.
	cl_event monitor_event;
	clstat = clSetKernelArg(
		config->gpu.kernels[1], 0, sizeof(cl_mem), &monitor.lock);
	const size_t single = 1;
	clstat = clEnqueueTask(
		monitor_queue, config->gpu.kernels[1],
		0, NULL, &monitor_event);
	if (clstat != CL_SUCCESS)
	{
		fprintf(stderr, "Cannot launch monitoring GPU kernel: %s\n",
			get_error_string(clstat));
		return 1;
	}
	
	// Execute target GPU kernel.
	cl_event target_event;
	clstat = clSetKernelArg(
		config->cpu.kernels[0], 0, sizeof(cl_mem), &target.data);
	clstat = clSetKernelArg(
		config->cpu.kernels[0], 1, sizeof(int), &size);
	clstat = clSetKernelArg(
		config->cpu.kernels[0], 2, sizeof(int), &npasses);
	clstat = clSetKernelArg(
		config->cpu.kernels[0], 3, sizeof(cl_mem), &target.lock);
	clstat = clSetKernelArg(
		config->cpu.kernels[0], 4, sizeof(cl_mem), &target.finish);
	clstat = clSetKernelArg(
		config->cpu.kernels[0], 5, sizeof(cl_mem), &target.maxidx);
	clstat = clSetKernelArg(
		config->cpu.kernels[0], 6, sizeof(cl_mem), &target.maxval);
	clstat = clEnqueueTask(
		target_queue, config->cpu.kernels[0],
		0, NULL, &target_event);
	if (clstat != CL_SUCCESS)
	{
		fprintf(stderr, "Cannot launch target GPU kernel: %s\n",
			get_error_string(clstat));
		return 1;
	}
	clstat = clFlush(target_queue);
	if (clstat != CL_SUCCESS)
	{
		fprintf(stderr, "Cannot flush queue with target GPU kernel: %s\n",
			get_error_string(clstat));
		return 1;
	}
#ifdef VERBOSE
	int istep = 0;
#endif
	while (1)
	{
		// Synchronize with monitoring kernel.
		clstat = clWaitForEvents(1, &monitor_event);
		if (clstat != CL_SUCCESS)
		{
			fprintf(stderr, "Cannot synchronize GPU monitor kernel: %s\n",
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
                if (finish == 1) break;
	
		// Again, launch GPU monitoring kernel.
		clstat = clEnqueueTask(monitor_queue,
			config->gpu.kernels[1], 0, NULL, &monitor_event);
		if (clstat != CL_SUCCESS)
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

	// Synchronize with target kernel.
	clstat = clWaitForEvents(1, &target_event);
	if (clstat != CL_SUCCESS)
	{
		fprintf(stderr, "Cannot synchronize GPU target kernel: %s\n",
			get_error_string(clstat));
		return 1;
	}

	clstat = clReleaseMemObject(target.data);	
	if (clstat != CL_SUCCESS)
	{
		fprintf(stderr, "Cannot release GPU data buffer: %s\n",
			get_error_string(clstat));
		return 1;
	}
	clstat = clReleaseMemObject(target.maxidx);
	if (clstat != CL_SUCCESS)
	{
		fprintf(stderr, "Cannot release GPU maxidx buffer: %s\n",
			get_error_string(clstat));
		return 1;
	}
	clstat = clReleaseMemObject(target.maxval);
	if (clstat != CL_SUCCESS)
	{
		fprintf(stderr, "Cannot release GPU maxval buffer: %s\n",
			get_error_string(clstat));
		return 1;
	}
	clstat = clReleaseMemObject(target.finish);
	if (clstat != CL_SUCCESS)
	{
		fprintf(stderr, "Cannot release GPU finish buffer: %s\n",
			get_error_string(clstat));
		return 1;
	}
	clstat = clReleaseMemObject(target.lock);
	if (clstat != CL_SUCCESS)
	{
		fprintf(stderr, "Cannot release GPU lock buffer: %s\n",
			get_error_string(clstat));
		return 1;
	}

	return 0;
}

