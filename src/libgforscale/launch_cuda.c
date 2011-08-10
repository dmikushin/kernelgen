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

#include "gforscale_int.h"
#include "gforscale_int_cuda.h"

#include <malloc.h>
#include <string.h>
#include <stdio.h>

gforscale_status_t gforscale_launch_cuda(
	struct gforscale_launch_config_t* l,
	int* bx, int* ex, int* by, int* ey, int* bz, int* ez)
{
#ifdef HAVE_CUDA
	struct gforscale_cuda_config_t* cuda =
		(struct gforscale_cuda_config_t*)l->specific;

	// Setup kernel compute grid.
	cuda->threads.x = 1; cuda->threads.y = 1; cuda->threads.z = *ez - *bz + 1;
	cuda->blocks.x = *ex - *bx + 1; cuda->blocks.y = *ey - *by + 1; cuda->blocks.z = 1;

	gforscale_status_t result;

	// Configure kernel compute grid.
	cudaGetLastError();
	cudaError_t status = cudaConfigureCall(cuda->blocks, cuda->threads, 0, 0);
	if (status != cudaSuccess)
	{
		gforscale_print_error(gforscale_launch_verbose,
			"Cannot configure call to %s, status = %d: %s\n",
			l->kernel_name, status, cudaGetErrorString(status));
		goto finish;
	}

	// Submit arguments to arguments list.
	for (int i = 0; i < l->config->nargs; i++)
	{
		struct gforscale_kernel_symbol_t* arg = l->args + i;
		struct gforscale_memory_region_t* reg = arg->mdesc;

		// Compute target device-mapped address, shifted
		// from page boundary.
		void* ref = reg->mapping + reg->shift;

		// If memory region is for allocatable descriptor,
		// then the corresponding data vector it contains
		// must be replaced with device-mapped and restored
		// back after kernel completion.
		if (arg->allocatable)
		{
#ifdef HAVE_MAPPING
			void** dataptr = (void**)(arg->desc);
			arg->dev_ref = *dataptr;
			*dataptr = arg->mref->mapping + arg->mref->shift;
#else
			arg->dev_ref = *(void**)(arg->desc);
			void* dataptr = arg->mref->mapping + arg->mref->shift;
			status = cudaMemcpy(arg->mdesc->mapping, &dataptr, sizeof(void*), cudaMemcpyHostToDevice);
			if (status != cudaSuccess)
			{
				gforscale_print_error(gforscale_launch_verbose,
					"Cannot copy allocatable array descriptor, status = %d: %s\n",
					status, cudaGetErrorString(status));
				goto finish;
			}
#endif
		}

		// Submit argument.
		cudaGetLastError();
		status = cudaSetupArgument(&ref, sizeof(void*), i * sizeof(void*));
		if (status != cudaSuccess)
		{
			gforscale_print_error(gforscale_launch_verbose,
				"Cannot setup kernel argument, status = %d: %s\n",
				status, cudaGetErrorString(status));
			goto finish;
		}
	}

	// Copy kernel dependencies data to device memory.
	for (int i = 0; i < l->config->nmodsyms; i++)
	{
		struct gforscale_kernel_symbol_t* dep = l->deps + i;
		
		// If memory region is for allocatable descriptor,
		// then the corresponding data vector it contains
		// must be replaced with device-mapped and restored
		// back after kernel completion.
		if (dep->allocatable)
		{		
			// In comparison mode clone dependency descriptor.
			if (l->config->compare)
			{
				// Backup descriptor into shadowed descriptor.
				dep->sdesc = dep->desc;
				
				dep->desc = malloc(dep->desc_size);
				memcpy(dep->desc, dep->sdesc, dep->desc_size);
				*(void**)dep->desc = dep->ref;

				gforscale_print_debug(gforscale_launch_verbose,
					"dep \"%s\" desc = %p, size = %zu duplicated to %p for results comparison\n",
					dep->name, dep->sdesc, dep->desc_size, dep->desc);
			}

			void** dataptr = (void**)(dep->desc);
			dep->dev_ref = *dataptr;
			*dataptr = dep->mref->mapping + dep->mref->shift;
		}
		
		// Copy dependency data to device memory.
		cudaGetLastError();
		status = cudaMemcpy(dep->dev_desc, dep->desc, dep->desc_size,
			cudaMemcpyHostToDevice);
		if (status != cudaSuccess)
		{
			gforscale_print_error(gforscale_launch_verbose,
				"Cannot copy kernel dependency, status = %d: %s\n",
				status, cudaGetErrorString(status));
			goto finish;
		}
	}

	// Launch CUDA kernel and measure its execution time.
	struct gforscale_time_t start, finish;
	gforscale_get_time(&start);
	cudaGetLastError();
	status = cudaLaunch(l->kernel_name);
	if (status != cudaSuccess)
	{
		gforscale_print_error(gforscale_launch_verbose,
			"Cannot launch kernel %s, status = %d: %s\n",
			l->kernel_name, status, cudaGetErrorString(status));
		goto finish;
	}
	cudaGetLastError();
	status = cudaDeviceSynchronize();
	if (status != cudaSuccess)
	{
		gforscale_print_error(gforscale_launch_verbose,
			"Cannot synchronize device running kernel %s, status = %d: %s\n",
			l->kernel_name, status, cudaGetErrorString(status));
		goto finish;
	}
	gforscale_get_time(&finish);
	l->time = gforscale_get_time_diff(&start, &finish);

	// Copy kernel dependencies data from device memory.
	for (int i = 0; i < l->config->nmodsyms; i++)
	{
		struct gforscale_kernel_symbol_t* dep = l->deps + i;

		// Copy dependency data from device memory.
		cudaGetLastError();
		status = cudaMemcpy(dep->desc, dep->dev_desc, dep->desc_size,
			cudaMemcpyDeviceToHost);
		if (status != cudaSuccess)
		{
			gforscale_print_error(gforscale_launch_verbose,
				"Cannot copy kernel dependency data from device, status = %d: %s\n",
				status, cudaGetErrorString(status));
			goto finish;
		}
	}

finish:
	result.value = status;
	result.runmode = l->runmode;
	gforscale_set_last_error(result);
	return result;
#else
	gforscale_status_t result;
	result.value = gforscale_error_not_implemented;
	result.runmode = l->runmode;
	return result;
#endif
}

