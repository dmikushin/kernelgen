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

#include "kernelgen_int.h"
#include "kernelgen_int_cuda.h"
#include "stats.h"

#include <malloc.h>
#include <string.h>
#include <stdio.h>

kernelgen_status_t kernelgen_launch_cuda(
	struct kernelgen_launch_config_t* l,
	int* bx, int* ex, int* by, int* ey, int* bz, int* ez)
{
#ifdef HAVE_CUDA
	struct kernelgen_cuda_config_t* cuda =
		(struct kernelgen_cuda_config_t*)l->specific;

	// Setup kernel compute grid.
	dim3 threads, blocks;
	threads.x = 1; threads.y = 1; threads.z = *ez - *bz + 1;
	blocks.x = *ex - *bx + 1; blocks.y = *ey - *by + 1; blocks.z = 1;

	kernelgen_status_t result;

	// Configure kernel compute grid.
	cudaGetLastError();
	cudaError_t status = cudaConfigureCall(blocks, threads, 0, 0);
	if (status != cudaSuccess)
	{
		kernelgen_print_error(kernelgen_launch_verbose,
			"Cannot configure call to %s, status = %d: %s\n",
			l->kernel_name, status, cudaGetErrorString(status));
		goto finish;
	}

	// Submit arguments to arguments list.
	for (int i = 0; i < l->config->nargs; i++)
	{
		struct kernelgen_kernel_symbol_t* arg = l->args + i;
		struct kernelgen_memory_region_t* reg = arg->mdesc;

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
				kernelgen_print_error(kernelgen_launch_verbose,
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
			kernelgen_print_error(kernelgen_launch_verbose,
				"Cannot setup kernel argument, status = %d: %s\n",
				status, cudaGetErrorString(status));
			goto finish;
		}
	}

	// Copy kernel dependencies data to device memory.
	for (int i = 0; i < l->config->nmodsyms; i++)
	{
		struct kernelgen_kernel_symbol_t* dep = l->deps + i;

		// If symbol is defined on device
		if (dep->dev_desc)
		{
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

					kernelgen_print_debug(kernelgen_launch_verbose,
						"dep \"%s\" desc = %p, size = %zu duplicated to %p for results comparison\n",
						dep->name, dep->sdesc, dep->desc_size, dep->desc);
				}

				// Replace host data array reference in descriptor
				// with its clone in device memory.
				void** dataptr = (void**)(dep->desc);
				*dataptr = dep->mref->mapping + dep->mref->shift;
			}
			
			// Copy dependency data to device memory.
			cudaGetLastError();
			status = cudaMemcpy(dep->dev_desc, dep->desc, dep->desc_size,
				cudaMemcpyHostToDevice);
			if (status != cudaSuccess)
			{
				kernelgen_print_error(kernelgen_launch_verbose,
					"Cannot copy kernel dependency, status = %d: %s\n",
					status, cudaGetErrorString(status));
				goto finish;
			}
		}
	}

	// Launch CUDA kernel and measure its execution time.
	kernelgen_record_time_start(l->stats);
	cudaGetLastError();
	status = cudaLaunch(l->kernel_name);
	if (status != cudaSuccess)
	{
		kernelgen_print_error(kernelgen_launch_verbose,
			"Cannot launch kernel %s, status = %d: %s\n",
			l->kernel_name, status, cudaGetErrorString(status));
		goto finish;
	}
	cudaGetLastError();
	status = cudaDeviceSynchronize();
	if (status != cudaSuccess)
	{
		kernelgen_print_error(kernelgen_launch_verbose,
			"Cannot synchronize device running kernel %s, status = %d: %s\n",
			l->kernel_name, status, cudaGetErrorString(status));
		goto finish;
	}
	kernelgen_record_time_finish(l->stats);

	// Copy kernel dependencies data from device memory.
	for (int i = 0; i < l->config->nmodsyms; i++)
	{
		struct kernelgen_kernel_symbol_t* dep = l->deps + i;
		
		// If symbol is defined on device.
		if (dep->dev_desc)
		{
			// Copy dependency data from device memory.
			cudaGetLastError();
			status = cudaMemcpy(dep->desc, dep->dev_desc, dep->desc_size,
				cudaMemcpyDeviceToHost);
			if (status != cudaSuccess)
			{
				kernelgen_print_error(kernelgen_launch_verbose,
					"Cannot copy kernel dependency data from device, status = %d: %s\n",
					status, cudaGetErrorString(status));
				goto finish;
			}
		}
	}

finish:
	result.value = status;
	result.runmode = l->runmode;
	kernelgen_set_last_error(result);
	return result;
#else
	kernelgen_status_t result;
	result.value = kernelgen_error_not_implemented;
	result.runmode = l->runmode;
	return result;
#endif
}

