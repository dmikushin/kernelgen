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
#include "gforscale_int_opencl.h"

#include <malloc.h>
#include <string.h>
#include <stdio.h>

// Get the device representation for the specified host container
// of device address.
gforscale_status_t gforscale_devaddr_opencl(
	struct gforscale_launch_config_t* l,
	void* host_ptr, size_t offset, void** dev_ptr);

gforscale_status_t gforscale_launch_opencl(
	struct gforscale_launch_config_t* l,
	int* bx, int* ex, int* by, int* ey, int* bz, int* ez)
{
#ifdef HAVE_OPENCL
	struct gforscale_opencl_config_t* opencl =
		(struct gforscale_opencl_config_t*)l->specific;

	// Setup kernel compute grid.
	opencl->threads[0] = 1; opencl->threads[1] = 1; opencl->threads[2] = *ez - *bz + 1;
	opencl->blocks[0] = *ex - *bx + 1; opencl->blocks[1] = *ey - *by + 1; opencl->blocks[2] = 1;

	// Being quiet optimistic initially...
	gforscale_status_t result;
	result.value = CL_SUCCESS;
	result.runmode = l->runmode;

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
			arg->dev_ref = *(void**)(arg->desc);
			void* dataptr = NULL;
			result = gforscale_devaddr_opencl(
				l, arg->mref->mapping, arg->mref->shift, &dataptr);
			if (result.value != CL_SUCCESS)
			{
				goto finish;
			}
			cl_event sync;
			result.value = clEnqueueWriteBuffer(
				opencl->command_queue, arg->mdesc->mapping, CL_FALSE,
				0, sizeof(void*), &dataptr, 0, NULL, &sync);
			if (result.value != CL_SUCCESS)
			{
				gforscale_print_error(gforscale_launch_verbose,
					"Cannot copy allocatable array descriptor, status = %d: %s\n",
					result.value, gforscale_get_error_string(result));
				goto finish;
			}
			result.value = clWaitForEvents(1, &sync);
			if (result.value != CL_SUCCESS)
			{
				gforscale_print_error(gforscale_launch_verbose,
					"Cannot synchronize data copying, status = %d: %s\n",
					result.value, gforscale_get_error_string(result));
				goto finish;
			}
		}

		// Submit argument.
		result.value = clSetKernelArg(opencl->kernel, i, sizeof(void*), &ref);
		if (result.value != CL_SUCCESS)
		{
			gforscale_print_error(gforscale_launch_verbose,
				"Cannot setup kernel argument, status = %d: %s\n",
				result.value, gforscale_get_error_string(result));
			goto finish;
		}
	}

	// Create OpenCL device module symbols container.
	// TODO: do it once during initialization.
	cl_mem modsyms_container;
	if (l->config->nmodsyms)
	{
		// In case of OpenCL all modules symbols are copied
		// into one common container in device memory.
		// Determine its size and create it.
		size_t size = 
			(size_t)l->deps[l->config->nmodsyms - 1].dev_desc +
			l->deps[l->config->nmodsyms - 1].desc_size;
		modsyms_container = clCreateBuffer(opencl->context,
			CL_MEM_READ_WRITE, size, NULL, &result.value);
		if (result.value != CL_SUCCESS)
		{
			gforscale_print_error(gforscale_launch_verbose,
				"Cannot allocate device memory segment of size = %zu on device, status = %d: %s\n",
				size, result.value, gforscale_get_error_string(result));
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
			result = gforscale_devaddr_opencl(
				l, dep->mref->mapping, dep->mref->shift, dataptr);
			if (result.value != CL_SUCCESS)
			{
				goto finish;
			}
		}
		
		// Copy dependency data to device memory.
		cl_event sync;
		result.value = clEnqueueWriteBuffer(
			opencl->command_queue, modsyms_container, CL_FALSE,
			(size_t)dep->dev_desc, dep->desc_size, dep->desc, 0, NULL, &sync);
		if (result.value != CL_SUCCESS)
		{
			gforscale_print_error(gforscale_launch_verbose,
				"Cannot copy data from [%p .. %p] to [%p + %zu .. %p + %zu] for symbol \"%s\", status = %d: %s\n",
				dep->desc, dep->desc + dep->desc_size,
				modsyms_container, (size_t)dep->dev_desc,
				modsyms_container, (size_t)dep->dev_desc + dep->desc_size,
				result.value, gforscale_get_error_string(result));
			goto finish;
		}
		result.value = clWaitForEvents(1, &sync);
		if (result.value != CL_SUCCESS)
		{
			gforscale_print_error(gforscale_launch_verbose,
				"Cannot synchronize data copying from host to device, status = %d: %s\n",
				result.value, gforscale_get_error_string(result));
			goto finish;
		}
	}

	// Submit OpenCL device module symbols container
	// as the last kernel argument.
	if (l->config->nmodsyms)
	{
		result.value = clSetKernelArg(
			opencl->kernel, l->config->nargs, sizeof(cl_mem*), &modsyms_container);
		if (result.value != CL_SUCCESS)
		{
			gforscale_print_error(gforscale_launch_verbose,
				"Cannot setup kernel argument, status = %d: %s\n",
				result.value, gforscale_get_error_string(result));
			goto finish;
		}
	}

	// Launch OpenCL kernel and measure its execution time.
	struct gforscale_time_t start, finish;
	gforscale_get_time(&start);
	for (int i = 0; i < 3; i++)
		opencl->blocks[i] *= opencl->threads[i];
	cl_event sync;
	result.value = clEnqueueNDRangeKernel(
		opencl->command_queue, opencl->kernel, 3,
		NULL, opencl->blocks, opencl->threads, 0, NULL, &sync);
	if (result.value != CL_SUCCESS)
	{
		gforscale_print_error(gforscale_launch_verbose,
			"Cannot launch kernel %s, status = %d: %s\n",
			l->kernel_name, result.value, gforscale_get_error_string(result));
		goto finish;
	}
	result.value = clWaitForEvents(1, &sync);
	if (result.value != CL_SUCCESS)
	{
		gforscale_print_error(gforscale_launch_verbose,
			"Cannot synchronize device running kernel %s, status = %d: %s\n",
			l->kernel_name, result.value, gforscale_get_error_string(result));
		goto finish;
	}
	gforscale_get_time(&finish);
	l->time = gforscale_get_time_diff(&start, &finish);

	// Copy kernel dependencies data from device memory.
	for (int i = 0; i < l->config->nmodsyms; i++)
	{
		struct gforscale_kernel_symbol_t* dep = l->deps + i;

		// Copy dependency data from device memory.
		cl_event sync;
		result.value = clEnqueueReadBuffer(
			opencl->command_queue, modsyms_container, CL_FALSE,
			(size_t)dep->dev_desc, dep->desc_size, dep->desc, 0, NULL, &sync);
		if (result.value != CL_SUCCESS)
		{
			gforscale_print_error(gforscale_launch_verbose,
				"Cannot copy data from [%p + %zu .. %p + %zu] to [%p .. %p] for symbol \"%s\", status = %d: %s\n",
				modsyms_container, (size_t)dep->dev_desc,
				modsyms_container, (size_t)dep->dev_desc + dep->desc_size,
				dep->desc, dep->desc + dep->desc_size,
				result.value, gforscale_get_error_string(result));
			goto finish;
		}
		result.value = clWaitForEvents(1, &sync);
		if (result.value != CL_SUCCESS)
		{
			gforscale_print_error(gforscale_launch_verbose,
				"Cannot synchronize data copying from device to host, status = %d: %s\n",
				result.value, gforscale_get_error_string(result));
			goto finish;
		}
	}

	// Release OpenCL device module symbols container.
	// TODO: do it once during initialization.
	if (l->config->nmodsyms)
	{
		result.value = clReleaseMemObject(modsyms_container);
		if (result.value != CL_SUCCESS)
		{
			gforscale_print_error(gforscale_launch_verbose,
				"Cannot free device module symbols container, status = %d: %s\n",
				result.value, gforscale_get_error_string(result));
			gforscale_set_last_error(result);
		}
	}

finish:
	gforscale_set_last_error(result);
	return result;
#else
	gforscale_status_t result;
	result.value = gforscale_error_not_implemented;
	result.runmode = l->runmode;
	return result;
#endif
}

