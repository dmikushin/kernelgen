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

#include "kernelgen_int_opencl.h"
#include "init.h"

kernelgen_status_t kernelgen_load_regions_opencl(
	struct kernelgen_launch_config_t* l, int* nmapped)
{
#ifdef HAVE_OPENCL
	struct kernelgen_opencl_config_t* opencl =
		(struct kernelgen_opencl_config_t*)l->specific;

	int count = l->args_nregions + l->deps_nregions;
	struct kernelgen_memory_region_t* regs = l->regs;

	// Being quiet optimistic initially...
	kernelgen_status_t result;
	result.value = CL_SUCCESS;
	result.runmode = l->runmode;

	// The number of successfully mapped memory regions.
	*nmapped = 0;
	
	int iplatform = kernelgen_thread_platform_index;
	int idevice = kernelgen_thread_device_index;
	cl_context context = kernelgen_contexts[iplatform][idevice];
	
	// For each interval pin memory region and
	// map it to device memory.
	for (int i = 0; i < count; i++)
	{
		struct kernelgen_memory_region_t* reg = l->regs + i;
		struct kernelgen_kernel_symbol_t* sym = reg->symbol;
		
		// Pin & map memory region only for regions that
		// are not referring to primary (i.e. reusing other
		// memory region).
		if (!reg->primary)
		{
			// Explicitly create device memory region and copy input data to it.
			reg->mapping = clCreateBuffer(context,
				CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, reg->size,
				reg->base, &result.value);
			if (result.value != CL_SUCCESS)
			{
				kernelgen_print_error(kernelgen_launch_verbose,
					"Cannot allocate device memory segment of size = %zu on device, status = %d: %s\n",
					reg->size, result.value, kernelgen_get_error_string(result));
				goto finish;
			}

			// Increment mapped regions counter
			// (to safely unregister in case of failure).
			(*nmapped)++;

			kernelgen_print_debug(kernelgen_launch_verbose,
				"symbol \"%s\" maps memory segment [%p .. %p] to [%p .. %p + %zu]\n",
				reg->symbol->name, reg->base, reg->base + reg->size, reg->mapping, reg->mapping, reg->size);
		}
		else
		{
			cl_buffer_region subregion;
			subregion.origin = reg->shift;
			subregion.size = reg->symbol->size;
		
			// OpenCL has no direct addressing on device, so it is not
			// possible to apply shift directly to the mapping pointer.
			// Every non-primary region needs a sub-buffer instead.
			reg->mapping = clCreateSubBuffer(reg->primary->mapping, 0,
				CL_BUFFER_CREATE_TYPE_REGION, &subregion, &result.value);
			if (result.value != CL_SUCCESS)
			{
				kernelgen_print_error(kernelgen_launch_verbose,
					"Cannot allocate device memory segment of size = %zu on device, status = %d: %s\n",
					reg->symbol->size, result.value, kernelgen_get_error_string(result));
				goto finish;
			}

			kernelgen_print_debug(kernelgen_launch_verbose,
				"symbol \"%s\" memory segment [%p .. %p] reuses mapping created by symbol \"%s\"\n",
				reg->symbol->name, reg->base + reg->shift, reg->base + reg->shift + reg->symbol->size,
				reg->primary->symbol->name);
		}
	}

finish:

	// If something goes wrong, unmap previously mapped regions.
	if (result.value != CL_SUCCESS)
	{
		kernelgen_save_regions_opencl(l, *nmapped);
		*nmapped = 0;
	}

	kernelgen_set_last_error(result);
	return result;
#else
	kernelgen_status_t result;
	result.value = kernelgen_error_not_implemented;
	result.runmode = l->runmode;
	kernelgen_set_last_error(result);
	return result;
#endif
}

