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
#include "kernelgen_int_opencl.h"

kernelgen_status_t kernelgen_save_regions_opencl(
	struct kernelgen_launch_config_t* l, int nmapped)
{
#ifdef HAVE_OPENCL
	struct kernelgen_opencl_config_t* opencl =
		(struct kernelgen_opencl_config_t*)l->specific;

	int count = l->args_nregions + l->deps_nregions;
	struct kernelgen_memory_region_t* regs = l->regs;

	kernelgen_status_t result;
	result.value = CL_SUCCESS;
	result.runmode = l->runmode;

	// Unregister pinned memory regions.
	for (int i = 0; i < nmapped; i++)
	{
		struct kernelgen_memory_region_t* reg = l->regs + i;
		
		if (!reg->primary)
		{
			// Explicitly copy output data from device memory region and free it.
			cl_event sync;
			result.value = clEnqueueReadBuffer(
				opencl->command_queue, reg->mapping, CL_FALSE,
				0, reg->size, reg->base, 0, NULL, &sync);
			if (result.value != CL_SUCCESS)
			{
				kernelgen_print_error(kernelgen_launch_verbose,
					"Cannot copy data from [%p .. %p] to [%p .. %p] for symbol \"%s\", status = %d: %s\n",
					reg->mapping, reg->base + reg->size, reg->base, reg->mapping + reg->size,
					result.value, kernelgen_get_error_string(result));
				kernelgen_set_last_error(result);
			}
			result.value = clWaitForEvents(1, &sync);
			if (result.value != CL_SUCCESS)
			{
				kernelgen_print_error(kernelgen_launch_verbose,
					"Cannot synchronize data copying from device to host, status = %d: %s\n",
					result.value, kernelgen_get_error_string(result));
				kernelgen_set_last_error(result);
			}
			result.value = clReleaseMemObject(reg->mapping);
			if (result.value != CL_SUCCESS)
			{
				kernelgen_print_error(kernelgen_launch_verbose,
					"Cannot free device memory segment [%p .. %p] for symbol \"%s\", status = %d: %s\n",
					reg->base, reg->base + reg->size, reg->symbol->name,
					result.value, kernelgen_get_error_string(result));
				kernelgen_set_last_error(result);
			}
		}
	}
	return result;
#else
	kernelgen_status_t result;
	result.value = kernelgen_error_not_implemented;
	result.runmode = l->runmode;
	kernelgen_set_last_error(result);
	return result;
#endif
}

