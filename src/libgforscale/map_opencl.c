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

#include "gforscale_int_opencl.h"
#include "init.h"

gforscale_status_t gforscale_load_regions_opencl(
	struct gforscale_launch_config_t* l, int* nmapped)
{
#ifdef HAVE_OPENCL
	struct gforscale_opencl_config_t* opencl =
		(struct gforscale_opencl_config_t*)l->specific;

	int count = l->args_nregions + l->deps_nregions;
	struct gforscale_memory_region_t* regs = l->regs;

	// Being quiet optimistic initially...
	gforscale_status_t result;
	result.value = CL_SUCCESS;
	result.runmode = l->runmode;

	// The number of successfully mapped memory regions.
	*nmapped = 0;
	
	// For each interval pin memory region and
	// map it to device memory.
	for (int i = 0; i < count; i++)
	{
		struct gforscale_memory_region_t* reg = l->regs + i;
		struct gforscale_kernel_symbol_t* sym = reg->symbol;
		
		// Pin & map memory region only for regions that
		// are not referring to primary (i.e. reusing other
		// memory region).
		if (!reg->primary)
		{
			// Explicitly create device memory region and copy input data to it.
			reg->mapping = clCreateBuffer(opencl->context,
				CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, reg->size,
				reg->base, &result.value);
			if (result.value != CL_SUCCESS)
			{
				gforscale_print_error(gforscale_launch_verbose,
					"Cannot allocate device memory segment of size = %zu on device, status = %d: %s\n",
					reg->size, result.value, gforscale_get_error_string(result));
				goto finish;
			}

			// Increment mapped regions counter
			// (to safely unregister in case of failure).
			(*nmapped)++;

			gforscale_print_debug(gforscale_launch_verbose,
				"symbol \"%s\" maps memory segment [%p .. %p] to [%p .. %p]\n",
				reg->symbol->name, reg->base, reg->base + reg->size, reg->mapping, reg->mapping + reg->size);
		}
		else
		{
			reg->mapping = reg->primary->mapping;
			gforscale_print_debug(gforscale_launch_verbose,
				"symbol \"%s\" memory segment [%p .. %p] reuses mapping created by symbol \"%s\"\n",
				reg->symbol->name, reg->base, reg->base + reg->size, reg->primary->symbol->name);
		}
	}

finish:

	// If something goes wrong, unmap previously mapped regions.
	if (result.value != CL_SUCCESS)
	{
		gforscale_save_regions_opencl(l, *nmapped);
		*nmapped = 0;
	}

	gforscale_set_last_error(result);
	return result;
#else
	gforscale_status_t result;
	result.value = gforscale_error_not_implemented;
	result.runmode = l->runmode;
	gforscale_set_last_error(result);
	return result;
#endif
}

