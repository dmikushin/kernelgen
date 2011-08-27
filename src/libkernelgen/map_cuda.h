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

kernelgen_status_t kernelgen_load_regions_cuda(
	struct kernelgen_launch_config_t* l, int* nmapped)
{
#ifdef HAVE_CUDA
	int count = l->args_nregions + l->deps_nregions;
	struct kernelgen_memory_region_t* regs = l->regs;

	// Being quiet optimistic initially...
	cudaError_t status = cudaSuccess;

	// The number of successfully mapped memory regions.
	*nmapped = 0;
	
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
			// Register host memory segment incorporating entire kernel argument
			// to be shared with device.
#if 0
			cudaGetLastError();
			status = cudaHostRegister(reg->base, reg->size, cudaHostRegisterMapped);
			if (status != cudaSuccess)
			{
				kernelgen_print_error(kernelgen_launch_verbose,
					"Cannot register memory segment [%p .. %p] for symbol \"%s\", status = %d: %s\n",
					reg->base, reg->base + reg->size, reg->symbol->name, status, cudaGetErrorString(status));
				goto finish;
			}
#endif
#ifdef HAVE_MAPPING
			// Increment mapped regions counter
			// (to safely unregister in case of failure).
			(*nmapped)++;

			// Get device memory pointer for the previously shared segment.
			cudaGetLastError();
			status = cudaHostGetDevicePointer((void**)&reg->mapping, reg->base, 0);
			if (status != cudaSuccess)
			{
				kernelgen_print_error(kernelgen_launch_verbose,
					"Cannot map memory segment [%p .. %p] for symbol \"%s\", status = %d: %s\n",
					reg->base, reg->base + reg->size, reg->symbol->name, status, cudaGetErrorString(status));
				goto finish;
			}
#else
			// Explicitly create device memory region and copy input data to it.
			cudaGetLastError();
			status = cudaMalloc((void**)&reg->mapping, reg->size);
			if (status != cudaSuccess)
			{
				kernelgen_print_error(kernelgen_launch_verbose,
					"Cannot allocate device memory segment of size = %zu on device, status = %d: %s\n",
					reg->size, status, cudaGetErrorString(status));
				goto finish;
			}

			// Increment mapped regions counter
			// (to safely unregister in case of failure).
			(*nmapped)++;
			
			cudaGetLastError();
			status = cudaMemcpy(reg->mapping, reg->base, reg->size, cudaMemcpyHostToDevice);
			if (status != cudaSuccess)
			{
				kernelgen_print_error(kernelgen_launch_verbose,
					"Cannot copy data from [%p .. %p] to [%p .. %p] for symbol \"%s\", status = %d: %s\n",
					reg->base, reg->base + reg->size, reg->mapping, reg->mapping + reg->size,
					status, cudaGetErrorString(status));
				goto finish;
			}
#endif
			kernelgen_print_debug(kernelgen_launch_verbose,
				"symbol \"%s\" maps memory segment [%p .. %p] to [%p .. %p]\n",
				reg->symbol->name, reg->base, reg->base + reg->size, reg->mapping, reg->mapping + reg->size);
		}
		else
		{
			reg->mapping = reg->primary->mapping;
			kernelgen_print_debug(kernelgen_launch_verbose,
				"symbol \"%s\" memory segment [%p .. %p] reuses mapping created by symbol \"%s\"\n",
				reg->symbol->name, reg->base, reg->base + reg->size, reg->primary->symbol->name);
		}
	}

finish:

	// If something goes wrong, unmap previously mapped regions.
	if (status != cudaSuccess)
	{
		kernelgen_save_regions_cuda(l, *nmapped);
		*nmapped = 0;
	}

	kernelgen_status_t result;
	result.value = status;
	result.runmode = l->runmode;
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

