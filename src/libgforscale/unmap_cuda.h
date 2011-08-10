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

gforscale_status_t gforscale_save_regions_cuda(
	struct gforscale_launch_config_t* l, int nmapped)
{
#ifdef HAVE_CUDA
	int count = l->args_nregions + l->deps_nregions;
	struct gforscale_memory_region_t* regs = l->regs;

	gforscale_status_t result;
	result.runmode = l->runmode;

	// Unregister pinned memory regions.
	for (int i = 0; i < nmapped; i++)
	{
		struct gforscale_memory_region_t* reg = l->regs + i;
		
		if (!reg->primary)
		{
#ifdef HAVE_MAPPING
			cudaGetLastError();
			int status = cudaHostUnregister(reg->base);
			if (status != cudaSuccess)
			{
				gforscale_print_error(gforscale_launch_verbose,
					"Cannot unregister host memory segment [%p .. %p] for symbol \"%s\", status = %d: %s\n",
					reg->base, reg->base + reg->size, reg->symbol->name,
					status, cudaGetErrorString(status));
				result.value = status;
				gforscale_set_last_error(result);
			}
#else
			// Explicitly copy output data from device memory region and free it.
			cudaGetLastError();
			int status = cudaMemcpy(reg->base, reg->mapping, reg->size, cudaMemcpyDeviceToHost);
			if (status != cudaSuccess)
			{
				gforscale_print_error(gforscale_launch_verbose,
					"Cannot copy data from [%p .. %p] to [%p .. %p] for symbol \"%s\", status = %d: %s\n",
					reg->mapping, reg->base + reg->size, reg->base, reg->mapping + reg->size,
					status, cudaGetErrorString(status));
				result.value = status;
				gforscale_set_last_error(result);
			}
			cudaGetLastError();
			status = cudaFree(reg->mapping);
			if (status != cudaSuccess)
			{
				gforscale_print_error(gforscale_launch_verbose,
					"Cannot free device memory segment [%p .. %p] for symbol \"%s\", status = %d: %s\n",
					reg->base, reg->base + reg->size, reg->symbol->name,
					status, cudaGetErrorString(status));
				result.value = status;
				gforscale_set_last_error(result);
			}
#endif
		}
	}
	result.value = gforscale_success;
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

