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

long gforscale_reset_verbose = 1 << 2;

gforscale_status_t gforscale_reset_cuda(
	struct gforscale_launch_config_t* l)
{
#ifdef HAVE_CUDA
	cudaError_t status = cudaSuccess;

	// Until there no is straight algorithm of recovering
	// from kernel failure, in particular cleaning error
	// status, the only way to cleanup is device reset.
	cudaGetLastError();
	status = cudaDeviceReset();
	if (status != cudaSuccess)
	{
		gforscale_print_error(gforscale_reset_verbose,
			"Cannot reset device, status = %d: %s\n",
			status, cudaGetErrorString(status));
	}

	gforscale_status_t result;
	result.value = status;
	result.runmode = l->runmode;
	gforscale_set_last_error(result);
	return result;
#else
	gforscale_status_t result;
	result.value = gforscale_success;
	result.runmode = l->runmode;
	return result;
#endif
}

