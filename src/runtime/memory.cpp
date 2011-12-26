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

#include "bind.h"
#include "runtime.h"
#include "util.h"

#include "kernelgen_interop.h"

using namespace kernelgen::bind::cuda;
using namespace kernelgen::runtime;
using namespace std;

/*
 * This file contains device-side functions to support the dynamic
 * memory heap in GPU global memory. The rationale is to replace
 * builtin malloc/free calls, since they are incompatible with
 * concurrent kernels execution.
 */

// Setup the device global memory pool initial configuration.
kernelgen_memory_t* kernelgen::runtime::init_memory_pool(size_t szpool)
{
	// First, fill config on host.
	kernelgen_memory_t config_host;

	// Allocate pool and flush it to zero.
	int err = cuMemAlloc((void**)&config_host.pool, szpool);
	if (err) THROW("Error in cuMemAlloc: " << err);
	err = cuMemsetD8(config_host.pool, 0, szpool);
	if (err) THROW("Error in cuMemsetD8: " << err);

	config_host.szused = 0;
	config_host.szpool = szpool;
	config_host.count = 0;

	// Copy the resulting config to the special
	// device variable.
	kernelgen_memory_t* config_device = NULL;
	err = cuMemAlloc((void**)&config_device, szpool);
	if (err) THROW("Error in cuMemAlloc: " << err);
	err = cuMemcpyHtoD(config_device, &config_host, sizeof(kernelgen_memory_t));
	if (err) THROW("Error in cuMemcpyH2D: " << err);
	return config_device;
}

