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

#include "runtime.h"
#include "util.h"

using namespace kernelgen;
using namespace kernelgen::runtime;
using namespace std;

// Launch kernel from the specified source code address.
int kernelgen_launch(char* entry, int nargs, int* szargs, ...)
{
	kernel_t* kernel = (kernel_t*)entry;
	
	// Load args values into contiguous array.

	// Compute args array hash.
	
	// Check if kernel with the specified hash is
	// already compiled.
	bool compiled = false;
	if (!compiled)
	{
		// Compile kernel for the specified target.
		va_list list;
		va_start(list, szargs);
		compile(runmode, kernel, nargs, szargs, list);
		va_end(list);
	}
	
	// Execute kernel, depending on target.
	switch (runmode)
	{
		case KERNELGEN_RUNMODE_NATIVE :
		{
			// TODO: Launch kernel using FFI
			break;
		}
		case KERNELGEN_RUNMODE_CUDA :
		{
			// TODO: Launch kernel using CUDA Driver API
			break;
		}
		case KERNELGEN_RUNMODE_OPENCL :
		{
			// TODO: Launch kernel using OpenCL API
			break;
		}
		default :
			THROW("Unknown runmode " << runmode);
	}

	return 0;
}

