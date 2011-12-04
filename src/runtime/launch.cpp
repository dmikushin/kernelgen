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

#include <mhash.h>

using namespace kernelgen;
using namespace kernelgen::bind::cuda;
using namespace kernelgen::runtime;
using namespace std;

// Launch kernel from the specified source code address.
int kernelgen_launch(char* entry, int* args)
{
	kernel_t* kernel = (kernel_t*)entry;
	
	// Initialize hashing engine.
	MHASH td = mhash_init(MHASH_MD5);
	if (td == MHASH_FAILED)
		THROW("Cannot inilialize mhash");
	
	// Compute hash, depending on the runmode.
	switch (runmode)
	{
		case KERNELGEN_RUNMODE_NATIVE :
		{
			int64_t size = *(int64_t*)args;
			char* content = (char*)args + sizeof(int64_t);
			mhash(td, content, size);
			break;
		}
		case KERNELGEN_RUNMODE_CUDA :
		{
			int64_t size;
			cuMemcpyDtoH(&size, args, sizeof(int64_t));
			char* content = (char*)malloc(size);
			cuMemcpyDtoH(content, args + sizeof(int64_t), size);
			mhash(td, content, size);
			free(content);
			break;
		}
		case KERNELGEN_RUNMODE_OPENCL :
		{
			THROW("Unsupported runmode" << runmode);
		}
		default :
			THROW("Unknown runmode " << runmode);
	}
	unsigned char hash[16];	
	mhash_deinit(td, hash);
	if (verbose)
	{
		cout << kernel->name << " @ ";
		for (int i = 0 ; i < 16; i++)
			cout << (int)hash[i];
		cout << endl;
	}
	
	// Check if kernel with the specified hash is
	// already compiled.
	string strhash((char*)hash, 16);
	binaries_map_t& binaries =
		kernel->target[runmode].binaries;
	binaries_map_t::iterator
		binary = binaries.find(strhash);
	char* kernel_func = NULL;
	if (binary == binaries.end())
	{
		if (verbose)
			cout << "No prebuilt kernel, compiling..." << endl;
	
		// Compile kernel for the specified target.
		kernel_func = compile(runmode, kernel);
		binaries[strhash] = kernel_func;
	}
	else
		kernel_func = (*binary).second;
	
	// Execute kernel, depending on target.
	switch (runmode)
	{
		case KERNELGEN_RUNMODE_NATIVE :
		{
			kernel_func_t native_kernel_func =
				(kernel_func_t)kernel_func;
			native_kernel_func(args);
			break;
		}
		case KERNELGEN_RUNMODE_CUDA :
		{
			struct { unsigned int x, y, z; } gridDim, blockDim;
			gridDim.x = 1; gridDim.y = 1; gridDim.z = 1;
			blockDim.x = 1; blockDim.y = 1; blockDim.z = 1;
			size_t szshmem = 0;
			void* kernel_func_args[] = { (void*)&args };
			int err = cuLaunchKernel((void*)kernel_func,
				gridDim.x, gridDim.y, gridDim.z,
				blockDim.x, blockDim.y, blockDim.z, szshmem,
				NULL, kernel_func_args, NULL);
			if (err)
				THROW("Error in cuLaunchKernel " << err);
			err = cuCtxSynchronize();
			if (err)
				THROW("Error in cuCtxSynchronize " << err);
			break;
		}
		case KERNELGEN_RUNMODE_OPENCL :
		{
			// TODO: Launch kernel using OpenCL API
			THROW("Unsupported runmode" << runmode);
			break;
		}
		default :
			THROW("Unknown runmode " << runmode);
	}

	return 0;
}

