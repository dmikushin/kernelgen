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
			int64_t* size;
			void* monitor_stream =
				kernel->target[runmode].monitor_kernel_stream;
			int err = cuMemAllocHost((void**)&size, sizeof(int64_t));
			if (err) THROW("Error in cuMemAllocHost " << err); 
			err = cuMemcpyDtoHAsync(size, args, sizeof(int64_t), monitor_stream);
			if (err) THROW("Error in cuMemcpyDtoHAsync " << err);
			err = cuStreamSynchronize(monitor_stream);
			if (err) THROW("Error in cuStreamSynchronize " << err);
			char* content;
			err = cuMemAllocHost((void**)&content, *size);
			if (err) THROW("Error in cuMemAllocHost " << err);
			cuMemcpyDtoHAsync(content, args + sizeof(int64_t), *size, monitor_stream);
			if (err) THROW("Error in cuMemcpyDtoHAsync " << err);
			err = cuStreamSynchronize(monitor_stream);
			if (err) THROW("Error in cuStreamSynchronize " << err);
			mhash(td, content, *size);
			err = cuMemFreeHost(content);
			if (err) THROW("Error in cuMemFreeHost " << err);
			err = cuMemFreeHost(size);
			if (err) THROW("Error in cuMemFreeHost " << err);
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
			// If this is the main kernel being lauched,
			// first launch GPU monitor kernel, then launch
			// target kernel. Otherwise - vise versa.
			if (kernel->name != "__kernelgen_main")
			{
				// Launch GPU loop kernel.
				{
					struct { unsigned int x, y, z; } gridDim, blockDim;
					gridDim.x = 1; gridDim.y = 1; gridDim.z = 1;
					blockDim.x = 1; blockDim.y = 1; blockDim.z = 1;
					size_t szshmem = 0;
					void* kernel_func_args[] = { (void*)&args };
					int err = cuLaunchKernel((void*)kernel_func,
						gridDim.x, gridDim.y, gridDim.z,
						blockDim.x, blockDim.y, blockDim.z, szshmem,
						kernel->target[runmode].monitor_kernel_stream,
						kernel_func_args, NULL);
					if (err)
						THROW("Error in cuLaunchKernel " << err);
				}
				
				// Wait for loop kernel completion.
				int err = cuStreamSynchronize(
					kernel->target[runmode].monitor_kernel_stream);
				if (err) THROW("Error in cuStreamSynchronize " << err);
				break;
			}

			// Launch monitor GPU kernel.
			{
				struct { unsigned int x, y, z; } gridDim, blockDim;
				gridDim.x = 1; gridDim.y = 1; gridDim.z = 1;
				blockDim.x = 1; blockDim.y = 1; blockDim.z = 1;
				size_t szshmem = 0;
				void* monitor_kernel_func_args[] =
					{ (void*)&kernel->target[runmode].callback };
				int err = cuLaunchKernel(
					kernel->target[runmode].monitor_kernel_func,
					gridDim.x, gridDim.y, gridDim.z,
					blockDim.x, blockDim.y, blockDim.z, szshmem,
					kernel->target[runmode].monitor_kernel_stream,
					monitor_kernel_func_args, NULL);
				if (err)
					THROW("Error in cuLaunchKernel " << err);
			}
	
			// Launch main GPU kernel.
			{
				struct { unsigned int x, y, z; } gridDim, blockDim;
				gridDim.x = 1; gridDim.y = 1; gridDim.z = 1;
				blockDim.x = 1; blockDim.y = 1; blockDim.z = 1;
				size_t szshmem = 0;
				void* kernel_func_args[] = { (void*)&args };
				int err = cuLaunchKernel((void*)kernel_func,
					gridDim.x, gridDim.y, gridDim.z,
					blockDim.x, blockDim.y, blockDim.z, szshmem,
					kernel->target[runmode].kernel_stream,
					kernel_func_args, NULL);
				if (err)
					THROW("Error in cuLaunchKernel " << err);
			}

			while (1)
			{
				// Wait for monitor kernel completion.
				int err = cuStreamSynchronize(
					kernel->target[runmode].monitor_kernel_stream);
				if (err) THROW("Error in cuStreamSynchronize " << err);

				// Copy callback structure back to host memory and
				// check the state.
				struct kernelgen_callback_t callback;			
				err = cuMemcpyDtoHAsync(
					&callback, kernel->target[runmode].callback,
					sizeof(struct kernelgen_callback_t),
					kernel->target[runmode].monitor_kernel_stream);
				if (err) THROW("Error in cuMemcpyDtoHAsync");
				err = cuStreamSynchronize(
					kernel->target[runmode].monitor_kernel_stream);
				if (err) THROW("Error in cuStreamSynchronize " << err);
				switch (callback.state)
				{
					case KERNELGEN_STATE_INACTIVE :
					{
						if (verbose) 
							cout << "Kernel " << kernel->name <<
								" has finished" << endl;
						break;
					}
					case KERNELGEN_STATE_LOOPCALL :
					{
						if (verbose)
							cout << "Kernel " << kernel->name <<
								" requested loop kernel call " << endl;

						// TODO: handle loop call.

						break;
					}
					case KERNELGEN_STATE_HOSTCALL :
					{
						if (verbose)
							cout << "Kernel " << kernel->name <<
								" requested host function call " << endl;
					
						// TODO: handle host call

						break;
					}
					default :
						THROW("Unknown callback state : " << callback.state);
				}
				
				if (callback.state == KERNELGEN_STATE_INACTIVE) break;

				// Launch monitor GPU kernel.
				{
					struct { unsigned int x, y, z; } gridDim, blockDim;
					gridDim.x = 1; gridDim.y = 1; gridDim.z = 1;
					blockDim.x = 1; blockDim.y = 1; blockDim.z = 1;
					size_t szshmem = 0;
					void* monitor_kernel_func_args[] =
						{ (void*)&kernel->target[runmode].callback };
					int err = cuLaunchKernel(
						kernel->target[runmode].monitor_kernel_func,
						gridDim.x, gridDim.y, gridDim.z,
						blockDim.x, blockDim.y, blockDim.z, szshmem,
						kernel->target[runmode].monitor_kernel_stream,
						monitor_kernel_func_args, NULL);
					if (err)
						THROW("Error in cuLaunchKernel " << err);
				}
			}

			// Finally, sychronize kernel stream.
			int err = cuStreamSynchronize(
				kernel->target[runmode].kernel_stream);
			if (err) THROW("Error in cuStreamSynchronize " << err);
			
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

