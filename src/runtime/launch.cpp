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

#include "llvm/Function.h"
#include "llvm/ExecutionEngine/GenericValue.h"
#include "llvm/Target/TargetData.h"

using namespace kernelgen;
using namespace kernelgen::bind::cuda;
using namespace kernelgen::runtime;
using namespace llvm;
using namespace std;

// Launch the specified kernel.
int kernelgen_launch(kernel_t* kernel,
	unsigned long long szdata, unsigned long long szdatai,
	kernelgen_callback_data_t* data)
{
	if (!kernel->target[runmode].supported)
		return -1;

	if (verbose)
		cout << "Kernel function call " << kernel->name << endl;

	// Lookup for kernel in table, only if it has at least
	// one scalar to compute hash footprint. Otherwise, compile
	// "generalized" kernel.
	kernel_func_t kernel_func =
		kernel->target[runmode].binary;
	if (szdatai && (kernel->name != "__kernelgen_main"))
	{
		// Initialize hashing engine.
		MHASH td = mhash_init(MHASH_MD5);
		if (td == MHASH_FAILED)
			THROW("Cannot inilialize mhash");
	
		// Compute hash, depending on the runmode.
		switch (runmode)
		{
			case KERNELGEN_RUNMODE_NATIVE :
			{
				mhash(td, &data->args, szdatai);
				break;
			}
			case KERNELGEN_RUNMODE_CUDA :
			{
				void* monitor_stream =
					kernel->target[runmode].monitor_kernel_stream;
				char* content = (char*)malloc(szdatai);
				//int err = cuMemAllocHost((void**)&content, szdatai);
				//if (err) THROW("Error in cuMemAllocHost " << err);
				int err = cuMemcpyDtoHAsync(content, &data->args, szdatai, monitor_stream);
				if (err) THROW("Error in cuMemcpyDtoHAsync " << err);
				err = cuStreamSynchronize(monitor_stream);
				if (err) THROW("Error in cuStreamSynchronize " << err);
				mhash(td, content, szdatai);
				//err = cuMemFreeHost(content);
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
	}
	else
	{
		// Compile and store universal binary.
		if (!kernel_func)
		{
			if (verbose)
				cout << "No prebuilt kernel, compiling..." << endl;

			kernel_func = compile(runmode, kernel);		
			if (!kernel_func) return -1;
			kernel->target[runmode].binary = kernel_func;
		}
	}
	
	// Execute kernel, depending on target.
	switch (runmode)
	{
		case KERNELGEN_RUNMODE_NATIVE :
		{
			kernel_func_t native_kernel_func =
				(kernel_func_t)kernel_func;
			native_kernel_func(data);
			break;
		}
		case KERNELGEN_RUNMODE_CUDA :
		{
			// If this is the main kernel being lauched,
			// first launch GPU monitor kernel, then launch
			// target kernel. Otherwise - vise versa.
			if (kernel->name != "__kernelgen_main")
			{
				// Launch GPU loop kernel, if it is compiled.
				{
					struct { unsigned int x, y, z; } gridDim, blockDim;
					gridDim.x = 1; gridDim.y = 1; gridDim.z = 1;
					blockDim.x = 1; blockDim.y = 1; blockDim.z = 1;
					size_t szshmem = 0;
					void* kernel_func_args[] = { (void*)&data };
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

			// Create host-pinned callback structure buffer.
			struct kernelgen_callback_t* callback =
				(struct kernelgen_callback_t*)malloc(
					sizeof(struct kernelgen_callback_t));
			//int err = cuMemAllocHost((void**)&callback, sizeof(struct kernelgen_callback_t));
			//if (err) THROW("Error in cuMemAllocHost " << err);

			// Launch monitor GPU kernel.
			{
				struct { unsigned int x, y, z; } gridDim, blockDim;
				gridDim.x = 1; gridDim.y = 1; gridDim.z = 1;
				blockDim.x = 1; blockDim.y = 1; blockDim.z = 1;
				size_t szshmem = 0;
				void* monitor_kernel_func_args[] =
					{ (void*)&kernel->target[runmode].callback };
				int err = cuLaunchKernel(
					(void*)kernel->target[runmode].monitor_kernel_func,
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
				void* kernel_func_args[] = { (void*)&data };
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
				err = cuMemcpyDtoHAsync(
					callback, kernel->target[runmode].callback,
					sizeof(struct kernelgen_callback_t),
					kernel->target[runmode].monitor_kernel_stream);
				if (err) THROW("Error in cuMemcpyDtoHAsync");
				err = cuStreamSynchronize(
					kernel->target[runmode].monitor_kernel_stream);
				if (err) THROW("Error in cuStreamSynchronize " << err);
				switch (callback->state)
				{
					case KERNELGEN_STATE_INACTIVE :
					{
						if (verbose) 
							cout << "Kernel " << kernel->name << " has finished" << endl;
						break;
					}
					case KERNELGEN_STATE_LOOPCALL :
					{
						// Launch the loop kernel.
						callback->kernel->target[runmode].monitor_kernel_stream =
							kernel->target[runmode].monitor_kernel_stream;
						if (kernelgen_launch(callback->kernel, callback->szdata,
							callback->szdatai, callback->data) != -1)
							break;

						// If kernel cannot be launched, fallback to
						// case KERNELGEN_STATE_HOSTCALL
					}
					case KERNELGEN_STATE_HOSTCALL :
					{
						kernelgen_hostcall(callback->kernel, callback->szdata,
							callback->szdatai, callback->data);					
						break;
					}
					default :
						THROW("Unknown callback state : " << callback->state);
				}
				
				if (callback->state == KERNELGEN_STATE_INACTIVE) break;

				// Launch monitor GPU kernel.
				{
					struct { unsigned int x, y, z; } gridDim, blockDim;
					gridDim.x = 1; gridDim.y = 1; gridDim.z = 1;
					blockDim.x = 1; blockDim.y = 1; blockDim.z = 1;
					size_t szshmem = 0;
					void* monitor_kernel_func_args[] =
						{ (void*)&kernel->target[runmode].callback };
					int err = cuLaunchKernel(
						(void*)kernel->target[runmode].monitor_kernel_func,
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
			
			//err = cuMemFreeHost(callback);
			//if (err) THROW("Error in cuMemFreeHost " << err);
			
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

