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
#include "util.h"

#include <stddef.h>

using namespace kernelgen;
using namespace kernelgen::bind::cuda;
using namespace std;

namespace kernelgen { namespace bind { namespace cuda {

	cuInit_t cuInit = NULL;
	cuDeviceGet_t cuDeviceGet = NULL;
	cuCtxCreate_t cuCtxCreate = NULL;
	cuCtxSynchronize_t cuCtxSynchronize = NULL;
	cuMemAlloc_t cuMemAlloc = NULL;
	cuMemFree_t cuMemFree = NULL;
	cuMemAlloc_t cuMemAllocHost = NULL;
	cuMemFree_t cuMemFreeHost = NULL;
	cuMemcpy_t cuMemcpyHtoD = NULL, cuMemcpyDtoH = NULL;
	cuMemcpyAsync_t cuMemcpyHtoDAsync = NULL, cuMemcpyDtoHAsync = NULL;
	cuModuleLoad_t cuModuleLoad = NULL;
	cuModuleLoad_t cuModuleLoadData = NULL;
	cuModuleLoadDataEx_t cuModuleLoadDataEx = NULL;
	cuModuleGetFunction_t cuModuleGetFunction = NULL;
	cuLaunchKernel_t cuLaunchKernel = NULL;
	cuStreamCreate_t cuStreamCreate = NULL;
	cuStreamSynchronize_t cuStreamSynchronize = NULL;

	void init()
	{
		// Load CUDA Driver API shared library.
		void* handle = dlopen("libcuda.so",
			RTLD_NOW | RTLD_GLOBAL | RTLD_DEEPBIND);
		if (!handle)
			THROW("Cannot dlopen libcuda.so " << dlerror());

		// Load functions.
		cuInit = (cuInit_t)dlsym(handle, "cuInit");
		if (!cuInit)
			THROW("Cannot dlsym cuInit " << dlerror());
		cuDeviceGet = (cuDeviceGet_t)dlsym(handle, "cuDeviceGet");
		if (!cuDeviceGet)
			THROW("Cannot dlsym cuDeviceGet " << dlerror());
		cuCtxCreate = (cuCtxCreate_t)dlsym(handle, "cuCtxCreate");
		if (!cuCtxCreate)
			THROW("Cannot dlsym cuCtxCreate " << dlerror());
		cuCtxSynchronize = (cuCtxSynchronize_t)dlsym(handle, "cuCtxSynchronize");
		if (!cuCtxSynchronize)
			THROW("Cannot dlsym cuCtxSynchronize " << dlerror());
		cuMemAlloc = (cuMemAlloc_t)dlsym(handle, "cuMemAlloc");
		if (!cuMemAlloc)
			THROW("Cannot dlsym cuMemAlloc " << dlerror());
		cuMemFree = (cuMemFree_t)dlsym(handle, "cuMemFree");
		if (!cuMemFree)
			THROW("Cannot dlsym cuMemFree " << dlerror());
		cuMemAllocHost = (cuMemAlloc_t)dlsym(handle, "cuMemAllocHost");
		if (!cuMemAllocHost)
			THROW("Cannot dlsym cuMemAllocHost " << dlerror());
		cuMemFreeHost = (cuMemFree_t)dlsym(handle, "cuMemFreeHost");
		if (!cuMemFreeHost)
			THROW("Cannot dlsym cuMemFreeHost " << dlerror());
		cuMemcpyHtoD = (cuMemcpy_t)dlsym(handle, "cuMemcpyHtoD");
		if (!cuMemcpyHtoD)
			THROW("Cannot dlsym cuMemcpyHtoD " << dlerror());
		cuMemcpyDtoH = (cuMemcpy_t)dlsym(handle, "cuMemcpyDtoH");
		if (!cuMemcpyDtoH)
			THROW("Cannot dlsym cuMemcpyDtoH " << dlerror());
		cuMemcpyHtoDAsync = (cuMemcpyAsync_t)dlsym(handle, "cuMemcpyHtoDAsync");
		if (!cuMemcpyHtoDAsync)
			THROW("Cannot dlsym cuMemcpyHtoDAsync " << dlerror());
		cuMemcpyDtoHAsync = (cuMemcpyAsync_t)dlsym(handle, "cuMemcpyDtoHAsync");
		if (!cuMemcpyDtoHAsync)
			THROW("Cannot dlsym cuMemcpyDtoHAsync " << dlerror());
		cuModuleLoad = (cuModuleLoad_t)dlsym(handle, "cuModuleLoad");
		if (!cuModuleLoad)
			THROW("Cannot dlsym cuModuleLoad " << dlerror());
		cuModuleLoadData = (cuModuleLoad_t)dlsym(handle, "cuModuleLoadData");
		if (!cuModuleLoadData)
			THROW("Cannot dlsym cuModuleLoadData " << dlerror());
		cuModuleLoadDataEx = (cuModuleLoadDataEx_t)dlsym(handle, "cuModuleLoadDataEx");
		if (!cuModuleLoadDataEx)
			THROW("Cannot dlsym cuModuleLoadDataEx " << dlerror());
		cuModuleGetFunction = (cuModuleGetFunction_t)dlsym(handle, "cuModuleGetFunction");
		if (!cuModuleGetFunction)
			THROW("Cannot dlsym cuModuleGetFunction " << dlerror());
		cuLaunchKernel = (cuLaunchKernel_t)dlsym(handle, "cuLaunchKernel");
		if (!cuLaunchKernel)
			THROW("Cannot dlsym cuLaunchKernel " << dlerror());
		cuStreamCreate = (cuStreamCreate_t)dlsym(handle, "cuStreamCreate");
		if (!cuStreamCreate)
			THROW("Cannot dlsym cuStreamCreate " << dlerror());
		cuStreamSynchronize = (cuStreamSynchronize_t)dlsym(handle, "cuStreamSynchronize");
		if (!cuStreamSynchronize)
			THROW("Cannot dlsym cuStreamSynchronize " << dlerror());

		int err = cuInit(0);
		if (err)
			THROW("Error in cuInit " << err);
		
		int device;
		err = cuDeviceGet(&device, 0);
		if (err)
			THROW("Error in cuDeviceGet " << err);
		
		int context;
		err = cuCtxCreate(&context, 0, device);
		if (err)
			THROW("Error in cuCtxCreate " << err);
	}

}}}

