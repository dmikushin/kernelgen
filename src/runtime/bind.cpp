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

#include <dlfcn.h>
#include <stddef.h>

using namespace kernelgen;
using namespace std;

namespace kernelgen { namespace bind { namespace cuda {

cuInit_t cuInit;
cuDeviceGet_t cuDeviceGet;
cuCtxCreate_t cuCtxCreate;
cuCtxSynchronize_t cuCtxSynchronize;
cuMemAlloc_t cuMemAlloc;
cuMemFree_t cuMemFree;
cuMemAlloc_t cuMemAllocHost;
cuMemFree_t cuMemFreeHost;
cuMemcpy_t cuMemcpyHtoD, cuMemcpyDtoH;
cuMemcpyAsync_t cuMemcpyHtoDAsync, cuMemcpyDtoHAsync;
cuMemGetAddressRange_t cuMemGetAddressRange;
cuMemsetD8_t cuMemsetD8;
cuMemsetD32_t cuMemsetD32;
cuMemsetD32Async_t cuMemsetD32Async;
cuMemHostRegister_t cuMemHostRegister;
cuMemHostUnregister_t cuMemHostUnregister;
cuModuleLoad_t cuModuleLoad;
cuModuleLoad_t cuModuleLoadData;
cuModuleLoadDataEx_t cuModuleLoadDataEx;
cuModuleUnload_t cuModuleUnload;
cuModuleGetFunction_t cuModuleGetFunction;
cuModuleGetGlobal_t cuModuleGetGlobal;
cuLaunchKernel_t cuLaunchKernel;
cuStreamCreate_t cuStreamCreate;
cuStreamSynchronize_t cuStreamSynchronize;

const context context::init(int capacity)
{
	// Do not init again, if already bound.
	if (cuInit) context(NULL, capacity);

	// Load CUDA Driver API shared library.
	void* handle = dlopen("libcuda.so",
		RTLD_NOW | RTLD_GLOBAL | RTLD_DEEPBIND);
	if (!handle)
		THROW("Cannot dlopen libcuda.so " << dlerror());
	
	return context(handle, capacity);
}

context::context(void* handle, int capacity) :

handle(handle)

{
	if (handle)
	{
		cuInit = (cuInit_t)dlsym(handle, "cuInit");
		if (!cuInit)
			THROW("Cannot dlsym cuInit " << dlerror());
		cuDeviceGet = (cuDeviceGet_t)dlsym(handle, "cuDeviceGet");
		if (!cuDeviceGet)
			THROW("Cannot dlsym cuDeviceGet " << dlerror());
		cuCtxCreate = (cuCtxCreate_t)dlsym(handle, "cuCtxCreate_v2");
		if (!cuCtxCreate)
			THROW("Cannot dlsym cuCtxCreate " << dlerror());
		cuCtxSynchronize = (cuCtxSynchronize_t)dlsym(handle, "cuCtxSynchronize");
		if (!cuCtxSynchronize)
			THROW("Cannot dlsym cuCtxSynchronize " << dlerror());
		cuMemAlloc = (cuMemAlloc_t)dlsym(handle, "cuMemAlloc_v2");
		if (!cuMemAlloc)
			THROW("Cannot dlsym cuMemAlloc " << dlerror());
		cuMemFree = (cuMemFree_t)dlsym(handle, "cuMemFree_v2");
		if (!cuMemFree)
			THROW("Cannot dlsym cuMemFree " << dlerror());
		cuMemAllocHost = (cuMemAlloc_t)dlsym(handle, "cuMemAllocHost_v2");
		if (!cuMemAllocHost)
			THROW("Cannot dlsym cuMemAllocHost " << dlerror());
		cuMemFreeHost = (cuMemFree_t)dlsym(handle, "cuMemFreeHost");
		if (!cuMemFreeHost)
			THROW("Cannot dlsym cuMemFreeHost " << dlerror());
		cuMemcpyHtoD = (cuMemcpy_t)dlsym(handle, "cuMemcpyHtoD_v2");
		if (!cuMemcpyHtoD)
			THROW("Cannot dlsym cuMemcpyHtoD " << dlerror());
		cuMemcpyDtoH = (cuMemcpy_t)dlsym(handle, "cuMemcpyDtoH_v2");
		if (!cuMemcpyDtoH)
			THROW("Cannot dlsym cuMemcpyDtoH " << dlerror());
		cuMemcpyHtoDAsync = (cuMemcpyAsync_t)dlsym(handle, "cuMemcpyHtoDAsync_v2");
		if (!cuMemcpyHtoDAsync)
			THROW("Cannot dlsym cuMemcpyHtoDAsync " << dlerror());
		cuMemcpyDtoHAsync = (cuMemcpyAsync_t)dlsym(handle, "cuMemcpyDtoHAsync_v2");
		if (!cuMemcpyDtoHAsync)
			THROW("Cannot dlsym cuMemcpyDtoHAsync " << dlerror());
		cuMemGetAddressRange = (cuMemGetAddressRange_t)dlsym(handle, "cuMemGetAddressRange_v2");
		if (!cuMemGetAddressRange)
			THROW("Cannot dlsym cuMemGetAddressRange " << dlerror());
		cuMemsetD8 = (cuMemsetD8_t)dlsym(handle, "cuMemsetD8_v2");
		if (!cuMemsetD8)
			THROW("Cannot dlsym cuMemsetD8 " << dlerror());
		cuMemsetD32 = (cuMemsetD32_t)dlsym(handle, "cuMemsetD32_v2");
		if (!cuMemsetD32)
			THROW("Cannot dlsym cuMemsetD32 " << dlerror());
		cuMemsetD32Async = (cuMemsetD32Async_t)dlsym(handle, "cuMemsetD32Async");
		if (!cuMemsetD32Async)
			THROW("Cannot dlsym cuMemsetD32Async " << dlerror());
		cuMemHostRegister = (cuMemHostRegister_t)dlsym(handle, "cuMemHostRegister");
		if (!cuMemHostRegister)
			THROW("Cannot dlsym cuMemHostRegister " << dlerror());
		cuMemHostUnregister = (cuMemHostUnregister_t)dlsym(handle, "cuMemHostUnregister");
		if (!cuMemHostUnregister)
			THROW("Cannot dlsym cuMemHostUnregister " << dlerror());
		cuModuleLoad = (cuModuleLoad_t)dlsym(handle, "cuModuleLoad");
		if (!cuModuleLoad)
			THROW("Cannot dlsym cuModuleLoad " << dlerror());
		cuModuleLoadData = (cuModuleLoad_t)dlsym(handle, "cuModuleLoadData");
		if (!cuModuleLoadData)
			THROW("Cannot dlsym cuModuleLoadData " << dlerror());
		cuModuleLoadDataEx = (cuModuleLoadDataEx_t)dlsym(handle, "cuModuleLoadDataEx");
		if (!cuModuleLoadDataEx)
			THROW("Cannot dlsym cuModuleLoadDataEx " << dlerror());
		cuModuleUnload = (cuModuleUnload_t)dlsym(handle, "cuModuleUnload");
		if (!cuModuleUnload)
			THROW("Cannot dlsym cuModuleUnload " << dlerror());
		cuModuleGetFunction = (cuModuleGetFunction_t)dlsym(handle, "cuModuleGetFunction");
		if (!cuModuleGetFunction)
			THROW("Cannot dlsym cuModuleGetFunction " << dlerror());
		cuModuleGetGlobal = (cuModuleGetGlobal_t)dlsym(handle, "cuModuleGetGlobal_v2");
		if (!cuModuleGetGlobal)
			THROW("Cannot dlsym cuModuleGetGlobal " << dlerror());
		cuLaunchKernel = (cuLaunchKernel_t)dlsym(handle, "cuLaunchKernel");
		if (!cuLaunchKernel)
			THROW("Cannot dlsym cuLaunchKernel " << dlerror());
		cuStreamCreate = (cuStreamCreate_t)dlsym(handle, "cuStreamCreate");
		if (!cuStreamCreate)
			THROW("Cannot dlsym cuStreamCreate " << dlerror());
		cuStreamSynchronize = (cuStreamSynchronize_t)dlsym(handle, "cuStreamSynchronize");
		if (!cuStreamSynchronize)
			THROW("Cannot dlsym cuStreamSynchronize " << dlerror());
	}

	CUresult err = cuInit(0);
	if (err)
		THROW("Error in cuInit " << err);
	
	int device;
	err = cuDeviceGet(&device, 0);
	if (err)
		THROW("Error in cuDeviceGet " << err);
	
	err = cuCtxCreate(&ctx, 0, device);
	if (err)
		THROW("Error in cuCtxCreate " << err);

	// Initialize the dynamic kernels loader.
	err = cudyInit(&loader, capacity);
	if (err)
		THROW("Cannot initialize the dynamic loader " << err);
}

context::~context()
{
	// TODO: destroy context, dlclose.

	// Dispose the dynamic kernels loader.
	CUresult err = cudyDispose(loader);
	if (err)
		THROW("Cannot dispose the dynamic loader " << err);
}

}}}

