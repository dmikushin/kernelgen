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

#include "KernelGen.h"
#include "Cuda.h"

#include <cstdlib>
#include <dlfcn.h>
#include <stddef.h>

using namespace kernelgen;
using namespace std;

namespace kernelgen {
namespace bind {
namespace cuda {

cuDeviceComputeCapability_t cuDeviceComputeCapability;
cuDeviceGetProperties_t cuDeviceGetProperties;
cuDeviceGetAttribute_t cuDeviceGetAttribute;
cuInit_t cuInit;
cuDeviceGet_t cuDeviceGet;
cuCtxCreate_t cuCtxCreate;
cuCtxGetDevice_t cuCtxGetDevice;
cuCtxSynchronize_t cuCtxSynchronize;
cuMemAlloc_t cuMemAlloc_;
cuMemFree_t cuMemFree_;
cuMemAlloc_t cuMemAllocHost;
cuMemFree_t cuMemFreeHost;
cuMemcpyHtoD_t cuMemcpyHtoD;
cuMemcpyDtoH_t cuMemcpyDtoH;
cuMemcpyHtoDAsync_t cuMemcpyHtoDAsync;
cuMemcpyDtoHAsync_t cuMemcpyDtoHAsync;
cuMemGetAddressRange_t cuMemGetAddressRange;
cuMemsetD8_t cuMemsetD8;
cuMemsetD32_t cuMemsetD32;
cuMemsetD32Async_t cuMemsetD32Async;
cuMemHostRegister_t cuMemHostRegister;
cuMemHostGetDevicePointer_t cuMemHostGetDevicePointer;
cuMemHostUnregister_t cuMemHostUnregister;
cuModuleLoad_t cuModuleLoad;
cuModuleLoadData_t cuModuleLoadData;
cuModuleLoadDataEx_t cuModuleLoadDataEx;
cuModuleUnload_t cuModuleUnload;
cuModuleGetFunction_t cuModuleGetFunction;
cuModuleGetGlobal_t cuModuleGetGlobal;
cuLaunchKernel_t cuLaunchKernel;
cuStreamCreate_t cuStreamCreate;
cuStreamSynchronize_t cuStreamSynchronize;
cuStreamDestroy_t cuStreamDestroy;
cuEventCreate_t cuEventCreate;
cuEventDestroy_t cuEventDestroy;
cuEventElapsedTime_t cuEventElapsedTime;
cuEventRecord_t cuEventRecord;
cuEventSynchronize_t cuEventSynchronize;
cuFuncGetAttribute_t cuFuncGetAttribute;
cuCtxSetCacheConfig_t cuCtxSetCacheConfig;

CUresult cuMemAlloc(void** ptr, size_t size) {
	// Create a possibly unaligned base buffer and
	// a strictly aligned return buffer on top of it.
	void* base = NULL;
	int err = cuMemAlloc_(&base, size + 4096);
	if (err)
		return err;
	*ptr = (char*) base + 4096 - (size_t) base % 4096;
	return CUDA_SUCCESS;
}

CUresult cuMemFree(void* ptr) {
	// Unpack carrier for the specified aligned buffer.
	void* base = NULL;
	int err = cuMemGetAddressRange(&base, NULL, ptr);
	if (err)
		return err;
	err = cuMemFree_(base);
}

context* context::init(int capacity) {
	// Do not init again, if already bound.
	if (cuInit)
		new context(NULL, capacity);

	// Load CUDA Driver API shared library.
	void* handle = dlopen("libcuda.so", RTLD_NOW | RTLD_GLOBAL | RTLD_DEEPBIND);
	if (!handle)
		THROW("Cannot dlopen libcuda.so " << dlerror());

	return new context(handle, capacity);
}

context::context(void* handle, int capacity) :

		handle(handle), capacity(capacity), ptxas("ptxas")

{
	#define DL_SAFE_CALL(name, suffix) \
	{ \
		name = (name##_t) dlsym(handle, "" #name suffix); \
		if (!name) \
			THROW("Cannot dlsym " #name "" << dlerror()); \
	}

	if (handle) {
		cuMemAlloc_ = (cuMemAlloc_t) dlsym(handle, "cuMemAlloc_v2");
		if (!cuMemAlloc)
			THROW("Cannot dlsym cuMemAlloc " << dlerror());
		cuMemFree_ = (cuMemFree_t) dlsym(handle, "cuMemFree_v2");
		if (!cuMemFree)
			THROW("Cannot dlsym cuMemFree " << dlerror());

		DL_SAFE_CALL(cuDeviceComputeCapability, "");
		DL_SAFE_CALL(cuDeviceGetProperties, "");
		DL_SAFE_CALL(cuDeviceGetAttribute, "");
		DL_SAFE_CALL(cuInit, "");
		DL_SAFE_CALL(cuDeviceGet, "");
		DL_SAFE_CALL(cuCtxCreate, "_v2");
		DL_SAFE_CALL(cuCtxGetDevice, "");
		DL_SAFE_CALL(cuCtxSynchronize, "");
		DL_SAFE_CALL(cuMemAllocHost, "_v2");
		DL_SAFE_CALL(cuMemFreeHost, "");
		DL_SAFE_CALL(cuMemcpyHtoD, "_v2");
		DL_SAFE_CALL(cuMemcpyDtoH, "_v2");
		DL_SAFE_CALL(cuMemcpyHtoDAsync, "_v2");
		DL_SAFE_CALL(cuMemcpyDtoHAsync, "_v2");
		DL_SAFE_CALL(cuMemGetAddressRange, "_v2");
		DL_SAFE_CALL(cuMemsetD8, "_v2");
		DL_SAFE_CALL(cuMemsetD32, "_v2");
		DL_SAFE_CALL(cuMemsetD32Async, "");
		DL_SAFE_CALL(cuMemHostRegister, "");
		DL_SAFE_CALL(cuMemHostGetDevicePointer, "");
		DL_SAFE_CALL(cuMemHostUnregister, "");
		DL_SAFE_CALL(cuModuleLoad, "");
		DL_SAFE_CALL(cuModuleLoadData, "");
		DL_SAFE_CALL(cuModuleLoadDataEx, "");
		DL_SAFE_CALL(cuModuleUnload, "");
		DL_SAFE_CALL(cuModuleGetFunction, "");
		DL_SAFE_CALL(cuModuleGetGlobal, "_v2");
		DL_SAFE_CALL(cuLaunchKernel, "");
		DL_SAFE_CALL(cuStreamCreate, "");
		DL_SAFE_CALL(cuStreamSynchronize, "");
		DL_SAFE_CALL(cuStreamDestroy, "_v2");
		DL_SAFE_CALL(cuEventCreate, "");
		DL_SAFE_CALL(cuEventDestroy, "");
		DL_SAFE_CALL(cuEventElapsedTime, "");
		DL_SAFE_CALL(cuEventRecord, "");
		DL_SAFE_CALL(cuEventSynchronize, "");
		DL_SAFE_CALL(cuFuncGetAttribute, "");
		DL_SAFE_CALL(cuCtxSetCacheConfig, "");
	}

	CU_SAFE_CALL(cuInit(0));

	// KernelGen-managed process always works with device #0.
	// In order to work with other devices, application should
	// control this using CUDA_VISIBLE_DEVICES env variable.
	CUdevice device;
	CU_SAFE_CALL(cuDeviceGet(&device, 0));

	// Determine device compute capability. Here we require used GPU
	// to be at least sm_20.
	CU_SAFE_CALL(cuDeviceComputeCapability(&subarchMajor, &subarchMinor, device));
	int isubarch = subarchMajor * 10 + subarchMinor;
	stringstream subarchStr;
	subarchStr << "sm_" << subarchMajor << subarchMinor;
	subarch = subarchStr.str();
	if (subarchMajor < 2)
		THROW("Available GPU must be at least sm_20 (have " << subarch << ")");

	// Get the threadsPerBlock property.
	CU_SAFE_CALL(cuDeviceGetAttribute(&threadsPerBlock,
			CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK, device));

	// Get the regsPerBlock property.
	CU_SAFE_CALL(cuDeviceGetAttribute(&regsPerBlock,
			CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_BLOCK, device));

	CU_SAFE_CALL(cuCtxCreate(&ctx, CU_CTX_MAP_HOST, device));

	// Since KernelGen does not utilize shared memory at the moment,
	// use larger L1 cache by default.
	CU_SAFE_CALL(cuCtxSetCacheConfig(CU_FUNC_CACHE_PREFER_L1));

	// Initialize LEPC buffer.
	size_t szlepc = 4;
	lepcBuffer = NULL;
	CU_SAFE_CALL(cuMemAlloc((CUdeviceptr*)&lepcBuffer, szlepc));
	CU_SAFE_CALL(cuMemsetD8((CUdeviceptr)lepcBuffer, 0, szlepc));

	// Initialize streams.
	CU_SAFE_CALL(cuStreamCreate(&primaryStream, 0));
	CU_SAFE_CALL(cuStreamCreate(&secondaryStream, 0));

	// Setup PTXAS executable, if specified.
	char* kernelgen_ptxas = getenv("kernelgen_ptxas");
	if (kernelgen_ptxas)
		ptxas = kernelgen_ptxas;
}

unsigned int context::getLEPC() const
{
	// Substract 0x10, because LEPC instruction comes as 3rd instruction on Kepler.
	// TODO: 2nd instruction on Fermi.
	size_t szlepc = 4;
	unsigned int lepc;
	CU_SAFE_CALL(cuMemcpyDtoHAsync(&lepc, (CUdeviceptr)lepcBuffer, szlepc, secondaryStream));
	CU_SAFE_CALL(cuStreamSynchronize(secondaryStream));
	return lepc;
}

context::~context() {
	// TODO: destroy context, dlclose.

	// Free the LEPC buffer.
	CU_SAFE_CALL(cuMemFree((CUdeviceptr)lepcBuffer));

	// Dispose the dynamic kernels loader.
	CU_SAFE_CALL(cudyDispose(loader));

	// Dispose streams.
	CU_SAFE_CALL(cuStreamDestroy(primaryStream));
	CU_SAFE_CALL(cuStreamDestroy(secondaryStream));
}

}
}
}
