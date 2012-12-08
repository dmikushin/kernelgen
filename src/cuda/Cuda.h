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

#ifndef KERNELGEN_BIND_H
#define KERNELGEN_BIND_H

typedef struct CUevent_st* 	CUevent;
typedef struct CUmodule_st*	CUmodule;
typedef int 			CUresult;
typedef struct CUstream_st*	CUstream;
typedef struct CUfunction_st*	CUfunction;
typedef void*			CUdeviceptr;

#define CUDA_SUCCESS					0
#define CUDA_ERROR_OUT_OF_MEMORY			2
#define CUDA_ERROR_INVALID_SOURCE			300
#define CUDA_ERROR_HOST_MEMORY_ALREADY_REGISTERED	712
#define CUDA_ERROR_HOST_MEMORY_NOT_REGISTERED		713

#define CU_LAUNCH_PARAM_BUFFER_POINTER			((void*)0x01)
#define CU_LAUNCH_PARAM_BUFFER_SIZE			((void*)0x02)
#define CU_LAUNCH_PARAM_END				((void*)0x00)

#define CU_FUNC_ATTRIBUTE_NUM_REGS			4

#include "cuda_dyloader.h"

namespace kernelgen { namespace bind { namespace cuda {

typedef CUresult (*cuDeviceComputeCapability_t)(int*, int*, int);
typedef CUresult (*cuDeviceGetProperties_t)(void*, int);
typedef CUresult (*cuInit_t)(unsigned int);
typedef CUresult (*cuDeviceGet_t)(int*, int);
typedef CUresult (*cuCtxCreate_t)(void**, unsigned int, int);
typedef CUresult (*cuCtxSynchronize_t)(void);
typedef CUresult (*cuMemAlloc_t)(void**, size_t);
typedef CUresult (*cuMemFree_t)(void*);
typedef CUresult (*cuMemcpy_t)(void*, void*, size_t);
typedef CUresult (*cuMemcpyAsync_t)(void*, void*, size_t, void*);
typedef CUresult (*cuMemGetAddressRange_t)(void**, size_t*, void*);
typedef CUresult (*cuMemsetD8_t)(void*, unsigned char, size_t);
typedef CUresult (*cuMemsetD32_t)(void*, unsigned int, size_t);
typedef CUresult (*cuMemsetD32Async_t)(void*, unsigned int, size_t, void*);
typedef CUresult (*cuMemHostRegister_t)(void*, size_t, unsigned int);
typedef CUresult (*cuMemHostGetDevicePointer_t)(void**, void*, unsigned int);
typedef CUresult (*cuMemHostUnregister_t)(void*);
typedef CUresult (*cuModuleLoad_t)(CUmodule*, const char*);
typedef CUresult (*cuModuleLoadDataEx_t)(CUmodule*, const char*, unsigned int, int* options, void**);
typedef CUresult (*cuModuleUnload_t)(void*);
typedef CUresult (*cuModuleGetFunction_t)(CUfunction*, void*, const char*);
typedef CUresult (*cuModuleGetGlobal_t)(void**, size_t*, void*, const char*);
typedef CUresult (*cuLaunchKernel_t)(void*, unsigned int, unsigned int, unsigned int,
	unsigned int, unsigned int, unsigned int, unsigned int, void*, void**, void**);
typedef CUresult (*cuStreamCreate_t)(void*, unsigned int);
typedef CUresult (*cuStreamSynchronize_t)(void*);
typedef CUresult (*cuEventCreate_t)(CUevent*, unsigned int);
typedef CUresult (*cuEventDestroy_t)(CUevent);
typedef CUresult (*cuEventElapsedTime_t)(float*, CUevent, CUevent);
typedef CUresult (*cuEventRecord_t)(CUevent, CUstream);
typedef CUresult (*cuEventSynchronize_t)(CUevent);
typedef CUresult (*cuFuncGetAttribute_t)(int*, int, CUfunction);

extern cuDeviceComputeCapability_t cuDeviceComputeCapability;
extern cuDeviceGetProperties_t cuDeviceGetProperties;
extern cuInit_t cuInit;
extern cuDeviceGet_t cuDeviceGet;
extern cuCtxCreate_t cuCtxCreate;
extern cuCtxSynchronize_t cuCtxSynchronize;
extern cuMemAlloc_t cuMemAlloc_;
extern cuMemFree_t cuMemFree_;
extern cuMemAlloc_t cuMemAllocHost;
extern cuMemFree_t cuMemFreeHost;
extern cuMemcpy_t cuMemcpyHtoD, cuMemcpyDtoH;
extern cuMemcpyAsync_t cuMemcpyHtoDAsync, cuMemcpyDtoHAsync;
extern cuMemGetAddressRange_t cuMemGetAddressRange;
extern cuMemsetD8_t cuMemsetD8;
extern cuMemsetD32_t cuMemsetD32;
extern cuMemsetD32Async_t cuMemsetD32Async;
extern cuMemHostRegister_t cuMemHostRegister;
extern cuMemHostGetDevicePointer_t cuMemHostGetDevicePointer;
extern cuMemHostUnregister_t cuMemHostUnregister;
extern cuModuleLoad_t cuModuleLoad;
extern cuModuleLoad_t cuModuleLoadData;
extern cuModuleLoadDataEx_t cuModuleLoadDataEx;
extern cuModuleUnload_t cuModuleUnload;
extern cuModuleGetFunction_t cuModuleGetFunction;
extern cuModuleGetGlobal_t cuModuleGetGlobal;
extern cuLaunchKernel_t cuLaunchKernel;
extern cuStreamCreate_t cuStreamCreate;
extern cuStreamSynchronize_t cuStreamSynchronize;
extern cuEventCreate_t cuEventCreate;
extern cuEventDestroy_t cuEventDestroy;
extern cuEventElapsedTime_t cuEventElapsedTime;
extern cuEventRecord_t cuEventRecord;
extern cuEventSynchronize_t cuEventSynchronize;
extern cuFuncGetAttribute_t cuFuncGetAttribute;

CUresult cuMemAlloc(void** ptr, size_t size);
CUresult cuMemFree(void* ptr);

struct context {

	// Initialize a new instance of CUDA host API bindings.
	static context* init(int capacity);

private :

	// CUDA shared library handle.
	void* handle;

	// CUDA context.
	void* ctx;

	context(void* handle, int capacity);
	
	void* lepcBuffer;

public :

	inline void* getLEPCBufferPtr() const { return lepcBuffer; }

	// Dynamic loader.
	CUDYloader loader;

	int capacity;

	~context();
};

class CUBIN {

public:

	// Align cubin global data to the specified boundary.
	static void AlignData(const char* cubin, size_t align);

	// Merge two input CUBIN ELF images into single output image.
	static void Merge(const char* input1, const char* input2, const char* output);

	// Insert commands to perform LEPC reporting.
	static void InsertLEPCReporter(const char* cubin, const char* ckernel_name);
};

} // namespace cuda
} // namespace bind
} // namespace kernelgen

#endif // KERNELGEN_BIND_H

