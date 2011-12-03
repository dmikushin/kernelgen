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

#include <dlfcn.h>

namespace kernelgen { namespace bind {

namespace cuda {

	typedef int (*cuInit_t)(unsigned int);
	typedef int (*cuDeviceGet_t)(int*, int);
	typedef int (*cuCtxCreate_t)(int*, unsigned int, int);
	typedef int (*cuMemAlloc_t)(void**, size_t);
	typedef int (*cuMemFree_t)(void*);
	typedef int (*cuMemcpy_t)(void*, void*, size_t);

	extern cuInit_t cuInit;
	extern cuDeviceGet_t cuDeviceGet;
	extern cuCtxCreate_t cuCtxCreate;
	extern cuMemAlloc_t cuMemAlloc;
	extern cuMemFree_t cuMemFree;
	extern cuMemcpy_t cuMemcpyHtoD, cuMemcpyDtoH;

	void init();
}}}

#endif // KERNELGEN_BIND_H

