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

#ifndef KERNELGEN_RUNTIME_H
#define KERNELGEN_RUNTIME_H

#include <stdio.h>

#include "kernelgen_interop.h"

#define __device__  static __inline__ __attribute__((always_inline))

extern __attribute__((__malloc__)) void *malloc(size_t);
extern void free(void*);

extern __attribute__((device)) int __iAtomicCAS(
	int *address, int compare, int val);

extern __attribute__((device)) unsigned int* __kernelgen_callback;

typedef struct { unsigned int x, y, z; } uint3;

uint3 extern const threadIdx, blockIdx, blockDim, gridDim;
int extern const warpSize;

__device__ void kernelgen_hostcall(unsigned char* name, unsigned long long szarg, unsigned int* arg)
{
	// Unblock the monitor kernel and wait for being
	// unblocked by new instance of monitor.
	struct kernelgen_callback_t* callback =
		(struct kernelgen_callback_t*)__kernelgen_callback;
	callback->state = KERNELGEN_STATE_HOSTCALL;
	callback->name = name;
	callback->szarg = szarg;
	callback->arg = arg;
	__iAtomicCAS(&callback->lock, 0, 1);
	while (__iAtomicCAS(&callback->lock, 0, 0)) continue;
}

__device__ int kernelgen_launch(unsigned char* name, unsigned long long szarg, unsigned int* arg)
{
	struct kernelgen_callback_t* callback =
		(struct kernelgen_callback_t*)__kernelgen_callback;
	callback->state = KERNELGEN_STATE_LOOPCALL;
	callback->name = name;
	callback->szarg = szarg;
	callback->arg = arg;
	__iAtomicCAS(&callback->lock, 0, 1);
	while (__iAtomicCAS(&callback->lock, 0, 0)) continue;

	// The launch status is returned through the
	// state value. If it is -1, then serial version
	// of kernel is executed in the main thread.
	return callback->state;
}

__device__ void kernelgen_finish()
{
	// Unblock the monitor kernel.
	struct kernelgen_callback_t* callback =
		(struct kernelgen_callback_t*)__kernelgen_callback;
	callback->state = KERNELGEN_STATE_INACTIVE;
	__iAtomicCAS(&callback->lock, 0, 1);
}

__device__ int puts(const char* str)
{
	int ret = printf("%s\n", str);
	if (ret < 0) return EOF;
	return ret;
}

__device__ int kernelgen_threadIdx_x() { return threadIdx.x; }
__device__ int kernelgen_threadIdx_y() { return threadIdx.y; }
__device__ int kernelgen_threadIdx_z() { return threadIdx.z; }

__device__ int kernelgen_blockIdx_x() { return blockIdx.x; }
__device__ int kernelgen_blockIdx_y() { return blockIdx.y; }
__device__ int kernelgen_blockIdx_z() { return blockIdx.z; }

__device__ int kernelgen_blockDim_x() { return blockDim.x; }
__device__ int kernelgen_blockDim_y() { return blockDim.y; }
__device__ int kernelgen_blockDim_z() { return blockDim.z; }

__device__ int kernelgen_gridDim_x() { return gridDim.x; }
__device__ int kernelgen_gridDim_y() { return gridDim.y; }
__device__ int kernelgen_gridDim_z() { return gridDim.z; }

#endif // KERNELGEN_RUNTIME_H

