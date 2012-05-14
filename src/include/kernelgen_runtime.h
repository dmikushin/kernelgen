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

#define __device__ static __inline__ __attribute__((always_inline))

extern int __iAtomicCAS(int *address, int compare, int val);

extern unsigned int* __kernelgen_callback;

extern unsigned int* __kernelgen_memory;

#include "kernelgen_interop.h"
#include "kernelgen_memory.h"

typedef struct { unsigned int x, y, z; } uint3;

uint3 extern const threadIdx, blockIdx, blockDim, gridDim;
int extern const warpSize;

__device__ void kernelgen_hostcall(unsigned char* kernel,
	unsigned long long szdata, unsigned long long szdatai, unsigned int* data)
{
	// Unblock the monitor kernel and wait for being
	// unblocked by new instance of monitor.
	struct kernelgen_callback_t* callback =
		(struct kernelgen_callback_t*)__kernelgen_callback;
	callback->state = KERNELGEN_STATE_HOSTCALL;
	callback->kernel = (struct kernel_t*)kernel;
	callback->szdata = szdata;
	callback->szdatai = szdatai;
	callback->data = (struct kernelgen_callback_data_t*)data;
	__iAtomicCAS(&callback->lock, 0, 1);
	while (__iAtomicCAS(&callback->lock, 0, 0)) continue;
}

__device__ int kernelgen_launch(unsigned char* kernel,
	unsigned long long szdata, unsigned long long szdatai, unsigned int* data)
{
	// Client passes NULL for name/entry argument to indicate
	// the call is performed from kernel loop and must always
	// return -1.
	if (!kernel) return -1;

	struct kernelgen_callback_t* callback =
		(struct kernelgen_callback_t*)__kernelgen_callback;
	callback->state = KERNELGEN_STATE_LOOPCALL;
	callback->kernel = (struct kernel_t*)kernel;
	callback->szdata = szdata;
	callback->szdatai = szdatai;
	callback->data = (struct kernelgen_callback_data_t*)data;
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

