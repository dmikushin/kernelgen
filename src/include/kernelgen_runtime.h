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

#include <stddef.h>

#include "kernelgen_interop.h"

extern __attribute__((__malloc__)) void *malloc(size_t);
extern void free(void*);

extern __attribute__((device)) int __iAtomicCAS(
	int *address, int compare, int val);

extern __attribute__((device)) unsigned int* __kernelgen_callback;

static __inline__ __attribute__((always_inline)) void kernelgen_hostcall(
	unsigned char* name, unsigned int* args)
{
	// Unblock the monitor kernel and wait for being
	// unblocked by new instance of monitor.
	struct kernelgen_callback_t* callback =
		(struct kernelgen_callback_t*)__kernelgen_callback;
	callback->state = KERNELGEN_STATE_HOSTCALL;
	callback->name = name;
	callback->arg = args;
	__iAtomicCAS(&callback->lock, 0, 1);
	while (__iAtomicCAS(&callback->lock, 0, 0)) continue;
}

static __inline__ __attribute__((always_inline)) int kernelgen_launch(
	unsigned char* name, unsigned int* args)
{
	struct kernelgen_callback_t* callback =
		(struct kernelgen_callback_t*)__kernelgen_callback;
	callback->state = KERNELGEN_STATE_LOOPCALL;
	callback->name = name;
	callback->arg = args;

	// FIXME: Currently, not launching any other kernels.
	return -1;
}

static __inline__ __attribute__((always_inline)) kernelgen_finish()
{
	// Unblock the monitor kernel.
	struct kernelgen_callback_t* callback =
		(struct kernelgen_callback_t*)__kernelgen_callback;
	callback->state = KERNELGEN_STATE_INACTIVE;
	__iAtomicCAS(&callback->lock, 0, 1);
}

#endif // KERNELGEN_RUNTIME_H

