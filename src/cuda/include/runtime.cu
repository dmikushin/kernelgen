//===- runtime.cu - Runtime functions for NVIDIA GPUs ---------------------===//
//
//     KernelGen -- A prototype of LLVM-based auto-parallelizing Fortran/C
//        compiler for NVIDIA GPUs, targeting numerical modeling code.
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements runtime functions for NVIDIA GPUs.
//
//===----------------------------------------------------------------------===//

#ifndef KERNELGEN_RUNTIME_H
#define KERNELGEN_RUNTIME_H

#ifdef __cplusplus
extern "C"
{
#endif

extern unsigned int* __attribute__((device)) __kernelgen_callback;
extern unsigned int* __attribute__((device)) __kernelgen_memory;

#include "kernelgen_interop.h"

#define KERNELGEN_MEM_NEW_MCB	0
#define KERNELGEN_MEM_NO_MCB	1
#define KERNELGEN_MEM_REUSE_MCB	2

#define KERNELGEN_MEM_FREE	0
#define KERNELGEN_MEM_IN_USE	1

typedef struct
{
	int is_available;
	int size;
	
	// Align structure to 4096, or there will be
	// problems with 128-bit loads/stores in
	// optimized Fermi ISA (nvopencc issue?).
	char padding[4096 - 8];
}
kerelgen_memory_chunk_t;

__attribute__((device)) __attribute__((always_inline)) void* kernelgen_malloc(size_t size)
{
	// Align size.
	if (size % 4096) size += 4096 - size % 4096;

	kernelgen_memory_t* km = (kernelgen_memory_t*)__kernelgen_memory;

	// If there is less free space in pool, than requested,
	// then just return NULL.
	if (size + sizeof(kerelgen_memory_chunk_t) > km->szpool -
		(km->szused + km->count * sizeof(kerelgen_memory_chunk_t)))
		return NULL;

	// Find a free memory chunk.
	size_t i = 0;
	for ( ; i + size + sizeof(kerelgen_memory_chunk_t) < km->szpool; )
	{
		kerelgen_memory_chunk_t* p_mcb =
			(kerelgen_memory_chunk_t*)(km->pool + i);
		if (p_mcb->is_available == KERNELGEN_MEM_FREE)
		{
			// If this is a new unused chunk in the tail of pool.
			if (p_mcb->size == 0)
			{
				p_mcb->is_available = KERNELGEN_MEM_IN_USE;
				p_mcb->size = size + sizeof(kerelgen_memory_chunk_t);
				km->count++;
				km->szused += size;
				return (char*)p_mcb + sizeof(kerelgen_memory_chunk_t);
			}

			// If size of the available chunk is equal to greater
			// than required size, use that chunk.
			if (p_mcb->size >= (size + sizeof(kerelgen_memory_chunk_t)))
			{
				p_mcb->is_available = KERNELGEN_MEM_IN_USE;
				size = p_mcb->size - sizeof(kerelgen_memory_chunk_t);
				km->count++;
				km->szused += size;

				// TODO: Mark the rest of the used chunk as a new
				// free chunk?

				return (char*)p_mcb + sizeof(kerelgen_memory_chunk_t);
			}
		}
		i += p_mcb->size;	
	}
	
	return NULL;
}

__attribute__((device)) __attribute__((always_inline)) int kernelgen_posix_memalign(void** ptr, size_t alignment, size_t size)
{
	// TODO: Do actual alignment somehow, currently
	// memory is always aligned to 4096 bytes.
	*ptr = kernelgen_malloc(size);
	return 0;
}

__attribute__((device)) __attribute__((always_inline)) void* kernelgen_memalign(size_t boundary, size_t size)
{
	// TODO: Do actual alignment somehow, currently
	// memory is always aligned to 4096 bytes.
	return kernelgen_malloc(size);
	return 0;
}

__attribute__((device)) __attribute__((always_inline)) void kernelgen_free(void* p)
{
	kernelgen_memory_t* km = (kernelgen_memory_t*)__kernelgen_memory;

	// Mark in MCB that this chunk is free.
	kerelgen_memory_chunk_t* ptr = (kerelgen_memory_chunk_t*)p - 1;
	if (ptr->is_available != KERNELGEN_MEM_FREE)
	{
		km->count--;
		ptr->is_available = KERNELGEN_MEM_FREE;
		km->szused -= (ptr->size - sizeof(kerelgen_memory_chunk_t));
	}

	// TODO: if the last chunk is freed, then we need
	// to flush its contents to zero.
}

__attribute__((device)) __attribute__((always_inline)) int __iAtomicCAS(volatile int *p, int compare, int val)
{
	int ret;
	asm volatile (
		"atom.global.cas.b32    %0, [%1], %2, %3; \n\t"
		: "=r"(ret) : "l"(p), "r"(compare), "r"(val)
	);
	return ret;
}

__attribute__((device)) __attribute__((always_inline)) void kernelgen_hostcall(unsigned char* kernel,
	unsigned long long szdata, unsigned long long szdatai, unsigned int* data)
{
	// Unblock the monitor kernel and wait for being
	// unblocked by new instance of monitor.
	struct kernelgen_callback_t* callback =
		(struct kernelgen_callback_t*)__kernelgen_callback;
	callback->state = KERNELGEN_STATE_HOSTCALL;
#ifdef __cplusplus
	callback->kernel = (kernelgen::Kernel*)kernel;
#else
	callback->kernel = (struct Kernel*)kernel;
#endif 
	callback->szdata = szdata;
	callback->szdatai = szdatai;
	callback->data = (struct CallbackData*)data;
	__iAtomicCAS(&callback->lock, 0, 1);
	while (__iAtomicCAS(&callback->lock, 0, 0)) continue;
}

__attribute__((device)) __attribute__((always_inline)) void kernelgen_start()
{
	// Wait for being unblocked by an instance of monitor.
	struct kernelgen_callback_t* callback =
		(struct kernelgen_callback_t*)__kernelgen_callback;
	while (__iAtomicCAS(&callback->lock, 0, 0)) continue;
}

__attribute__((device)) __attribute((always_inline)) int kernelgen_launch(unsigned char* kernel,
	unsigned long long szdata, unsigned long long szdatai, unsigned int* data)
{
	// Client passes NULL for name/entry argument to indicate
	// the call is performed from kernel loop and must always
	// return -1.
	if (!kernel) return -1;

	struct kernelgen_callback_t* callback =
		(struct kernelgen_callback_t*)__kernelgen_callback;
	callback->state = KERNELGEN_STATE_LOOPCALL;
#ifdef __cplusplus
	callback->kernel = (kernelgen::Kernel*)kernel;
#else
	callback->kernel = (struct Kernel*)kernel;
#endif
	callback->szdata = szdata;
	callback->szdatai = szdatai;
	callback->data = (struct CallbackData*)data;
	__iAtomicCAS(&callback->lock, 0, 1);
	while (__iAtomicCAS(&callback->lock, 0, 0)) continue;

	// The launch status is returned through the
	// state value. If it is -1, then serial version
	// of kernel is executed in the main thread.
	return callback->state;
}

__attribute__((device)) __attribute__((always_inline)) void kernelgen_finish()
{
	// Unblock the monitor kernel.
	struct kernelgen_callback_t* callback =
		(struct kernelgen_callback_t*)__kernelgen_callback;
	callback->state = KERNELGEN_STATE_INACTIVE;
	__iAtomicCAS(&callback->lock, 0, 1);
}

__attribute__((global)) void kernelgen_memcpy(void* dst, void* src, size_t size)
{
	for (int i = 0; i < size; i++)
		((char*)dst)[i] = ((char*)src)[i];
}

#ifdef __cplusplus
}
#endif

#endif // KERNELGEN_RUNTIME_H

