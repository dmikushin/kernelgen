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

#define kernelgen_malloc_device(...) kernelgen_malloc(__VA_ARGS__)
#define kernelgen_free_device(...) kernelgen_free(__VA_ARGS__)

#include "malloc.h"
#include "free.h"

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

