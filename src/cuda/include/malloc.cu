//===- Memory.h - Device memory pool for NVIDIA GPUs ----------------------===//
//
//     KernelGen -- A prototype of LLVM-based auto-parallelizing Fortran/C
//        compiler for NVIDIA GPUs, targeting numerical modeling code.
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains functions to support the dynamic memory heap in GPU
// global memory. The rationale is to replace builtin malloc/free calls, since
// they are incompatible with concurrent kernels execution.
//
//===----------------------------------------------------------------------===//

#define kernelgen_malloc_device(...) kernelgen_malloc_device(__VA_ARGS__, unsigned int* __kernelgen_memory)
#include "malloc.h"
#undef kernelgen_malloc_device

struct malloc_callback_t
{
	void** ptr;
	size_t size;
};

extern "C" __attribute__((global)) void kernelgen_malloc(int* callback, unsigned int* __kernelgen_memory)
{
	malloc_callback_t* cb = (malloc_callback_t*)callback;
	*cb->ptr = kernelgen_malloc_device(cb->size, __kernelgen_memory);
}
