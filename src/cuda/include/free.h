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

#ifndef KERNELGEN_FREE_H
#define KERNELGEN_FREE_H

#include "kernelgen_interop.h"

#define KERNELGEN_MEM_FREE	0
#define KERNELGEN_MEM_IN_USE	1

#ifndef kernelgen_memory_chunk
#define kernelgen_memory_chunk
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
#endif

__attribute__((device)) __attribute__((always_inline)) void kernelgen_free_device(void* p)
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

#endif // KERNELGEN_FREE_H
