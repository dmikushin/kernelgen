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

#ifndef KERNELGEN_MALLOC_H
#define KERNELGEN_MALLOC_H

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

__attribute__((device)) __attribute__((always_inline)) void* kernelgen_malloc_device(size_t size)
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

#endif // KERNELGEN_MALLOC_H
