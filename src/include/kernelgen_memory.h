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

#ifndef KERNELGEN_MEMORY_H
#define KERNELGEN_MEMORY_H

/*
 * This file contains device-side functions to support the dynamic
 * memory heap in GPU global memory. The rationale is to replace
 * builtin malloc/free calls, since they are incompatible with
 * concurrent kernels execution.
 */

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

__device__ void* kernelgen_malloc(size_t size)
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

__device__ int kernelgen_posix_memalign(void** ptr, size_t alignment, size_t size)
{
	// TODO: Do actual alignment somehow, currently
	// memory is always aligned to 4096 bytes.
	*ptr = kernelgen_malloc(size);
	return 0;
}

__device__ void kernelgen_free(void* p)
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

#endif // KERNELGEN_MEMORY_H

