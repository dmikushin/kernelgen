/*
 * KGen - the LLVM-based compiler with GPU kernels generation over C backend.
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

// Test case for merge_intervals. Creates the specified number of
// random integers, then randomly connects pairs into intervals
// and merges into non-overlapping intervals using merge_intervals.

#include "gforscale_int.h"

#include <malloc.h>
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char* argv[])
{
	if (argc != 3)
	{
		printf("Usage: %s <n> <max>\n", argv[0]);
		return 0;
	}
	
	int n = atoi(argv[1]);
	int max = atoi(argv[2]);

	int* pnts = (int*)malloc(sizeof(int) * n);
		
	struct gforscale_memory_region_t* regs =
		(struct gforscale_memory_region_t*)malloc(
			sizeof(struct gforscale_memory_region_t) * n);

	struct gforscale_kernel_symbol_t* args =
		(struct gforscale_kernel_symbol_t*)malloc(
			sizeof(struct gforscale_kernel_symbol_t) * n);
	for (int i = 0; i < n; i++)
	{
		args[i].index = i;
		args[i].mref = regs + i;
		args[i].mdesc = regs + i;
	}

	// Generate points.
	for (int i = 0; i < n; i++)
	{
		pnts[i] = rand() % max;
	}
	
	// Generate intervals.
	for (int i = 0; i < n; i++)
	{
		int ibegin = rand() % n;
		int iend = rand() % n;
		
		struct gforscale_memory_region_t* reg = regs + i;
		
		reg->base = (void*)(size_t)(pnts[ibegin]);
		reg->size = (size_t)pnts[iend];
		reg->shift = (unsigned int)(rand() % reg->size);
		reg->symbol = args + i;
	}
	
	for (int i = 0; i < n; i++)
	{
		struct gforscale_memory_region_t* reg = regs + i;
		
		printf("reg in  : %d [%p + %u, %p]\n",
			i, reg->base, reg->shift, (void*)((size_t)reg->base + reg->size));
	}
	printf("\n");
	
	gforscale_merge_regions(regs, n);

	for (int i = 0; i < n; i++)
	{
		struct gforscale_memory_region_t* reg = regs + i;

		printf("reg out : %d [%p + %u, %p]",
			i, reg->base, reg->shift, (void*)((size_t)reg->base + reg->size));
		if (!reg->primary) printf("*");
		printf("\n");
	}
	
	free(pnts);
	free(regs);
	free(args);

	return 0;
}

