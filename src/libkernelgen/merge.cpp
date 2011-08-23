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

#include "kernelgen_int.h"

#include <list>

using namespace std;

// Compare two args by base value.
static bool compare(
	struct kernelgen_memory_region_t* first,
	struct kernelgen_memory_region_t* second)
{
	return (size_t)first->base < (size_t)second->base;
}

// Merge specified memory regions into non-overlapping regions.
extern "C" kernelgen_status_t kernelgen_merge_regions(
	struct kernelgen_memory_region_t* regs,
	int count)
{
	// Load agruments lregs with values.
	list<struct kernelgen_memory_region_t*> lregs;
	for (int i = 0; i < count; i++)
	{	
		lregs.push_back(regs + i);
		regs[i].primary = NULL;
	}
	
	// Sort lregs by base value.
	lregs.sort(compare);
	
	// Walk through each pair of intervals sorted in
	// ascending order.
	list<struct kernelgen_memory_region_t*>::iterator it1 = lregs.begin();
	list<struct kernelgen_memory_region_t*>::iterator it2 = lregs.begin();
	for (int i = 0; i < count - 1; i++, it1++)
	{
		it2++;

		// Get two left-most intervals relative layout.
		struct kernelgen_memory_region_t* reg1 = *it1;
		struct kernelgen_memory_region_t* reg2 = *it2;
		
		// The left border of second interval is inside
		// first interval.
		// XXX: temporary disabled regions concatenation
		// due to bugs in AMD OpenCL (here "<" instead of "<=").
		if ((size_t)reg2->base < (size_t)reg1->base + reg1->size)
		{
			// The right border of second interval is outside
			// first interval (i.e. first does not contain second).
			if ((size_t)reg2->base + reg2->size > (size_t)reg1->base + reg1->size)
			{
				// Extend first interval size to cover second.
				reg2->size = reg1->size + (size_t)reg2->base +
					reg2->size - (size_t)reg1->base - reg1->size;
				
				// Go backwards and update sizes of all previous
				// intervals with the same base as first.
				for (list<struct kernelgen_memory_region_t*>::iterator it3 =
					lregs.begin(); it3 != it2; it3++)
				{
					struct kernelgen_memory_region_t* reg3 = *it3;
					if (reg3->base == reg1->base)
						reg3->size = reg2->size;
				}
			}
			else
			{
				// Extend second interval to be equal to the first.
				reg2->size = reg1->size;
			}

			// Calculate second shift relative to first base.
			reg2->shift += (size_t)reg2->base - (size_t)reg1->base;
			
			// Use base of first interval in second.
			reg2->base = reg1->base;

			// Go backwards and find interval with the smallest
			// index.
			struct kernelgen_memory_region_t* rmin = reg2;
			for (list<struct kernelgen_memory_region_t*>::iterator it3 =
				lregs.begin(); it3 != it2; it3++)
			{
				struct kernelgen_memory_region_t* reg3 = *it3;
				if (reg3->base == reg1->base)
				{
					if (reg3->symbol->index < rmin->symbol->index)
						rmin = reg3;
					reg3->primary = NULL;
				}
			}
			
			// Go backwards and set interval with smallest
			// index as primary.
			reg2->primary = rmin;
			for (list<struct kernelgen_memory_region_t*>::iterator it3 =
				lregs.begin(); it3 != it2; it3++)
			{
				struct kernelgen_memory_region_t* reg3 = *it3;
				if (reg3->base == reg1->base)
					reg3->primary = rmin;
			}
			rmin->primary = NULL;
		}
	}
	
	kernelgen_status_t result;
	result.value = kernelgen_success;
	result.runmode = 0;
	return result;
}

