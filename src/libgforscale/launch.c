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

#include "gforscale_int.h"

#include <malloc.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

long gforscale_launch_verbose = 1;

// Launch kernel with the specified 3D compute grid, arguments and modules symbols.
void gforscale_launch_(
	struct gforscale_kernel_config_t* config,
	int* bx, int* ex, int* by, int* ey, int* bz, int* ez,
	int* nargs, int* nmodsyms, ...)
{
	// For each used runmode, populate its corresponding launch
	// config, handle data and execute the kernel.
	// NOTE We skip first runmode index, as it is always stands for
	// execution of original loop on host.
	for (int irunmode = 1, nmapped = 0, nregions; irunmode < gforscale_runmodes_count; irunmode++)
	{
		// Indirect indexing automatically has
		// unsupported runmodes skipped.
		int runmode = gforscale_runmodes[irunmode];
		
		// Although, specific kernel may have supported
		// runmode disabled.
		if (!(config->runmode & runmode)) continue;

		// Kernel config structure.
		struct gforscale_launch_config_t* l = config->launch + irunmode;

		gforscale_print_debug(gforscale_launch_verbose,
			"Launching %s for device runmode \"%s\"\n",
			l->kernel_name, gforscale_runmodes_names[irunmode]);

		// Being quiet optimistic initially...
		gforscale_status_t status;
		status.value = gforscale_success;
		status.runmode = runmode;

		va_list list;
		va_start(list, nmodsyms);
		status = gforscale_parse_args(l, nargs, list);
		va_end(list);
		
		if (status.value != gforscale_success)
		{
			va_end(list);
			goto failsafe;
		}

		va_start(list, nmodsyms);
		for (int i = 0; i < *nargs; i++)
		{
			va_arg(list, void*);
			va_arg(list, size_t*);
			va_arg(list, void*);
		}
		status = gforscale_parse_modsyms[irunmode](l, nmodsyms, list);
		va_end(list);

		if (status.value != gforscale_success)
		{
			va_end(list);
			goto failsafe;
		}

		// Merge memory regions into non-overlapping regions.
		nregions = l->args_nregions + l->deps_nregions;
		status = gforscale_merge_regions(l->regs, nregions);
	
		// Map or load regions into device memory space.
		status = gforscale_load_regions[irunmode](l, &nmapped);

		if (status.value != gforscale_success)
			goto failsafe;

		status = gforscale_launch[irunmode](l, bx, ex, by, ey, bz, ez);

		if (status.value != gforscale_success)
			goto failsafe;

		// Unmap or save regions from device memory space.
		status = gforscale_save_regions[irunmode](l, nmapped);

		if (status.value != gforscale_success)
			goto failsafe;

		for (int i = 0; i < config->nargs; i++)
		{ 
			struct gforscale_kernel_symbol_t* arg = l->args + i;

			// Restore original data pointers in allocatable
			// arguments descriptors.
			if (arg->allocatable)
				*(void**)(arg->desc) = arg->dev_ref;
			
			// Free space used for symbol name.
			free(arg->name);
		}

		for (int i = 0; i < config->nmodsyms; i++)
		{
			struct gforscale_kernel_symbol_t* dep = l->deps + i;

			// Restore original data pointers in allocatable
			// modules symbols descriptors.
			if (dep->allocatable)
				*(void**)(dep->desc) = dep->dev_ref;
		}
		
		continue;
	
	failsafe :

		if (config->runmode & GFORSCALE_RUNMODE_HOST)
		{
			for (int i = 0; i < config->nargs; i++)
			{
				struct gforscale_kernel_symbol_t* arg = l->args + i;

				// Restore original data pointers in allocatable
				// arguments descriptors.
				if (arg->allocatable)
					*(void**)(arg->desc) = arg->dev_ref;
		
				// Release copied memory regions.
				if (arg->sref) free(arg->ref);
				if (arg->sdesc) free(arg->desc);
				
				// Free space used for symbol name.
				free(arg->name);
			}
			for (int i = 0; i < config->nmodsyms; i++)
			{
				struct gforscale_kernel_symbol_t* dep = l->deps + i;
				
				// Restore original data pointers in allocatable
				// modules symbols descriptors.
				if (dep->allocatable)
					*(void**)(dep->desc) = dep->dev_ref;
			
				// Release copied memory regions.
				if (dep->sref) free(dep->ref);
				if (dep->sdesc && dep->allocatable)
					free(dep->desc);		
			}
		}

		// Reset device in case error might be not recoverable.
		gforscale_reset[irunmode](l);
		
		// Disable the failing runmode.
		config->runmode &= ~runmode;
		
		gforscale_set_last_error(status);
	}	
}

