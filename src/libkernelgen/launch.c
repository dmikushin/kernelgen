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
#include "stats.h"

#include <malloc.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

long kernelgen_launch_verbose = 1;

// Launch kernel with the specified 3D compute grid, arguments and modules symbols.
void kernelgen_launch_(
	struct kernelgen_kernel_config_t* config,
	int* bx, int* ex, int* by, int* ey, int* bz, int* ez,
	int* nargs, int* nmodsyms, ...)
{
	// Discard error status.
	while (kernelgen_pop_last_error().value != kernelgen_success);
	
	// Initialize devices.
	// TODO: thread-safe!
	if (kernelgen_thread_id != pthread_self())
		kernelgen_init_thread();
	
	// For each used runmode, populate its corresponding launch
	// config, handle data and execute the kernel.
	// NOTE We skip first runmode index, as it is always stands for
	// execution of original loop on host.
	for (int irunmode = 1, nmapped = 0, nregions; irunmode < kernelgen_runmodes_count; irunmode++)
	{
		// Indirect indexing automatically has
		// unsupported runmodes skipped.
		int runmode = kernelgen_runmodes[irunmode];
		
		// Although, specific kernel may have supported
		// runmode disabled.
		if (!(config->runmode & runmode & kernelgen_thread_runmode)) continue;

		// Kernel config structure.
		struct kernelgen_launch_config_t* l = config->launch + irunmode;

		kernelgen_print_debug(kernelgen_launch_verbose,
			"Launching %s for device %s:%d runmode \"%s\"\n", l->kernel_name,
			kernelgen_platforms_names[kernelgen_thread_platform_index],
			kernelgen_thread_device_index, kernelgen_runmodes_names[irunmode]);

		// Being quiet optimistic initially...
		kernelgen_status_t status;
		status.value = kernelgen_success;
		status.runmode = runmode;
		
		// If kernel was not previously executed on entire device,
		// rebuild it.
		kernelgen_status_t result = kernelgen_build[irunmode](l);
		if (result.value != kernelgen_success)
		{
			goto failsafe;
		}

		va_list list;
		va_start(list, nmodsyms);
		status = kernelgen_parse_args(l, nargs, list);
		va_end(list);
		
		if (status.value != kernelgen_success)
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
		status = kernelgen_parse_modsyms[irunmode](l, nmodsyms, list);
		va_end(list);

		if (status.value != kernelgen_success)
		{
			va_end(list);
			goto failsafe;
		}

		// Merge memory regions into non-overlapping regions.
		nregions = l->args_nregions + l->deps_nregions;
		status = kernelgen_merge_regions(l->regs, nregions);
	
		// Map or load regions into device memory space.
		status = kernelgen_load_regions[irunmode](l, &nmapped);

		if (status.value != kernelgen_success)
			goto failsafe;

		status = kernelgen_launch[irunmode](l, bx, ex, by, ey, bz, ez);

		if (status.value != kernelgen_success)
			goto failsafe;

		// Unmap or save regions from device memory space.
		status = kernelgen_save_regions[irunmode](l, nmapped);

		if (status.value != kernelgen_success)
			goto failsafe;

		for (int i = 0; i < config->nargs; i++)
		{ 
			struct kernelgen_kernel_symbol_t* arg = l->args + i;

			// Restore original data pointers in allocatable
			// arguments descriptors.
			if (arg->allocatable)
				*(void**)(arg->desc) = arg->dev_ref;
			
			// Free space used for symbol name.
			free(arg->name);
		}

		for (int i = 0; i < config->nmodsyms; i++)
		{
			struct kernelgen_kernel_symbol_t* dep = l->deps + i;

			// Restore original data pointers in allocatable
			// modules symbols descriptors.
			if (dep->allocatable)
				*(void**)(dep->desc) = dep->dev_ref;
		}
		
		continue;
	
	failsafe :

		if (config->runmode & KERNELGEN_RUNMODE_HOST)
		{
			for (int i = 0; i < config->nargs; i++)
			{
				struct kernelgen_kernel_symbol_t* arg = l->args + i;

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
				struct kernelgen_kernel_symbol_t* dep = l->deps + i;
				
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
		kernelgen_reset[irunmode](l);
		
		// Disable the failing runmode.
		config->runmode &= ~runmode;
		
		kernelgen_set_last_error(status);
	}

	// In comparison mode this function call preceedes
	// regular CPU kernel invocation. Start timer to
	// measure its execution time.
	if (config->runmode != KERNELGEN_RUNMODE_HOST)
		kernelgen_record_time_start(config->stats);
}

