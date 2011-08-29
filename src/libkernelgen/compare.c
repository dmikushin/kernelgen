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

#include <assert.h>
#include <ffi.h>
#include <malloc.h>
#include <stdio.h>
#include <stdlib.h>

long kernelgen_compare_verbose = 1 << 1;

// Compare currently cached kernel results with results of the 
// regular host loop.
void kernelgen_compare_(
	struct kernelgen_kernel_config_t* config,
	kernelgen_compare_function_t compare, double* maxdiff)
{
	// In comparison mode this function call follows
	// regular CPU kernel invocation. Stop previously
	// started timer measuring its execution time.
	kernelgen_record_time_finish(config->stats);
	
	int count = 2 * (config->nargs + config->nmodsyms);
	
	// For each used runmode, populate its corresponding launch
	// config, handle data and execute the kernel.
	// NOTE We skip first runmode index, as it is always stands for
	// execution of original loop on host.
	for (int irunmode = 1; irunmode < kernelgen_runmodes_count; irunmode++)
	{
		// Being quiet optimistic initially...
		int status = 0;

		// Indirect indexing automatically has
		// unsupported runmodes skipped.
		int runmode = kernelgen_runmodes[irunmode];
		
		// Although, specific kernel may have supported
		// runmode disabled.
		if (!(config->runmode & runmode)) continue;

		// Kernel config structure.
		struct kernelgen_launch_config_t* l = config->launch + irunmode;
	
		kernelgen_print_debug(kernelgen_compare_verbose,
			"Checking %s for device runmode \"%s\"\n",
			l->kernel_name, kernelgen_runmodes_names[irunmode]);

		void** values = (void**)malloc(sizeof(void*) * (count + 1));
		values[0] = &maxdiff;
		
		// Set call arguments types (all void in our case).
		ffi_type** types = (ffi_type**)malloc(sizeof(ffi_type*) * (count + 1));
		for (int i = 0; i < count + 1; i++)
			types[i] = &ffi_type_pointer;
		
		// Get the ffi_cif handle.
		ffi_cif cif;
		ffi_status fstatus;
		if ((fstatus = ffi_prep_cif(&cif, FFI_DEFAULT_ABI,
			count + 1, &ffi_type_sint, types)) != FFI_OK)
		{
			kernelgen_print_error(kernelgen_compare_verbose,
				"Cannot get ffi_cif handle, status = %d", fstatus);
			status = kernelgen_error_ffi_setup;
			goto finish;
		}

		// Setup argument list containing data references
		// for kernel arguments and kernel modules symbols,
		// both original and copied.
		for (int i = 0, j = 0; i < config->nargs; i++, j++)
		{
			struct kernelgen_kernel_symbol_t* arg = l->args + j;
			values[i + 1] = arg->allocatable ? &arg->sdesc : &arg->sref;
		}
		for (int i = config->nargs, j = 0;
			i < config->nargs + config->nmodsyms; i++, j++)
		{
			struct kernelgen_kernel_symbol_t* dep = l->deps + j;
			values[i + 1] = dep->allocatable ? &dep->sdesc : &dep->sref;
			
			// If module symbol is not defined on device,
			// use host's for unification (then there will be
			// a fictive self-comparison for this argument).
			if (!*(void**)(values[i + 1]))
			{
				values[i + 1] = dep->allocatable ? &dep->desc : &dep->ref;
			}
		}
		for (int i = config->nargs + config->nmodsyms,
			j = 0; i < 2 * config->nargs + config->nmodsyms; i++, j++)
		{
			struct kernelgen_kernel_symbol_t* arg = l->args + j;
			values[i + 1] = arg->allocatable ? &arg->desc : & arg->ref;
		}
		for (int i = 2 * config->nargs + config->nmodsyms,
			j = 0; i < count; i++, j++)
		{
			struct kernelgen_kernel_symbol_t* dep = l->deps + j;
			values[i + 1] = dep->allocatable ? &dep->desc : &dep->ref;
		}

		// Invoke the comparison function.
		ffi_arg fresult = 0;
		ffi_call(&cif, FFI_FN(compare), &fresult, values);

		kernelgen_status_t result;
		result.runmode = l->runmode;
		
		// Decode and interpret the received result value.
		if ((int)fresult)
		{
			kernelgen_print_error(kernelgen_compare_verbose,
				"wrong results for kernel %s\n", l->kernel_name);
			
			// If result is wrong for the current runmode,
			// disable it.
			status = kernelgen_error_results_mismatch;
			l->config->runmode &= ~runmode;
		}
		else
		{
			// Check if executing entire kernel on device
			// is worthless by performance criteria.
			if (kernelgen_discard(l, config->stats, l->stats))
				l->config->runmode &= ~runmode;
			else
				kernelgen_print_debug(kernelgen_compare_verbose,
					"correct results for kernel %s\n", l->kernel_name);
		}

	finish:

		free(types);
		free(values);

		// Release copied memory regions.
		for (int i = 0; i < config->nargs; i++)
		{
			struct kernelgen_kernel_symbol_t* arg = l->args + i;
		
			if (arg->sref) free(arg->ref);
			if (arg->sdesc) free(arg->desc);
		}
		for (int i = 0; i < config->nmodsyms; i++)
		{
			struct kernelgen_kernel_symbol_t* dep = l->deps + i;
			
			if (dep->sref) free(dep->ref);
			if (dep->sdesc && dep->allocatable)
				free(dep->desc);		
		}

		result.value = status;
		kernelgen_set_last_error(result);
	}
}
