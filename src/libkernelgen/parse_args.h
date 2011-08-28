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

kernelgen_status_t kernelgen_parse_args(
	struct kernelgen_launch_config_t* l,
	int* nargs, va_list list)
{
	// Kernel config structure.
	struct kernelgen_kernel_config_t* config = l->config;

	// Check with selector config for entire kernel. If it is
	// requested to be executed both on host and device, enable
	// kernel input/output data duplicating. In this mode instead
	// of reading/writing the specified memory regions directly,
	// copies are created, and GPU operates on copied data. Thus,
	// on return there are both GPU-processed results and unchanged
	// original data.
	int compare = config->compare;
	
	// The number of memory regions.
	l->args_nregions = 0;
	
	// TODO: remove this line when CUDA order of initialization
	// issue will be fixed.
	l->deps_nregions = 0;

	// Fill kernel arguments array.
	for (int i = 0; i < *nargs; i++)
	{
		struct kernelgen_kernel_symbol_t* arg = l->args + i;
		arg->index = i;
		
		// Assign each kernel argument with one or two
		// memory regions - one for data vector, another -
		// for container descriptor if argument is allocatable.
		// Initially only first one is used.
		arg->mref = l->regs + l->deps_nregions + l->args_nregions;
		arg->mdesc = arg->mref;
		
		// Set stub for symbol name;
		char* name = (char*)malloc(sizeof(char) * 10);
		strcpy(name, "unknown");
		arg->name = name;
	
		arg->ref = va_arg(list, void*);
		arg->size = *(size_t*)va_arg(list, size_t*);
		arg->desc = va_arg(list, void*);

		kernelgen_print_debug(kernelgen_launch_verbose,
			"arg \"%s\" ref = %p, size = %zu, desc = %p\n", arg->name, arg->ref, arg->size,
			(arg->desc == arg->ref) ? arg->desc : *(void**)arg->desc);

		// Start with filling ref region.
		struct kernelgen_memory_region_t* reg = arg->mref;

		// Argument data reference may be uninitialized.
		// In this case argument is skipped.
		if (arg->ref)
		{
			// In comparison mode clone argument reference.
			if (compare)
			{
				// Backup reference into shadowed reference.
				arg->sref = arg->ref;
				
				arg->ref = malloc(arg->size);
				memcpy(arg->ref, arg->sref, arg->size);
				if (arg->desc == arg->sref) arg->desc = arg->ref;
				
				kernelgen_print_debug(kernelgen_launch_verbose,
					"arg \"%s\" ref = %p, size = %zu duplicated to %p for results comparison\n",
					arg->name, arg->sref, arg->size, arg->ref);
			}
		
			// Pin region to the parent kernel argument.
			reg->symbol = arg;

			reg->size = arg->size;
			reg->shift = 0;
			reg->base = arg->ref;
#ifdef HAVE_ALIGNING
			// Compute HAVE_ALIGNING address.
			reg->shift = (size_t)arg->ref % SZPAGE;
			reg->base = arg->ref - reg->shift;

			// Compute HAVE_ALIGNING size: account overheads to page left and right borders.
			if (arg->size % SZPAGE)
				reg->size += SZPAGE - arg->size % SZPAGE;
			while ((size_t)reg->base + reg->size < (size_t)arg->ref + arg->size)
				reg->size += SZPAGE;
#endif
			l->args_nregions++;
			reg++;
			
			arg->allocatable = 0;
		}

		if (arg->ref == arg->desc) continue;
		
		arg->allocatable = 1;
		
		// Continue with filling desc region.
		arg->mdesc = reg;

		// Pin region to the parent kernel argument.
		reg->symbol = arg;

		// The case of ref and desc are different means the argument
		// is allocatable. Dereference its packed descriptior and
		// add another mapped region.		
		arg->desc = *(void**)arg->desc;

		// In comparison mode clone argument descriptor.
		if (compare)
		{
			// Backup descriptor into shadowed descriptor.
			arg->sdesc = arg->desc;
			
			arg->desc = malloc(SZDESC);
			memcpy(arg->desc, arg->sdesc, SZDESC);
			*(void**)arg->desc = arg->ref;

			kernelgen_print_debug(kernelgen_launch_verbose,
				"arg \"%s\" desc = %p, size = %zu duplicated to %p for results comparison\n",
				arg->name, arg->sdesc, (size_t)SZDESC, arg->desc);
		}

		// XXX The exact descriptor size is not known, so we can only try to use
		// XXX big enough value to cover it.
		reg->size = SZDESC;
		reg->shift = 0;
		reg->base = arg->desc;
#ifdef HAVE_ALIGNING
		// Compute HAVE_ALIGNING address.
		reg->shift = (size_t)arg->desc % SZPAGE;
		reg->base = arg->desc - reg->shift;

		// Compute HAVE_ALIGNING size: account overheads to page left and right borders.
		if (SZDESC % SZPAGE)
			reg->size += SZPAGE - SZDESC % SZPAGE;
		while ((size_t)reg->base + reg->size < (size_t)arg->desc + SZDESC)
			reg->size += SZPAGE;
#endif
		l->args_nregions++;
	}
	
	kernelgen_status_t result;
	result.value = kernelgen_success;
	result.runmode = 0;
	return result;
}

