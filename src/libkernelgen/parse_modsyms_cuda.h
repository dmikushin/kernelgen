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

#ifdef HAVE_CUDA
#include <cuda_runtime.h>
#endif

#include <fcntl.h>
#include <gelf.h>
#include <libelf.h>
#include <malloc.h>
#include <stdio.h>
#include <unistd.h>

#ifdef __cplusplus
extern "C"
#endif

kernelgen_status_t kernelgen_parse_modsyms_cuda(
	struct kernelgen_launch_config_t* l,
	int* nargs, va_list list)
{
#ifdef HAVE_CUDA
	struct kernelgen_kernel_config_t* config = l->config;

	// Check with selector config for entire kernel. If it is
	// requested to be executed both on host and device, enable
	// kernel input/output data duplicating. In this mode instead
	// of reading/writing the specified memory regions directly,
	// copies are created, and GPU operates on copied data. Thus,
	// on return there are both GPU-processed results and unchanged
	// original data.
	int compare = config->compare;

	// Being quiet optimistic initially...
	kernelgen_status_t result;
	result.value = cudaSuccess;
	result.runmode = l->runmode;

	int fd = -1;
	Elf* e = NULL;
	
	if (config->nmodsyms)
	{
		// Get ELF symbols table.
		static Elf_Data* symbols = NULL;

		GElf_Shdr shdr;
		int nsymbols = 0;
		
		if (!l->deps_init)
		{
			const char* filename = "/proc/self/exe";

			int fd = open(filename, O_RDONLY);
			if (fd < 0)
			{
				kernelgen_print_error(kernelgen_launch_verbose,
					"Cannot open file %s\n", filename);
				result.value = kernelgen_error_not_found;
				goto finish;
			}

			if (elf_version(EV_CURRENT) == EV_NONE)
			{
				kernelgen_print_error(kernelgen_launch_verbose,
					"ELF library initialization failed: %s\n",
					elf_errmsg(-1));
				result.value = kernelgen_initialization_failed;
				goto finish;
			}

			e = elf_begin(fd, ELF_C_READ, NULL);
			if (!e)
			{
				kernelgen_print_error(kernelgen_launch_verbose,
					"elf_begin() failed for %s: %s\n",
					filename, elf_errmsg(-1));
				result.value = kernelgen_initialization_failed;
				goto finish;
			}

			Elf_Scn* scn = NULL;
			while ((scn = elf_nextscn(e, scn)) != NULL)
			{
				if (!gelf_getshdr(scn, &shdr))
				{
					kernelgen_print_error(kernelgen_launch_verbose,
						"gelf_getshdr() failed for %s: %s\n",
						filename, elf_errmsg(-1));
					result.value = kernelgen_initialization_failed;
					goto finish;
				}

				if (shdr.sh_type == SHT_SYMTAB)
				{
					if (shdr.sh_entsize)
						nsymbols = shdr.sh_size / shdr.sh_entsize;

					symbols = elf_getdata(scn, NULL);
					if (!symbols)
					{
						kernelgen_print_error(kernelgen_launch_verbose,
							"elf_getdata() failed for %s: %s\n",
							filename, elf_errmsg(-1));
						result.value = kernelgen_initialization_failed;
						goto finish;
					}
					break;
				}
			}

			if (!scn)
			{
				kernelgen_print_error(kernelgen_launch_verbose,
					"Cannot find valid sections in %s\n",
					filename);
				result.value = kernelgen_initialization_failed;
				goto finish;
			}
		}

		// Fill used modules symbols array.
		for (int i = 0; i < config->nmodsyms; i++)
		{
			struct kernelgen_kernel_symbol_t* dep = l->deps + i;
			dep->index = config->nargs + i;
		
			// Assign each kernel dependency with one memory
			// region - for data vector, in case it is allocatable.
			dep->mref = NULL;
			dep->mdesc = NULL;

			dep->ref = va_arg(list, void*);
			dep->size = *(size_t*)va_arg(list, size_t*);
			dep->desc = va_arg(list, void*);
		
			dep->allocatable = 0;
			
			// The case of ref and desc are different means the argument
			// is allocatable.
			if (dep->ref != dep->desc)
			{
				dep->allocatable = 1;
				
				// Dereference dep packed descriptior.
				dep->desc = *(void**)dep->desc;
			}
			
			if (compare)
			{
				// Backup reference into shadowed reference.
				dep->sref = dep->ref;
				
				// In comparison mode clone allocatable dependency
				// reference.
				dep->ref = malloc(dep->size);
				if (dep->sref)
					memcpy(dep->ref, dep->sref, dep->size);
			}
			
			// There is no need to setup mapped region for module
			// symbol, unless we execute in comparison mode or
			// argument is allocatable. If dependency is not
			// allocatable, desc is equal to ref, and its memory
			// region is always explicitly copied over instead of
			// mapping. This is because global symbols addresses
			// on device are predefined and mapping cannot be
			// performed to specific address.
			if (compare || dep->allocatable)
			{
				// If symbol is allocatable, submit its data pointer
				// memory region for mapping.
				dep->mref = l->regs + l->deps_nregions + l->args_nregions;
				struct kernelgen_memory_region_t* reg = dep->mref;
		
				// Pin region to the parent kernel argument.
				reg->symbol = dep;

				reg->shift = 0;
				reg->base = dep->ref;
				reg->size = dep->size;
#ifdef HAVE_ALIGNING
				// Compute HAVE_ALIGNING address.
				reg->shift = (size_t)dep->ref % SZPAGE;
				reg->base = dep->ref - reg->shift;

				// Compute HAVE_ALIGNING size: account overheads to page left and right borders.
				if (dep->size % SZPAGE)
					reg->size += SZPAGE - dep->size % SZPAGE;
				while ((size_t)reg->base + reg->size < (size_t)dep->ref + dep->size)
					reg->size += SZPAGE;
#endif
				l->deps_nregions++;
			}

			if (!l->deps_init)
			{
				char* symname = NULL;
				
				// In symbols table find the name of entire symbol
				// by the known address.
				for (int isymbol = 0; isymbol < nsymbols; isymbol++)
				{
					GElf_Sym symbol;
					gelf_getsym(symbols, isymbol, &symbol);
					if ((size_t)dep->desc == symbol.st_value)
					{
						// Set symbol size as in ELF table.
						dep->desc_size = symbol.st_size;
						symname = elf_strptr(e, shdr.sh_link, symbol.st_name);
						break;
					}
				}

				if (!symname)
				{
					kernelgen_print_error(kernelgen_launch_verbose,
						"Cannot determine symbol name for address %p\n", dep->desc);
					result.value = kernelgen_error_not_found;
					goto finish;
				}

				// Set symbol name.
				dep->name = (char*)malloc(strlen(symname) +
					strlen(l->kernel_name) + 2);
				strcpy(dep->name, symname);
				strcpy(dep->name + strlen(symname), "_");
				strcpy(dep->name + strlen(symname) + 1, l->kernel_name);
				dep->name[strlen(dep->name)] = '\0';
			}

			kernelgen_print_debug(kernelgen_launch_verbose,
				"dep \"%s\" ref = %p, size = %zu, desc = %p\n",
				dep->name, dep->sref, dep->size, dep->desc);
			
			if (compare || dep->allocatable)
			{
				kernelgen_print_debug(kernelgen_launch_verbose,
					"dep \"%s\" ref = %p, size = %zu duplicated to %p for results comparison\n",
					dep->name, dep->sref, dep->size, dep->ref);
			}

			kernelgen_print_debug(kernelgen_launch_verbose,
				"found module symbol %s at address %p\n", dep->name, dep->desc);

			if (!l->deps_init)
			{
				// Get the address for the same name as found, but
				// in the device memory.
				dep->dev_desc = NULL;
				cudaGetLastError();
				result.value = cudaGetSymbolAddress(&dep->dev_desc, dep->name);
				if (result.value != cudaSuccess)
				{
					kernelgen_print_error(kernelgen_launch_verbose,
						"Cannot get address of symbol \"%s\" on device, status = %d: %s\n",
						dep->name, result.value, kernelgen_get_error_string(result));
					goto finish;
				}
			}

			kernelgen_print_debug(kernelgen_launch_verbose,
				"found module symbol %s device counterpart at address %p, size = %zu\n",
				dep->name, dep->dev_desc, dep->size);
		}

		if (!l->deps_init)
			l->deps_init = 1;
	}

finish:

	if (e) elf_end(e);
	if (fd >= 0) close(fd);

	kernelgen_set_last_error(result);
	return result;
#else
	kernelgen_status_t result;
	result.value = kernelgen_success;
	result.runmode = l->runmode;
	kernelgen_set_last_error(result);
	return result;
#endif
}
