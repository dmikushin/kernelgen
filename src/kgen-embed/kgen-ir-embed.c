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

#include <CL/cl.h>
#include <malloc.h>
#include <stdio.h>
#include <string.h>

int main(int argc, char* argv[])
{
	if (argc < 2)
	{
		printf("KernelGen IR embedder CLI\n");
		printf("Usage: %s [options] source.ir\n", argv[0]);
		return 0;
	}
	
	// Take entire executable to clone some of its elf properties.
	GElf_Ehdr ehdr;
	if (kernelgen_elf_read_eheader("/proc/self/exe", &ehdr))
		return 1;
	
	int nsources = 0;
	char** sources = (char**)malloc(sizeof(char*) * argc);
	char* output = NULL;
	int output_param = 0;
	char* symbol = NULL;
	for (int i = 1; i < argc; i++)
	{
		// Skip argument following -o.
		if (output_param)
		{
			output_param = 0;
			continue;
		}
		
		char* option = argv[i];
		size_t szoption = strlen(option);
		if (strcmp(option, "-o") && strcmp(option + szoption - 3, ".ir"))
		{
			if (!strncmp(option, "-Wk,", 4))
			{
				if (!strncmp(argv[i] + 4, "--name=", 7))
				{
					char* name = argv[i] + 11;
					symbol = (char*)malloc(strlen(name) + 1);
					strcpy(symbol, name);
				}
			}
			if (!strcmp(option, "-m32"))
			{
				ehdr.e_machine = EM_386;
			}
		}
		else
		{
			if (!strcmp(option, "-o"))
			{
				output = argv[i + 1];
				output_param = 1;
			}
			else
				sources[nsources++] = argv[i];
		}
	}

	// Embed each IR source.
	for (int i = 0; i < nsources; i++)
	{
		size_t szname = strlen(sources[i]);
		char* input = (char*)malloc(sizeof(char) * (szname + 1));
		strcpy(input, sources[i]);
		
		// TODO: generate output filename, if not set.

		size_t szsource = 0;
		kernelgen_load_source(input, &sources[i], &szsource);

		// Create a symbol name from filename, if unspecified.
		if (!symbol)
		{
			symbol = (char*)malloc(strlen(basename(output)) + 1);
			strcpy(symbol, basename(output));
			for (int i = strlen(symbol) - 1; i >= 0; i--)
				if (symbol[i] == '.') symbol[i] = '\0';
		}

		// Create the kernel source symbol.
		const char* fmtsymsource = "%s_ir";
		size_t szsymsource = snprintf(NULL, 0, fmtsymsource, symbol);
		if (szsymsource <= 0)
		{
			fprintf(stderr, "Cannot determine the length of kernel source symbol\n");
			free(sources[i]);
			free(input);
			free(sources);
			free(symbol);
			return 1;
		}
		szsymsource++;
		char* symsource = (char*)malloc(szsymsource);
		sprintf(symsource, fmtsymsource, symbol);
		
		// Store IR in the output object.
		int status = kernelgen_elf_write_many(output, &ehdr, 1,
			symsource, sources[i], szsource);
		if (status)
		{
			free(sources[i]);
			free(input);
			free(sources);
			free(symbol);
			free(symsource);
			return status;
		}

		free(input);
		free(sources[i]);
		free(symbol);
		free(symsource);
	}

	free(sources);

	return 0;
}
 
