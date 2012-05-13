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
 * 
 * Created with help of:
 * 	"libelf by Example" by Joseph Koshy
 * 	"Executable and Linkable Format (ELF)"
 * 	people on #gcc@irc.oftc.net
 */

#include "elf_write.h"

#include <ffi.h>
#include <malloc.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>

int main(int argc, char **argv)
{
	if (argc != 5)
	{
		fprintf(stdout, "Purpose: write ELF image containg specified symbols and values\n");
		fprintf(stdout, "Usage: %s <filename> <count> <base_symname> <base_symdata>\n", argv[0]);
		return 0;
	}
	
	char* filename = argv[1];
	int count = atoi(argv[2]);
	char* symname = argv[3];
	char* symdata = argv[4];
	
	if (count <= 0)
	{
		fprintf(stderr, "Symbols count must be a positive value!\n");
		return 1;
	}
	
	int status = 0;
	if (count == 1)
	{
		status = util_elf_write(filename, symname, symdata, strlen(symdata) + 1);
	}
	else
	{
		void** args = NULL;
		char **symnames = NULL, **symdatas = NULL;
		size_t* lengths = NULL;
	
		// Set ffi call arguments types.
		int nargs = 3 * count + 2;
		ffi_type** types = (ffi_type**)malloc(sizeof(ffi_type*) * nargs);
		types[0] = &ffi_type_pointer; // filename
		types[1] = &ffi_type_sint; // arguments count
		ffi_type* ffi_type_size_t =
			(sizeof(size_t) == sizeof(unsigned int)) ? &ffi_type_uint : &ffi_type_ulong;
		for (int i = 0; i < 3 * count; i += 3)
		{
			types[i + 2] = &ffi_type_pointer; // symbol name
			types[i + 3] = &ffi_type_pointer; // symbol data
			types[i + 4] = ffi_type_size_t; // symbol data length
		}
	
		// Get the ffi_cif handle.
		ffi_cif cif;
		ffi_status fstatus;
		if ((fstatus = ffi_prep_cif(&cif, FFI_DEFAULT_ABI,
			nargs, &ffi_type_sint, types)) != FFI_OK)
		{
			fprintf(stderr,	"Cannot get ffi_cif handle, status = %d", fstatus);
			status = 1;
			goto finish;
		}

		const char* fmt = "%s_%d";
		args = (void**)malloc(sizeof(void*) * nargs);
		args[0] = &filename;
		args[1] = &count;
		symnames = (char**)malloc(sizeof(char*) * count);
		symdatas = (char**)malloc(sizeof(char*) * count);
		memset(symnames, 0, sizeof(char*) * count);
		memset(symdatas, 0, sizeof(char*) * count);
		lengths = (size_t*)malloc(sizeof(size_t) * count);
		for (int i = 0; i < count; i++)
		{
			// Generate unique symbol name, using the specified base.
			size_t szsymname = snprintf(NULL, 0, fmt, symname, i);
			if (szsymname < 0)
			{
				fprintf(stderr, "Cannot determine the length of the symbol\n");
				status = 1;
				goto finish;
			}
			szsymname++;
			symnames[i] = (char*)malloc(sizeof(char) * szsymname);
			sprintf(symnames[i], fmt, symname, i);
			
			// Generate unique symbol data, using the specified base.
			const char* fmtsymdata = "%s_%d";
			size_t szsymdata = snprintf(NULL, 0, fmt, symdata, i);
			if (szsymdata < 0)
			{
				fprintf(stderr, "Cannot determine the length of the data item\n");
				status = 1;
				goto finish;
			}
			szsymdata++;
			symdatas[i] = (char*)malloc(sizeof(char) * szsymdata);
			sprintf(symdatas[i], fmt, symdata, i);
			lengths[i] = szsymdata;
			
			args[3 * i + 2] = &symnames[i];
			args[3 * i + 3] = &symdatas[i];
			args[3 * i + 4] = &lengths[i];
		}

		// Invoke ELF writing function.
		ffi_arg result = 0;
		ffi_call(&cif, FFI_FN(elf_write_many), &result, args);
		status = (int)result;
		
finish:
		free(types);

		if (args && symnames && symdatas && lengths)
		{
			free(args);
			for (int i = 0; i < count; i++)
			{
				if (symnames[i]) free(symnames[i]);
				if (symdatas[i]) free(symdatas[i]);
			}
			free(symnames);
			free(symdatas);
			free(lengths);
		}
	}
	return status;
}

