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

#include "elf_write.h"

#include <CL/cl.h>
#include <fcntl.h>
#include <libelf.h>
#include <malloc.h>
#include <stdio.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

#define SZLINE 1024
#define NLINES 1024

static int load_source(const char* filename, char** source, size_t* szsource)
{
	if (!filename)
	{
		fprintf(stderr, "Invalid filename pointer\n");
		return 1;
	}
	if (!source)
	{
		fprintf(stderr, "Invalid source pointer\n");
		return 1;
	}
	if (!szsource)
	{
		fprintf(stderr, "Invalid size pointer\n");
		return 1;
	}
	FILE * fp = fopen(filename, "r");
	if (!fp)
	{
		fprintf(stderr, "Cannot open file %s\n", filename);
		return 1;
	}
	struct stat st; stat(filename, &st);
	*szsource = st.st_size;
	*source = (char*)malloc(sizeof(char) * *szsource);
	fread(*source, *szsource, 1, fp);
	int ierr = ferror(fp);
	fclose(fp);
	if (ierr)
	{
		fprintf(stderr, "Error reading from %s, code = %d", filename, ierr);
		return 1;
	}
	return 0;
}

static int load_header(GElf_Ehdr* ehdr)
{
	int status = 0;
	Elf* e = NULL;
	
	int fd = open("/proc/self/exe", O_RDONLY);
	if (fd < 0)
	{
		fprintf(stderr, "Cannot open file %s\n", "/proc/self/exe");
		status = 1;
		goto finish;
	}

	if (elf_version(EV_CURRENT) == EV_NONE)
	{
		fprintf(stderr, "ELF library initialization failed: %s\n",
			elf_errmsg(-1));
		status = 1;
		goto finish;
	}
	
	e = elf_begin(fd, ELF_C_READ, NULL);
	if (!e)
	{
		fprintf(stderr, "elf_begin() failed for %s: %s\n",
			"/proc/self/exe", elf_errmsg(-1));
		status = 1;
		goto finish;
	}

	// Get executable elf program header.
	if (!gelf_getehdr(e, ehdr))
	{
		fprintf(stderr, "gelf_getehdr() failed: %s\n",
			elf_errmsg(-1));
		status = 1;
		goto finish;
	}

finish:

	if (e)	elf_end(e);
	if (fd >= 0) close(fd);
	return status;
}

int main(int argc, char* argv[])
{
	if (argc < 2)
	{
		printf("KernelGen OpenCL kernels compiler CLI\n");
		printf("Usage: %s [options] source.cl\n", argv[0]);
		return 0;
	}
	
	// Take entire executable to clone some of its elf properties.
	GElf_Ehdr ehdr;
	if (load_header(&ehdr)) return 1;
	
	// Count the total length of command line arguments.
	int szoptions = 0;
	for (int i = 1; i < argc; i++)
		szoptions += strlen(argv[i]) + 1;
	
	// Create common options string and fill it with
	// command line arguments, except those representing
	// OpenCL source files names. Place source filenames
	// in the end.
	char* options = (char*)malloc(sizeof(char) * (szoptions + 1));
	options[0] = '\0';
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
		if (strcmp(option, "-o") && strcmp(option + szoption - 3, ".cl"))
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
			else if (!strcmp(option, "-m32"))
			{
				ehdr.e_machine = EM_386;
			}
			else
			{
				// Append to OpenCL options.
				strcat(options, option);
				strcat(options, " ");
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

	// Get OpenCL platform ID.
	cl_platform_id id;
	cl_int status = clGetPlatformIDs(1, &id, NULL);
	if (status != CL_SUCCESS)
	{
		fprintf(stderr, "clGetPlatformIDs returned %d\n", (int)status);
		free(options);
		free(sources);
		return 1;
	}

	// Get OpenCL devices count.
	int opencl_devices_count;
	status = clGetDeviceIDs(id, CL_DEVICE_TYPE_ALL,
		0, NULL, &opencl_devices_count);
	if (status != CL_SUCCESS)
	{
		fprintf(stderr, "clGetDeviceIDs returned %d\n", (int)status);
		free(options);
		free(sources);
		return 1;
	}
	if (opencl_devices_count < 1)
	{
		fprintf(stderr, "No OpenCL devices found\n");
		free(options);
		free(sources);
		return 1;
	}

	// Get OpenCL devices.
	cl_device_id device;
	status = clGetDeviceIDs(id, CL_DEVICE_TYPE_ALL,
		1, &device, NULL);
	if (status != CL_SUCCESS)
	{
		fprintf(stderr, "clGetDeviceIDs returned %d\n", (int)status);
		free(options);
		free(sources);
		return 1;
	}
	
	// Create device context.
	cl_context context = clCreateContext(
		NULL, 1, &device, NULL, NULL, &status);
	if (status != CL_SUCCESS)
	{
		fprintf(stderr, "clCreateContext returned %d\n", (int)status);
		free(options);
		free(sources);
		return 1;
	}

	// Build each OpenCL source code file.
	for (int i = 0; i < nsources; i++)
	{
		size_t szname = strlen(sources[i]);
		char* input = (char*)malloc(sizeof(char) * (szname + 1));
		strcpy(input, sources[i]);
		
		// TODO: generate output filename, if not set.

		size_t szsource = 0;
		load_source(input, &sources[i], &szsource);

		cl_program program = clCreateProgramWithSource(
			context, 1, (const char**)&sources[i], &szsource, &status);
		if (status != CL_SUCCESS)
		{
			fprintf(stderr, "clCreateProgramWithSource returned %d\n", (int)status);
			free(options);
			free(sources[i]);
			free(input);
			free(sources);
			return 1;
		}

		status = clBuildProgram(program, 1, &device, NULL, NULL, NULL);
		if (status != CL_SUCCESS)
		{
			clGetProgramBuildInfo(program, device,
				CL_PROGRAM_BUILD_LOG, szsource, sources[i], NULL);
			fprintf(stderr, "%s\n", sources[i]);
			free(options);
			free(sources[i]);
			free(input);
			free(sources);
			return 1;
		}

		// Get the OpenCL program binary size.
		size_t szbinary = 0;
		status = clGetProgramInfo(program, CL_PROGRAM_BINARY_SIZES,
			sizeof(size_t), &szbinary, NULL);
		if (status != CL_SUCCESS)
		{
			fprintf(stderr, "clGetProgramInfo returned %d\n", (int)status);
			free(options);
			free(sources[i]);
			free(input);
			free(sources);
			return 1;
		}

		// Get the OpenCL program binary.
		unsigned char* binary = (unsigned char*)malloc(szbinary);
		status = clGetProgramInfo(program, CL_PROGRAM_BINARIES,
			sizeof(unsigned char*), &binary, NULL);
		if (status != CL_SUCCESS)
		{
			fprintf(stderr, "clGetProgramInfo returned %d\n", (int)status);
			free(options);
			free(sources[i]);
			free(input);
			free(sources);
			free(binary);
			return 1;
		}

		// Create a symbol name from filename, if unspecified.
		if (!symbol)
		{
			symbol = (char*)malloc(strlen(basename(output)) + 1);
			strcpy(symbol, basename(output));
			for (int i = strlen(symbol) - 1; i >= 0; i--)
				if (symbol[i] == '.') symbol[i] = '\0';
		}

		// Create the kernel source symbol.
		const char* fmtsymsource = "%s_source";
		size_t szsymsource = snprintf(NULL, 0, fmtsymsource, symbol);
		if (szsymsource <= 0)
		{
			fprintf(stderr, "Cannot determine the length of kernel source symbol\n");
			free(options);
			free(sources[i]);
			free(input);
			free(sources);
			free(binary);
			free(symbol);
			return 1;
		}
		szsymsource++;
		char* symsource = (char*)malloc(szsymsource);
		sprintf(symsource, fmtsymsource, symbol);
		
		// Create the kernel binary symbol.
		const char* fmtsymbinary = "%s_binary";
		size_t szsymbinary = snprintf(NULL, 0, fmtsymbinary, symbol);
		if (szsymbinary <= 0)
		{
			fprintf(stderr, "Cannot determine the length of kernel binary symbol\n");
			free(options);
			free(sources[i]);
			free(input);
			free(sources);
			free(binary);
			free(symbol);
			free(symsource);
			return 1;			
		}
		szsymbinary++;
		char* symbinary = (char*)malloc(szsymbinary);
		sprintf(symbinary, fmtsymbinary, symbol);
		
		// Store OpenCL program binary in the output object.
		int status = elf_write_many(output, &ehdr, 2,
			symsource, sources[i], szsource,
			symbinary, binary, szbinary);
		if (status)
		{
			free(options);
			free(sources[i]);
			free(input);
			free(sources);
			free(binary);
			free(symbol);
			free(symsource);
			free(symbinary);
			return status;
		}

		free(input);
		free(sources[i]);
		free(binary);
		free(symbol);
		free(symsource);
		free(symbinary);
	}

	free(options);
	free(sources);

	return 0;
}
