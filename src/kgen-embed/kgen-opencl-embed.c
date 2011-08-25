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
	kernelgen_status_t status;
	status.runmode = KERNELGEN_RUNMODE_DEVICE_OPENCL;

	if (argc < 2)
	{
		printf("KernelGen OpenCL kernels compiler CLI\n");
		printf("Usage: %s [options] source.cl\n", argv[0]);
		return 0;
	}
	
	// Take entire executable to clone some of its elf properties.
	GElf_Ehdr ehdr;
	status.value = kernelgen_elf_read_eheader("/proc/self/exe", &ehdr);
	if (status.value != CL_SUCCESS)
		return status.value;
	
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
	int output_param = -1;
	char* symbol = NULL;
	for (int i = 1; i < argc; i++)
	{
		// Skip argument following -o.
		if (output_param != -1)
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
	status.value = clGetPlatformIDs(1, &id, NULL);
	if (status.value != CL_SUCCESS)
	{
		fprintf(stderr, "clGetPlatformIDs returned %d: %s\n",
			(int)status.value, kernelgen_get_error_string(status));
		free(options);
		free(sources);
		return 1;
	}

	// Get OpenCL devices count.
	int opencl_devices_count;
	status.value = clGetDeviceIDs(id, CL_DEVICE_TYPE_ALL,
		0, NULL, &opencl_devices_count);
	if (status.value != CL_SUCCESS)
	{
		fprintf(stderr, "clGetDeviceIDs returned %d: %s\n",
			(int)status.value, kernelgen_get_error_string(status));
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
	status.value = clGetDeviceIDs(id, CL_DEVICE_TYPE_ALL,
		1, &device, NULL);
	if (status.value != CL_SUCCESS)
	{
		fprintf(stderr, "clGetDeviceIDs returned %d: %s\n",
			(int)status.value, kernelgen_get_error_string(status));
		free(options);
		free(sources);
		return 1;
	}
	
	// Create device context.
	cl_context context = clCreateContext(
		NULL, 1, &device, NULL, NULL, &status.value);
	if (status.value != CL_SUCCESS)
	{
		fprintf(stderr, "clCreateContext returned %d: %s\n",
			(int)status.value, kernelgen_get_error_string(status));
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

		// Generate output filename, if not set.
		if (output_param == -1)
		{
			output = (char*)malloc(sizeof(char*) * (szname + 1));
			strcpy(output, input);
			strcpy(output + szname - 2, "o");
		}

		size_t szsource = 0;
		kernelgen_load_source(input, &sources[i], &szsource);

		cl_program program = clCreateProgramWithSource(
			context, 1, (const char**)&sources[i], &szsource, &status.value);
		if (status.value != CL_SUCCESS)
		{
			fprintf(stderr, "clCreateProgramWithSource returned %d: %s\n",
				(int)status.value, kernelgen_get_error_string(status));
			free(options);
			free(sources[i]);
			free(input);
			if (output_param == -1) free(output);
			free(sources);
			return 1;
		}

		status.value = clBuildProgram(program, 1, &device, NULL /*options*/, NULL, NULL);
		if (status.value != CL_SUCCESS)
		{
			fprintf(stderr, "clBuildProgram returned %d: %s\n",
				(int)status.value, kernelgen_get_error_string(status));
			status.value = clGetProgramBuildInfo(program, device,
				CL_PROGRAM_BUILD_LOG, szsource, sources[i], NULL);
			if (status.value != CL_SUCCESS)
			{
				fprintf(stderr, "clGetProgramBuildInfo returned %d: %s\n",
					(int)status.value, kernelgen_get_error_string(status));
			}
			else
			{
				fprintf(stderr, "%s\n", sources[i]);
			}
			free(options);
			free(sources[i]);
			free(input);
			if (output_param == -1) free(output);
			free(sources);
			return 1;
		}

		// Get the OpenCL program binary size.
		size_t szbinary = 0;
		status.value = clGetProgramInfo(program, CL_PROGRAM_BINARY_SIZES,
			sizeof(size_t), &szbinary, NULL);
		if (status.value != CL_SUCCESS)
		{
			fprintf(stderr, "clGetProgramInfo returned %d: %s\n",
				(int)status.value, kernelgen_get_error_string(status));
			free(options);
			free(sources[i]);
			free(input);
			if (output_param == -1) free(output);
			free(sources);
			return 1;
		}

		// Get the OpenCL program binary.
		unsigned char* binary = (unsigned char*)malloc(szbinary);
		status.value = clGetProgramInfo(program, CL_PROGRAM_BINARIES,
			sizeof(unsigned char*), &binary, NULL);
		if (status.value != CL_SUCCESS)
		{
			fprintf(stderr, "clGetProgramInfo returned %d: %s\n",
				(int)status.value, kernelgen_get_error_string(status));
			free(options);
			free(sources[i]);
			free(input);
			if (output_param == -1) free(output);
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
	
		// Create the kernel binary symbol.
		const char* fmtsymbinary = "%s_binary";
		size_t szsymbinary = snprintf(NULL, 0, fmtsymbinary, symbol);
		if (szsymbinary <= 0)
		{
			fprintf(stderr, "Cannot determine the length of kernel binary symbol\n");
			free(options);
			free(sources[i]);
			free(input);
			if (output_param == -1) free(output);
			free(sources);
			free(binary);
			free(symbol);
			return 1;			
		}
		szsymbinary++;
		char* symbinary = (char*)malloc(szsymbinary);
		sprintf(symbinary, fmtsymbinary, symbol);
		
		// Store OpenCL program binary in the output object.
		status.value = kernelgen_elf_write_many(output, &ehdr, 1,
			symbinary, binary, szbinary);
		if (status.value)
		{
			free(options);
			free(sources[i]);
			free(input);
			if (output_param == -1) free(output);
			free(sources);
			free(binary);
			free(symbol);
			free(symbinary);
			return status.value;
		}

		free(input);
		if (output_param == -1) free(output);
		free(sources[i]);
		free(binary);
		free(symbol);
		free(symbinary);
	}

	free(options);
	free(sources);

	return 0;
}
