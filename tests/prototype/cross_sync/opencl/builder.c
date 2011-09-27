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
 */

#include <malloc.h>
#include <stdio.h>
#include <string.h>
#include <sys/stat.h>

#include "builder.h"
#include "error.h"

// Init extension function pointers
#define INIT_CL_EXT_FCN_PTR(name) \
	if(!pfn_##name) { \
		pfn_##name = (name##_fn) clGetExtensionFunctionAddress(#name); \
		if(!pfn_##name) { \
			fprintf(stderr, "Cannot get pointer to ext. fcn. " #name); \
		goto failure; \
	} \
}

static clCreateSubDevicesEXT_fn pfn_clCreateSubDevicesEXT = NULL;
static clReleaseDeviceEXT_fn pfn_clReleaseDeviceEXT = NULL;

// Load contents of the specified text file.
int load_source(const char* filename, char** source, size_t* szsource)
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

int builder_deinit(builder_config_t* config)
{
	if (!config)
	{
		fprintf(stderr, "Invalid builder config\n");
		return 1;
	}
	free(config);
	return 0;
}

builder_config_t* builder_init(
	const char* filename, const char* options, int nkernels)
{
	// Initialize clCreateSubDevicesEXT and clReleaseDeviceEXT function pointers
	INIT_CL_EXT_FCN_PTR(clCreateSubDevicesEXT);
	INIT_CL_EXT_FCN_PTR(clReleaseDeviceEXT);

	char* source = NULL;

	// Create builder config structure.
	builder_config_t* config =
		(builder_config_t*)malloc(sizeof(builder_config_t) +
			sizeof(cl_kernel) * nkernels);
	config->kernels = (cl_kernel*)(config + 1);

	// Get OpenCL platform ID.
	cl_int status = clGetPlatformIDs(1, &config->id, NULL);
	if (status != CL_SUCCESS)
	{
		fprintf(stderr, "clGetPlatformIDs returned %d: %s\n",
			(int)status, get_error_string(status));
		goto failure;
	}

	size_t length;
	status = clGetPlatformInfo(
		config->id, CL_PLATFORM_VENDOR, 0, NULL, &length);
	char* name = (char*)malloc(sizeof(char) * (length + 1));
	status = clGetPlatformInfo(
		config->id, CL_PLATFORM_VENDOR, length,
		name, NULL);
	name[length] = '\0';
	printf("Using platform %s\n", name);
	free(name);

	// Get OpenCL devices count.
	status = clGetDeviceIDs(config->id, CL_DEVICE_TYPE_ALL,
		0, NULL, &config->count);
	if (status != CL_SUCCESS)
	{
		fprintf(stderr, "clGetDeviceIDs returned %d: %s\n",
			(int)status, get_error_string(status));
		goto failure;
	}
	if (config->count < 1)
	{
		fprintf(stderr, "No OpenCL devices found\n");
		goto failure;
	}

	// Get OpenCL device.
	status = clGetDeviceIDs(config->id, CL_DEVICE_TYPE_ALL,
		1, config->devices, NULL);
	if (status != CL_SUCCESS)
	{
		fprintf(stderr, "clGetDeviceIDs returned %d: %s\n",
			(int)status, get_error_string(status));
		goto failure;
	}

	// Get two OpenCL subdevices (fission).
	cl_device_partition_property_ext part_props[3] =
	{
		CL_DEVICE_PARTITION_EQUALLY_EXT, 
		1, 
		CL_PROPERTIES_LIST_END_EXT
	};
	cl_uint ret_count = 0;
	status = pfn_clCreateSubDevicesEXT(
		config->devices[0], part_props, 0,
		NULL, &ret_count);
	if (status != CL_SUCCESS)
	{
		fprintf(stderr, "clCreateSubDevicesEXT1 returned %d: %s\n",
			(int)status, get_error_string(status));
		goto failure;
	}
	if (ret_count < 2)
	{
		fprintf(stderr, "Not enough subdevices available: %d\n", ret_count);
		goto failure;
	}
	status = pfn_clCreateSubDevicesEXT(
		config->devices[0], part_props, 2,
		config->devices + 1, &ret_count);
	if (status != CL_SUCCESS)
	{
		fprintf(stderr, "clCreateSubDevicesEXT returned %d: %s\n",
			(int)status, get_error_string(status));
		goto failure;
	}
	if (ret_count != 2)
	{
		fprintf(stderr, "Not all subdevices are available\n");
		goto failure;
	}

	// Create device context.
	config->context = clCreateContext(
		NULL, 1, &config->devices[0], NULL, NULL, &status);
	if (status != CL_SUCCESS)
	{
		fprintf(stderr, "clCreateContext returned %d: %s\n",
			(int)status, get_error_string(status));
		goto failure;
	}

	size_t szsource = 0;
	if (load_source(filename, &source, &szsource))
		goto failure;

	config->program = clCreateProgramWithSource(
		config->context, 1, (const char**)&source,
		&szsource, &status);
	if (status != CL_SUCCESS)
	{
		fprintf(stderr, "clCreateProgramWithSource returned %d: %s\n",
			(int)status, get_error_string(status));
		goto failure;
	}

	status = clBuildProgram(config->program, 1,
		&config->devices[0], options, NULL, NULL);
	if (status != CL_SUCCESS)
	{
		fprintf(stderr, "clBuildProgram returned %d: %s\n",
			(int)status, get_error_string(status));
		status = clGetProgramBuildInfo(
			config->program, config->devices[0],
			CL_PROGRAM_BUILD_LOG, szsource, source, NULL);
		if (status != CL_SUCCESS)
			fprintf(stderr, "clGetProgramBuildInfo returned %d: %s\n",
				(int)status, get_error_string(status));
		else
			fprintf(stderr, "%s\n", source);
		goto failure;
	}

	// Create OpenCL kernels.
	config->kernels = (cl_kernel*)malloc(sizeof(cl_kernel) * nkernels);
	cl_int nkernels_out;
	status = clCreateKernelsInProgram(config->program,
		nkernels, config->kernels, &nkernels_out);
	if (status != CL_SUCCESS)
	{
		fprintf(stderr, "clCreateKernel returned %d: %s\n",
			(int)status, get_error_string(status));
		goto failure;
	}
	if (nkernels != nkernels_out)
	{
		fprintf(stderr, "Not all kernels compiled: %d (expected %d)\n",
			nkernels_out, nkernels);
		goto failure;
	}

	free(source);
	return config;

failure:
	free(config);
	if (source) free(source);
	return NULL;
}

