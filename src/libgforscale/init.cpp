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

#include "gforscale_int_cuda.h"
#include "gforscale_int_opencl.h"
#include "init.h"

#include <fcntl.h>
#include <gelf.h>
#include <libelf.h>
#include <malloc.h>
#include <stdarg.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>

int gforscale_runmode;
int* gforscale_runmodes;
char** gforscale_runmodes_names;
int gforscale_runmodes_count;
int gforscale_runmodes_mask;

long gforscale_debug_output;
long gforscale_error_output;

gforscale_parse_modsyms_func_t* gforscale_parse_modsyms;
gforscale_load_regions_func_t* gforscale_load_regions;
gforscale_save_regions_func_t* gforscale_save_regions;
gforscale_build_func_t* gforscale_build;
gforscale_launch_func_t* gforscale_launch;
gforscale_reset_func_t* gforscale_reset;

// Global structure to hold device-specific configs.
// Is initialized once and then copied to each individual
// kernel config.
static size_t specific_configs_size = 0;
static gforscale_specific_config_t* specific_configs;

static unsigned int count_bits(unsigned n)
{
	unsigned int masks[] =
		{ 0x55555555, 0x33333333, 0x0F0F0F0F, 0x00FF00FF, 0x0000FFFF };
	for(unsigned int i = 0; i < 5; i++)
	{
		n = (n & masks[i]) + ((n >> (1 << i)) & masks[i]);
	}
	return n;
}

// Initialize kernel routine configuration.
void gforscale_kernel_init(
	struct gforscale_kernel_config_t* config,
	int iloop, int nloops, char* name,
	int nargs, int nmodsyms)
{
	config->iloop = iloop;
	config->nloops = nloops;
	config->runmode = gforscale_runmode;
	config->nargs = nargs;
	config->nmodsyms = nmodsyms;

	// Copy routine name.
	config->routine_name = (char*)malloc(strlen(name) + 1);
	strcpy(config->routine_name, name);
	
	// Check for individual per-kernel switch.
	const char* kernel_basename_fmt = "%s_loop_%d_gforscale";
	int length = snprintf(NULL, 0, kernel_basename_fmt, name, iloop);
	if (length < 0)
	{
		// TODO: release resources!
	}
	length++;
	char* kernel_basename = (char*)malloc(length);
	sprintf(kernel_basename, kernel_basename_fmt, name, iloop);

	// Check for individual per-kernel switch.
	char* crunmode = getenv(kernel_basename);
	if (crunmode)
	{
		// Check the supplied runmode contains
		// only known bits, and at least one of them
		// is supported.
		int runmode = atoi(crunmode);
		if (runmode & gforscale_runmodes_mask)
			config->runmode = runmode;
	}
	
	// Check how many bits set in runmode.
	config->compare = (count_bits(config->runmode) > 1) &&
		(config->runmode & GFORSCALE_RUNMODE_HOST);

	// Copy device-specific configs to entire kernel config.
	config->specific = (gforscale_specific_config_t*)
		malloc(specific_configs_size);
	memcpy(config->specific, specific_configs, specific_configs_size);
	
	// For each runmode allocate space for launch config
	// structure.
	config->launch =
		(struct gforscale_launch_config_t*)malloc(
			sizeof(struct gforscale_launch_config_t) *
			gforscale_runmodes_count);

	const char* kernel_name_fmt = "%s_%s";
	for (int irunmode = 1; irunmode < gforscale_runmodes_count; irunmode++)
	{
		// Indirect indexing automatically has
		// unsupported runmodes skipped.
		int runmode = gforscale_runmodes[irunmode];
		
		// Although, specific kernel may have supported
		// runmode disabled.
		if (!(config->runmode & runmode)) continue;

		struct gforscale_launch_config_t* l = config->launch + irunmode;
		l->runmode = runmode;
		l->config = config;
		l->specific = (gforscale_specific_config_t)config->specific +
			(size_t)(config->specific[irunmode]);
		
		// Build device-specific kernel name.
		length = snprintf(NULL, 0, kernel_name_fmt,
			kernel_basename, gforscale_runmodes_names[irunmode]);
		if (length < 0)
		{
			// TODO: release resources!
		}
		length++;
		l->kernel_name = (char*)malloc(length);
		sprintf(l->kernel_name, kernel_name_fmt,
			kernel_basename, gforscale_runmodes_names[irunmode]);
		
		// Allocate array to hold memory regions.
		// Maximum possible length is x2 times number of arguments
		// plus number of modules symbols, in case all arguments
		// and modules symbols are allocatable.
		l->regs = (struct gforscale_memory_region_t*)malloc(
			sizeof(struct gforscale_memory_region_t) * (nargs * 2 + nmodsyms));
		memset(l->regs, 0,
			sizeof(struct gforscale_memory_region_t) * (nargs * 2 + nmodsyms));

		// Allocate array to hold kernel arguments.	
		l->args = (struct gforscale_kernel_symbol_t*)malloc(
			sizeof(struct gforscale_kernel_symbol_t) * nargs);
		memset(l->args, 0,
			sizeof(struct gforscale_kernel_symbol_t) * nargs);

		// Allocate array to hold kernel used modules symbols.
		l->deps = (struct gforscale_kernel_symbol_t*)malloc(
			sizeof(struct gforscale_kernel_symbol_t) * nmodsyms);
		memset(l->deps, 0,
			sizeof(struct gforscale_kernel_symbol_t) * nmodsyms);

		l->args_nregions = 0;
		l->deps_nregions = 0;

		// TODO: this is a temporary flag.
		l->deps_init = 0;

		// Build device-specific kernel source symbol name.
		length = snprintf(NULL, 0, kernel_name_fmt,
			l->kernel_name, "source");
		if (length < 0)
		{
			// TODO: release resources!
		}
		length++;
		char* kernel_source_name = (char*)malloc(length);
		sprintf(kernel_source_name, kernel_name_fmt,
			l->kernel_name, "source");

		// Build device-specific kernel binary symbol name.
		char* kernel_binary_name = (char*)malloc(length);
		sprintf(kernel_binary_name, kernel_name_fmt,
			l->kernel_name, "binary");

		// Load kernel source and binary from the
		// entire ELF image.
		int status = elf_read("/proc/self/exe", kernel_source_name,
			&l->kernel_source, &l->kernel_source_size);
		if (status)
		{
			// TODO: handle errors
		}
		status = elf_read("/proc/self/exe", kernel_binary_name,
			&l->kernel_binary, &l->kernel_binary_size);
		if (status)
		{
			// TODO: handle errors
		}

		free(kernel_source_name);
		free(kernel_binary_name);

		// Build kernel for specific device.
		gforscale_status_t result = gforscale_build[irunmode](l);
		if (result.value != gforscale_success)
		{
			// TODO: handle errors
		}
	}
	
	free(kernel_basename);
}

long gforscale_kernel_init_deps_verbose = 1 << 2;

// Initialize kernel routine static dependencies.
void gforscale_kernel_init_deps_(
	struct gforscale_kernel_config_t* config, ...)
{
	// TODO: use this implementation instead of code in launcher,
	// when CUDA order of initialization issue will be fixed.
	
	return;
}

// Release resources used by kernel configuration.
void gforscale_kernel_free(
	struct gforscale_kernel_config_t* config)
{
	for (int irunmode = 1; irunmode < gforscale_runmodes_count; irunmode++)
	{
		// Indirect indexing automatically has
		// unsupported runmodes skipped.
		int runmode = gforscale_runmodes[irunmode];
		
		// Although, specific kernel may have supported
		// runmode disabled.
		if (!(config->runmode & runmode)) continue;

		struct gforscale_launch_config_t* l = config->launch + irunmode;

		free(l->kernel_name);
		free(l->args);
		free(l->deps);
		free(l->regs);
	}

	free(config->routine_name);
	free(config->launch);
	free(config->specific);
}

// Release resources used by kernel routine static dependencies.
void gforscale_kernel_free_deps(
	struct gforscale_kernel_config_t* config)
{
	// TODO: use this implementation instead of code in launcher,
	// when CUDA order of initialization issue will be fixed.
	for (int irunmode = 1; irunmode < gforscale_runmodes_count; irunmode++)
	{
		// Indirect indexing automatically has
		// unsupported runmodes skipped.
		int runmode = gforscale_runmodes[irunmode];
		
		// Although, specific kernel may have supported
		// runmode disabled.
		if (!(config->runmode & runmode)) continue;

		struct gforscale_launch_config_t* l = config->launch + irunmode;
		
		for (int i = 0; i < config->nmodsyms; i++)
		{
			struct gforscale_kernel_symbol_t* dep = l->deps + i;
			free(dep->name);		
		}
	}
}

// Read environment variables and setup runtime global configuration.
__attribute__ ((__constructor__(101))) void gforscale_init()
{
	// Count supported runmodes
	// and sizes of their specific configs.
	gforscale_runmodes_count = 2; // host + device CPU
#ifdef HAVE_CUDA
	specific_configs_size += sizeof(gforscale_cuda_config_t);
	gforscale_runmodes_count++;
#endif
#ifdef HAVE_OPENCL
	specific_configs_size += sizeof(gforscale_opencl_config_t);
	gforscale_runmodes_count++;
#endif

	// Allocate tables for device-specific functions pointers.
	gforscale_parse_modsyms = 
		(gforscale_parse_modsyms_func_t*)malloc(
			sizeof(gforscale_parse_modsyms_func_t) *
			gforscale_runmodes_count);
	gforscale_load_regions =
		(gforscale_load_regions_func_t*)malloc(
			sizeof(gforscale_load_regions_func_t) *
			gforscale_runmodes_count);
	gforscale_save_regions =
		(gforscale_save_regions_func_t*)malloc(
			sizeof(gforscale_save_regions_func_t) *
			gforscale_runmodes_count);
	gforscale_build =
		(gforscale_build_func_t*)malloc(
			sizeof(gforscale_build_func_t) *
			gforscale_runmodes_count);
	gforscale_launch =
		(gforscale_launch_func_t*)malloc(
			sizeof(gforscale_launch_func_t) *
			gforscale_runmodes_count);
	gforscale_reset =
		(gforscale_reset_func_t*)malloc(
			sizeof(gforscale_reset_func_t) *
			gforscale_runmodes_count);

#define BIND_RUNMODE(i, suffix) \
	{ \
		gforscale_parse_modsyms[i] = gforscale_parse_modsyms_##suffix; \
		gforscale_load_regions[i] = gforscale_load_regions_##suffix; \
		gforscale_save_regions[i] = gforscale_save_regions_##suffix; \
		gforscale_build[i] = gforscale_build_##suffix; \
		gforscale_launch[i] = gforscale_launch_##suffix; \
		gforscale_reset[i] = gforscale_reset_##suffix; \
		gforscale_runmodes_names[i] = (char*)malloc(strlen(#suffix) + 1); \
		strcpy(gforscale_runmodes_names[i], #suffix); \
	}

	// Create a list of values, names and mask of supported runmodes.
	// For each supported runmode bind its device-specific functions
	// to the common index list.
	gforscale_runmodes_names =
		(char**)malloc(sizeof(char*) * gforscale_runmodes_count);
	gforscale_runmodes =
		(int*)malloc(sizeof(int) * gforscale_runmodes_count);
	gforscale_runmodes_mask = 0;
	{
		int i = 0;
		
		gforscale_runmodes[i++] = GFORSCALE_RUNMODE_HOST;
		gforscale_runmodes_mask |= GFORSCALE_RUNMODE_HOST;

		BIND_RUNMODE(i, cpu);
		gforscale_runmodes[i++] = GFORSCALE_RUNMODE_DEVICE_CPU;
		gforscale_runmodes_mask |= GFORSCALE_RUNMODE_DEVICE_CPU;

#ifdef HAVE_CUDA
		BIND_RUNMODE(i, cuda);
		gforscale_runmodes[i++] = GFORSCALE_RUNMODE_DEVICE_CUDA;
		gforscale_runmodes_mask |= GFORSCALE_RUNMODE_DEVICE_CUDA;
#endif

#ifdef HAVE_OPENCL
		BIND_RUNMODE(i, opencl);
		gforscale_runmodes[i++] = GFORSCALE_RUNMODE_DEVICE_OPENCL;
		gforscale_runmodes_mask |= GFORSCALE_RUNMODE_DEVICE_OPENCL;
#endif
	}

	// By default run everything on host.
	gforscale_runmode = GFORSCALE_RUNMODE_HOST;
	char* crunmode = getenv("gforscale_runmode");
	if (crunmode)
	{
		// Check the supplied runmode contains
		// only known bits, and at least one of them
		// is supported.
		int runmode = atoi(crunmode);
		if (runmode & gforscale_runmodes_mask)
			gforscale_runmode = runmode;
	}
	
	// By default disable all debug output.
	gforscale_debug_output = 0;

	// By default enable all error output.
	gforscale_error_output = ~gforscale_debug_output;
	
	char* cdebug = getenv("gforscale_debug_output");
	if (cdebug) gforscale_debug_output = atoi(cdebug);
	
	char* cerror = getenv("gforscale_error_output");
	if (cerror) gforscale_error_output = atoi(cerror);

	// Create array of pointers to device-specific configs.
	// Initialize device-specific settings.
	size_t offset = 
		sizeof(gforscale_specific_config_t) * gforscale_runmodes_count;
	specific_configs_size += offset;
	specific_configs =
		(gforscale_specific_config_t*)malloc(specific_configs_size);
	specific_configs[0] = NULL;
	specific_configs[1] = NULL;
	{
		int i = 2;
		gforscale_status_t result;
#ifdef HAVE_CUDA
		struct gforscale_cuda_config_t* cuda =
			(struct gforscale_cuda_config_t*)(
			(gforscale_specific_config_t)(specific_configs) + offset);
		specific_configs[i++] = (gforscale_specific_config_t)offset;
		offset += sizeof(struct gforscale_cuda_config_t);

#ifdef HAVE_ALIGNED_MAPPING
		cuda->aligned = 1;
#else
		cuda->aligned = 0;
#endif

		// Set device flag to enable memory mapping.
		cudaGetLastError();
		result.value = cudaSetDeviceFlags(cudaDeviceMapHost);
		if (result.value != cudaSuccess)
		{
			gforscale_print_error(gforscale_launch_verbose,
				"Cannot set device flags, status = %d: %s\n",
				result.value, gforscale_get_error_string(result));
			gforscale_set_last_error(result);
			// TODO: release resources!
			return;
		}
		
		// TODO: check the number of available CUDA devices!
#endif
#ifdef HAVE_OPENCL	
		struct gforscale_opencl_config_t* opencl =
			(struct gforscale_opencl_config_t*)(
			(gforscale_specific_config_t)(specific_configs) + offset);
		specific_configs[i++] = (gforscale_specific_config_t)offset;
		offset += sizeof(struct gforscale_opencl_config_t);

		// Being quiet optimistic initially...
		result.value = CL_SUCCESS;
		result.runmode = GFORSCALE_RUNMODE_DEVICE_OPENCL;
		
		// Get OpenCL platform ID.
		result.value = clGetPlatformIDs(1, &opencl->id, NULL);
		if (result.value != CL_SUCCESS)
		{
			gforscale_print_error(gforscale_launch_verbose,
				"clGetPlatformIDs returned %d: %s\n", (int)result.value,
				gforscale_get_error_string(result));
			gforscale_set_last_error(result);
			// TODO: release resources!
			return;
		}

		// Get OpenCL devices count.
		result.value = clGetDeviceIDs(opencl->id,
			CL_DEVICE_TYPE_ALL, 0, NULL, &opencl->ndevs);
		if (result.value != CL_SUCCESS)
		{
			gforscale_print_error(gforscale_launch_verbose,
				"clGetDeviceIDs returned %d: %s\n", (int)result.value,
				gforscale_get_error_string(result));
			gforscale_set_last_error(result);
			// TODO: release resources!
			return;
		}
		if (opencl->ndevs < 1)
		{
			gforscale_print_error(gforscale_launch_verbose,
				"No OpenCL devices found\n");
			result.value = gforscale_error_not_found;
			gforscale_set_last_error(result);
			// TODO: release resources!
			return;
		}

		// Get OpenCL device.
		result.value = clGetDeviceIDs(
			opencl->id, CL_DEVICE_TYPE_ALL,
			1, &opencl->device, NULL);
		if (result.value != CL_SUCCESS)
		{
			gforscale_print_error(gforscale_launch_verbose,
				"clGetDeviceIDs returned %d: %s\n", (int)result.value,
				gforscale_get_error_string(result));
			gforscale_set_last_error(result);
			// TODO: release resources!
			return;
		}

		// Create OpenCL device context.
		opencl->context = clCreateContext(
			NULL, 1, &opencl->device,
			NULL, NULL, &result.value);
		if (result.value != CL_SUCCESS)
		{
			gforscale_print_error(gforscale_launch_verbose,
				"clCreateContext returned %d: %s\n", (int)result.value,
				gforscale_get_error_string(result));
			gforscale_set_last_error(result);
			// TODO: release resources!
			return;
		}

		opencl->command_queue = clCreateCommandQueue(
			opencl->context, opencl->device, 0, &result.value);
		if (result.value != CL_SUCCESS)
		{
			gforscale_print_error(gforscale_launch_verbose,
				"clCreateCommandQueue returned %d: %s\n", (int)result.value,
				gforscale_get_error_string(result));
			gforscale_set_last_error(result);
			// TODO: release resources!
			return;
		}

		char name[20];
		result.value = clGetDeviceInfo(opencl->device,
			CL_DEVICE_NAME, 20, &name, NULL);
		if (result.value != CL_SUCCESS)
		{
			gforscale_print_error(gforscale_launch_verbose,
				"clGetDeviceInfo returned %d: %s\n", (int)result.value,
				gforscale_get_error_string(result));
			gforscale_set_last_error(result);
			// TODO: release resources!
			return;
		}

		gforscale_print_debug(gforscale_launch_verbose,
			"OpenCL engine uses device \"%s\"\n", name);
#endif
	}
}

// Release resources used by runtime global configuration.
__attribute__ ((__destructor__(101))) void gforscale_free()
{
	free(gforscale_parse_modsyms);
	free(gforscale_load_regions);
	free(gforscale_save_regions);
	free(gforscale_build);
	free(gforscale_launch);
	free(gforscale_reset);
	free(gforscale_runmodes);
	for (int i = 1; i < gforscale_runmodes_count; i++)
		free(gforscale_runmodes_names[i]);
	free(gforscale_runmodes_names);
}

