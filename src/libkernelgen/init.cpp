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

#include "kernelgen_int_cuda.h"
#include "kernelgen_int_opencl.h"
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

// Defines per-process execution mode setting.
int kernelgen_process_runmode;

// Each host thread has individual runmode setting,
// that is a combination of process runmode setting
// and capabilities of the currently selected device.
__thread int kernelgen_thread_runmode;

// Each host thread keeps indexes of currently used platform and device.
__thread int kernelgen_thread_platform_index = 0;
__thread int kernelgen_thread_device_index = 0;

int* kernelgen_runmodes;
char** kernelgen_runmodes_names;
int kernelgen_runmodes_count;
int kernelgen_runmodes_mask;

long kernelgen_debug_output;
long kernelgen_error_output;

kernelgen_parse_modsyms_func_t* kernelgen_parse_modsyms;
kernelgen_load_regions_func_t* kernelgen_load_regions;
kernelgen_save_regions_func_t* kernelgen_save_regions;
kernelgen_build_func_t* kernelgen_build;
kernelgen_launch_func_t* kernelgen_launch;
kernelgen_reset_func_t* kernelgen_reset;

// Platforms handles and count.
#ifdef HAVE_OPENCL
cl_platform_id* kernelgen_platforms;
#endif
int kernelgen_platforms_count;
char** kernelgen_platforms_names;

// Devices handles and count.
#ifdef HAVE_OPENCL
cl_device_id** kernelgen_devices;
#endif
int* kernelgen_devices_count;

// Devices contexts and command queues.
#ifdef HAVE_OPENCL
cl_context** kernelgen_contexts;
cl_command_queue** kernelgen_queues;
#endif

// Global structure to hold device-specific configs.
// Is initialized once and then copied to each individual
// kernel config.
static size_t specific_configs_size = 0;
static kernelgen_specific_config_t* specific_configs;

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
void kernelgen_kernel_init(
	struct kernelgen_kernel_config_t* config,
	int iloop, int nloops, char* name,
	int nargs, int nmodsyms)
{
	config->iloop = iloop;
	config->nloops = nloops;
	config->nargs = nargs;
	config->nmodsyms = nmodsyms;

	// Since this code is executed during *process* initialization,
	// it should be fine to use the same runmode for entire kernel.
	config->runmode = kernelgen_process_runmode;

	// Copy routine name.
	config->routine_name = (char*)malloc(strlen(name) + 1);
	strcpy(config->routine_name, name);
	
	// Check for individual per-kernel switch.
	const char* kernel_basename_fmt = "%s_loop_%d_kernelgen";
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
		if (runmode & kernelgen_runmodes_mask)
			config->runmode = runmode;
	}

	// Set value for platform and device indexes in a way
	// that kernel must be always compiled initially.
	config->last_platform_index = -1;
	config->last_device_index = -1;
	
	// Check how many bits set in runmode.
	config->compare = (count_bits(config->runmode) > 1) &&
		(config->runmode & KERNELGEN_RUNMODE_HOST);

	// Copy device-specific configs to entire kernel config.
	config->specific = (kernelgen_specific_config_t*)
		malloc(specific_configs_size);
	memcpy(config->specific, specific_configs, specific_configs_size);
	
	// For each runmode allocate space for launch config
	// structure.
	config->launch =
		(struct kernelgen_launch_config_t*)malloc(
			sizeof(struct kernelgen_launch_config_t) *
			kernelgen_runmodes_count);

	const char* kernel_name_fmt = "%s_%s";
	for (int irunmode = 1; irunmode < kernelgen_runmodes_count; irunmode++)
	{
		// Indirect indexing automatically has
		// unsupported runmodes skipped.
		int runmode = kernelgen_runmodes[irunmode];
		
		// Although, specific kernel may have supported
		// runmode disabled.
		if (!(config->runmode & runmode)) continue;

		struct kernelgen_launch_config_t* l = config->launch + irunmode;
		l->runmode = runmode;
		l->config = config;
		l->specific = (kernelgen_specific_config_t)config->specific +
			(size_t)(config->specific[irunmode]);
		
		// Build device-specific kernel name.
		length = snprintf(NULL, 0, kernel_name_fmt,
			kernel_basename, kernelgen_runmodes_names[irunmode]);
		if (length < 0)
		{
			// TODO: release resources!
		}
		length++;
		l->kernel_name = (char*)malloc(length);
		sprintf(l->kernel_name, kernel_name_fmt,
			kernel_basename, kernelgen_runmodes_names[irunmode]);
		
		// Allocate array to hold memory regions.
		// Maximum possible length is x2 times number of arguments
		// plus number of modules symbols, in case all arguments
		// and modules symbols are allocatable.
		l->regs = (struct kernelgen_memory_region_t*)malloc(
			sizeof(struct kernelgen_memory_region_t) * (nargs * 2 + nmodsyms));
		memset(l->regs, 0,
			sizeof(struct kernelgen_memory_region_t) * (nargs * 2 + nmodsyms));

		// Allocate array to hold kernel arguments.	
		l->args = (struct kernelgen_kernel_symbol_t*)malloc(
			sizeof(struct kernelgen_kernel_symbol_t) * nargs);
		memset(l->args, 0,
			sizeof(struct kernelgen_kernel_symbol_t) * nargs);

		// Allocate array to hold kernel used modules symbols.
		l->deps = (struct kernelgen_kernel_symbol_t*)malloc(
			sizeof(struct kernelgen_kernel_symbol_t) * nmodsyms);
		memset(l->deps, 0,
			sizeof(struct kernelgen_kernel_symbol_t) * nmodsyms);

		l->args_nregions = 0;
		l->deps_nregions = 0;

		// TODO: this is a temporary flag.
		l->deps_init = 0;

		// Build device-specific kernel source symbol name.
		length = snprintf(NULL, 0, kernel_name_fmt,
			l->kernel_name, "ir");
		if (length < 0)
		{
			// TODO: release resources!
		}
		length++;
		char* kernel_source_name = (char*)malloc(length);
		sprintf(kernel_source_name, kernel_name_fmt,
			l->kernel_name, "ir");

		// Build device-specific kernel binary symbol name.
		length = snprintf(NULL, 0, kernel_name_fmt,
			l->kernel_name, "binary");
		if (length < 0)
		{
			// TODO: release resources!
		}
		length++;
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
	}
	
	free(kernel_basename);
}

long kernelgen_kernel_init_deps_verbose = 1 << 2;

// Initialize kernel routine static dependencies.
void kernelgen_kernel_init_deps_(
	struct kernelgen_kernel_config_t* config, ...)
{
	// TODO: use this implementation instead of code in launcher,
	// when CUDA order of initialization issue will be fixed.
	
	return;
}

// Release resources used by kernel configuration.
void kernelgen_kernel_free(
	struct kernelgen_kernel_config_t* config)
{
	for (int irunmode = 1; irunmode < kernelgen_runmodes_count; irunmode++)
	{
		// Indirect indexing automatically has
		// unsupported runmodes skipped.
		int runmode = kernelgen_runmodes[irunmode];
		
		// Although, specific kernel may have supported
		// runmode disabled.
		if (!(config->runmode & runmode)) continue;

		struct kernelgen_launch_config_t* l = config->launch + irunmode;

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
void kernelgen_kernel_free_deps(
	struct kernelgen_kernel_config_t* config)
{
	// TODO: use this implementation instead of code in launcher,
	// when CUDA order of initialization issue will be fixed.
	for (int irunmode = 1; irunmode < kernelgen_runmodes_count; irunmode++)
	{
		// Indirect indexing automatically has
		// unsupported runmodes skipped.
		int runmode = kernelgen_runmodes[irunmode];
		
		// Although, specific kernel may have supported
		// runmode disabled.
		if (!(config->runmode & runmode)) continue;

		struct kernelgen_launch_config_t* l = config->launch + irunmode;
		
		for (int i = 0; i < config->nmodsyms; i++)
		{
			struct kernelgen_kernel_symbol_t* dep = l->deps + i;
			free(dep->name);		
		}
	}
}

// Read environment variables and setup runtime global configuration.
__attribute__ ((__constructor__(101))) void kernelgen_init()
{
	// Count supported runmodes
	// and sizes of their specific configs.
	kernelgen_runmodes_count = 2; // host + device CPU
#ifdef HAVE_CUDA
	specific_configs_size += sizeof(kernelgen_cuda_config_t);
	kernelgen_runmodes_count++;
#endif
#ifdef HAVE_OPENCL
	specific_configs_size += sizeof(kernelgen_opencl_config_t);
	kernelgen_runmodes_count++;
#endif

	// Allocate tables for device-specific functions pointers.
	kernelgen_parse_modsyms = 
		(kernelgen_parse_modsyms_func_t*)malloc(
			sizeof(kernelgen_parse_modsyms_func_t) *
			kernelgen_runmodes_count);
	kernelgen_load_regions =
		(kernelgen_load_regions_func_t*)malloc(
			sizeof(kernelgen_load_regions_func_t) *
			kernelgen_runmodes_count);
	kernelgen_save_regions =
		(kernelgen_save_regions_func_t*)malloc(
			sizeof(kernelgen_save_regions_func_t) *
			kernelgen_runmodes_count);
	kernelgen_build =
		(kernelgen_build_func_t*)malloc(
			sizeof(kernelgen_build_func_t) *
			kernelgen_runmodes_count);
	kernelgen_launch =
		(kernelgen_launch_func_t*)malloc(
			sizeof(kernelgen_launch_func_t) *
			kernelgen_runmodes_count);
	kernelgen_reset =
		(kernelgen_reset_func_t*)malloc(
			sizeof(kernelgen_reset_func_t) *
			kernelgen_runmodes_count);

#define BIND_RUNMODE(i, suffix) \
	{ \
		kernelgen_parse_modsyms[i] = kernelgen_parse_modsyms_##suffix; \
		kernelgen_load_regions[i] = kernelgen_load_regions_##suffix; \
		kernelgen_save_regions[i] = kernelgen_save_regions_##suffix; \
		kernelgen_build[i] = kernelgen_build_##suffix; \
		kernelgen_launch[i] = kernelgen_launch_##suffix; \
		kernelgen_reset[i] = kernelgen_reset_##suffix; \
		kernelgen_runmodes_names[i] = (char*)malloc(strlen(#suffix) + 1); \
		strcpy(kernelgen_runmodes_names[i], #suffix); \
	}

	// Create a list of values, names and mask of supported runmodes.
	// For each supported runmode bind its device-specific functions
	// to the common index list.
	kernelgen_runmodes_names =
		(char**)malloc(sizeof(char*) * kernelgen_runmodes_count);
	kernelgen_runmodes =
		(int*)malloc(sizeof(int) * kernelgen_runmodes_count);
	kernelgen_runmodes_mask = 0;
	{
		int i = 0;
		
		kernelgen_runmodes[i++] = KERNELGEN_RUNMODE_HOST;
		kernelgen_runmodes_mask |= KERNELGEN_RUNMODE_HOST;

		BIND_RUNMODE(i, cpu);
		kernelgen_runmodes[i++] = KERNELGEN_RUNMODE_DEVICE_CPU;
		kernelgen_runmodes_mask |= KERNELGEN_RUNMODE_DEVICE_CPU;

#ifdef HAVE_CUDA
		BIND_RUNMODE(i, cuda);
		kernelgen_runmodes[i++] = KERNELGEN_RUNMODE_DEVICE_CUDA;
		kernelgen_runmodes_mask |= KERNELGEN_RUNMODE_DEVICE_CUDA;
#endif

#ifdef HAVE_OPENCL
		BIND_RUNMODE(i, opencl);
		kernelgen_runmodes[i++] = KERNELGEN_RUNMODE_DEVICE_OPENCL;
		kernelgen_runmodes_mask |= KERNELGEN_RUNMODE_DEVICE_OPENCL;
#endif
	}

	// By default run everything on host.
	kernelgen_process_runmode = KERNELGEN_RUNMODE_HOST;
	char* crunmode = getenv("kernelgen_runmode");
	if (crunmode)
	{
		// Check the supplied runmode contains
		// only known bits, and at least one of them
		// is supported.
		int runmode = atoi(crunmode);
		if (runmode & kernelgen_runmodes_mask)
			kernelgen_process_runmode = runmode;
	}
	
	// Initially per-thread runmode setting is same to
	// per-process.
	kernelgen_thread_runmode = kernelgen_process_runmode;
	
	// By default disable all debug output.
	kernelgen_debug_output = 0;

	// By default enable all error output.
	kernelgen_error_output = ~kernelgen_debug_output;
	
	char* cdebug = getenv("kernelgen_debug_output");
	if (cdebug) kernelgen_debug_output = atoi(cdebug);
	
	char* cerror = getenv("kernelgen_error_output");
	if (cerror) kernelgen_error_output = atoi(cerror);

	// Create array of pointers to device-specific configs.
	// Initialize device-specific settings.
	size_t offset = 
		sizeof(kernelgen_specific_config_t) * kernelgen_runmodes_count;
	specific_configs_size += offset;
	specific_configs =
		(kernelgen_specific_config_t*)malloc(specific_configs_size);
	specific_configs[0] = NULL;
	specific_configs[1] = NULL;
	{
		int i = 2;
		kernelgen_status_t result;
#ifdef HAVE_CUDA
		struct kernelgen_cuda_config_t* cuda =
			(struct kernelgen_cuda_config_t*)(
			(kernelgen_specific_config_t)(specific_configs) + offset);
		specific_configs[i++] = (kernelgen_specific_config_t)offset;
		offset += sizeof(struct kernelgen_cuda_config_t);

#ifdef HAVE_ALIGNED_MAPPING
		cuda->aligned = 1;
#else
		cuda->aligned = 0;
#endif

#ifndef OPENCL
		// Create single NVIDIA platform and count available devices.
		kernelgen_platforms_count = 1;
		const char* nvidia = "NVIDIA";
		kernelgen_platforms_names = (char**)malloc(strlen(nvidia) + 1 + sizeof(char*));
		strcpy(kernelgen_platforms_names + 1, nvidia);
		kernelgen_platforms_names[0] = (char*)(kernelgen_platforms_names + 1);
		kernelgen_devices_count = (int*)malloc(sizeof(int));
		kernelgen_devices_count[0] = 0;
		cudaGetLastError();
		result.value = cudaGetDeviceCount(kernelgen_devices_count);
		if (result.value != cudaSuccess)
		{
			kernelgen_print_error(kernelgen_launch_verbose,
				"Cannot get CUDA device count, status = %d: %s\n",
				result.value, kernelgen_get_error_string(result));
			kernelgen_set_last_error(result);
			return;
		}
		
		// TODO: disable runmode if no devices found.
#endif
#endif
#ifdef HAVE_OPENCL	
		struct kernelgen_opencl_config_t* opencl =
			(struct kernelgen_opencl_config_t*)(
			(kernelgen_specific_config_t)(specific_configs) + offset);
		specific_configs[i++] = (kernelgen_specific_config_t)offset;
		offset += sizeof(struct kernelgen_opencl_config_t);

		// Being quiet optimistic initially...
		result.value = CL_SUCCESS;
		result.runmode = KERNELGEN_RUNMODE_DEVICE_OPENCL;
		
		// Get OpenCL platforms count.
		cl_uint platforms_count = 0;
		result.value = clGetPlatformIDs(0, NULL, &platforms_count);
		if (result.value != CL_SUCCESS)
		{
			kernelgen_print_error(kernelgen_launch_verbose,
				"clGetPlatformIDs returned %d: %s\n", (int)result.value,
				kernelgen_get_error_string(result));
			kernelgen_set_last_error(result);
			// TODO: release resources!
			return;
		}
		kernelgen_platforms_count = platforms_count;
		
		// Get OpenCL platforms IDs.
		kernelgen_platforms = (cl_platform_id*)malloc(
			sizeof(cl_platform_id) * kernelgen_platforms_count);
		result.value = clGetPlatformIDs(kernelgen_platforms_count,
			kernelgen_platforms, NULL);
		if (result.value != CL_SUCCESS)
		{
			kernelgen_print_error(kernelgen_launch_verbose,
				"clGetPlatformIDs returned %d: %s\n", (int)result.value,
				kernelgen_get_error_string(result));
			kernelgen_set_last_error(result);
			// TODO: release resources!
			return;
		}
		
		// Get OpenCL platforms names.
		size_t total_names_length = 0;
		for (int i = 0; i < kernelgen_platforms_count; i++)
		{
			size_t length;
			clGetPlatformInfo(kernelgen_platforms[i],
				CL_PLATFORM_VENDOR, 0, NULL, &length);
			total_names_length += length;
		}
		kernelgen_platforms_names = (char**)malloc(
			sizeof(char*) * kernelgen_platforms_count +
			total_names_length);
		for (int i = 0, offset = 0; i < kernelgen_platforms_count; i++)
		{
			size_t length;
			clGetPlatformInfo(kernelgen_platforms[i],
				CL_PLATFORM_VENDOR, 0, NULL, &length);
			kernelgen_platforms_names[i] = offset +
				(char*)(kernelgen_platforms_names + kernelgen_platforms_count);
			clGetPlatformInfo(kernelgen_platforms[i],
				CL_PLATFORM_VENDOR, length,
				kernelgen_platforms_names[i], NULL);
			offset += length;
		}

		// Get OpenCL devices.
		int total_devices_count = 0;
		kernelgen_devices_count = (int*)malloc(
			sizeof(int) * kernelgen_platforms_count);
		for (int i = 0; i < kernelgen_platforms_count; i++)
		{
			cl_uint devices_count = 0;
			result.value = clGetDeviceIDs(kernelgen_platforms[i],
				CL_DEVICE_TYPE_ALL, 0, NULL, &devices_count);
			if (result.value != CL_SUCCESS)
			{
				kernelgen_print_error(kernelgen_launch_verbose,
					"clGetDeviceIDs returned %d: %s\n", (int)result.value,
					kernelgen_get_error_string(result));
				kernelgen_set_last_error(result);
				// TODO: release resources!
				return;
			}
			kernelgen_devices_count[i] = devices_count;
			total_devices_count += kernelgen_devices_count[i];
		}
		kernelgen_devices = (cl_device_id**)malloc(
			sizeof(cl_device_id*) * kernelgen_platforms_count +
			sizeof(cl_device_id) * total_devices_count);
		for (int i = 0, offset = 0; i < kernelgen_platforms_count; i++)
		{
			kernelgen_devices[i] = offset +
				(cl_device_id*)(kernelgen_devices + kernelgen_platforms_count);
			result.value = clGetDeviceIDs(
				kernelgen_platforms[i], CL_DEVICE_TYPE_ALL,
				kernelgen_devices_count[i], kernelgen_devices[i], NULL);
			if (result.value != CL_SUCCESS)
			{
				kernelgen_print_error(kernelgen_launch_verbose,
					"clGetDeviceIDs returned %d: %s\n", (int)result.value,
					kernelgen_get_error_string(result));
				kernelgen_set_last_error(result);
				// TODO: release resources!
				return;
			}
			offset += kernelgen_devices_count[i];
		}

		// Get OpenCL devices contexts and command queues.
		kernelgen_contexts = (cl_context**)malloc(
			sizeof(cl_context*) * kernelgen_platforms_count +
			sizeof(cl_context) * total_devices_count);
		kernelgen_queues = (cl_command_queue**)malloc(
			sizeof(cl_command_queue*) * kernelgen_platforms_count +
			sizeof(cl_command_queue) * total_devices_count);
		size_t szproperties =
			2 * sizeof(cl_context_properties) + sizeof(cl_platform_id);
		cl_context_properties* properties = (cl_context_properties*)malloc(szproperties);
		for (int i = 0, offset = 0; i < kernelgen_platforms_count; i++)
		{
			memset(properties, 0, szproperties);
			*properties = CL_CONTEXT_PLATFORM;
			*(cl_platform_id*)(properties + 1) = kernelgen_platforms[i];
			kernelgen_contexts[i] = offset +
				(cl_context*)(kernelgen_contexts + kernelgen_platforms_count);
			kernelgen_queues[i] = offset +
				(cl_command_queue*)(kernelgen_queues + kernelgen_platforms_count);
			for (int j = 0; j < kernelgen_devices_count[i]; j++)
			{
				kernelgen_contexts[i][j] = clCreateContext(
					properties, 1, kernelgen_devices[i] + j,
					NULL, NULL, &result.value);
				if (result.value != CL_SUCCESS)
				{
					kernelgen_print_error(kernelgen_launch_verbose,
						"clCreateContext returned %d: %s\n", (int)result.value,
						kernelgen_get_error_string(result));
					kernelgen_set_last_error(result);
					// TODO: release resources!
					return;
				}

				kernelgen_queues[i][j] = clCreateCommandQueue(
					kernelgen_contexts[i][j], kernelgen_devices[i][j],
					0, &result.value);
				if (result.value != CL_SUCCESS)
				{
					kernelgen_print_error(kernelgen_launch_verbose,
						"clCreateCommandQueue returned %d: %s\n", (int)result.value,
						kernelgen_get_error_string(result));
					kernelgen_set_last_error(result);
					// TODO: release resources!
					return;
				}
			}
			offset += kernelgen_devices_count[i];
		}
		free(properties);

		// TODO: disable runmode if no devices found.
#endif
	}
}

// Release resources used by runtime global configuration.
__attribute__ ((__destructor__(101))) void kernelgen_free()
{
	free(kernelgen_parse_modsyms);
	free(kernelgen_load_regions);
	free(kernelgen_save_regions);
	free(kernelgen_build);
	free(kernelgen_launch);
	free(kernelgen_reset);
	free(kernelgen_runmodes);
	for (int i = 1; i < kernelgen_runmodes_count; i++)
		free(kernelgen_runmodes_names[i]);
	free(kernelgen_runmodes_names);

	// TODO: cleanup OpenCL/CUDA devices info.
}

