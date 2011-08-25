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

#ifndef KERNELGEN_INT_H
#define KERNELGEN_INT_H

#include "kernelgen.h"

#include <gelf.h>
#include <stdio.h>

#ifdef HAVE_OPENCL
#include <CL/cl.h>
#endif

#ifdef __cplusplus
extern "C"
{
#endif

// Defines per-process execution mode setting.
extern int kernelgen_process_runmode;

// Each host thread has individual runmode setting,
// that is a combination of process runmode setting
// and capabilities of the currently selected device.
extern __thread int kernelgen_thread_runmode;

// Each host thread keeps indexes of currently used platform and device.
extern __thread int kernelgen_thread_platform_index;
extern __thread int kernelgen_thread_device_index;

// Set last kernel loop launching error.
void kernelgen_set_last_error(kernelgen_status_t error);

// Defines all supported runmodes values array.
extern int* kernelgen_runmodes;

// Defines all supported runmodes names array.
extern char** kernelgen_runmodes_names;

// Defines supported runmodes count.
extern int kernelgen_runmodes_count;

// Defines supported runmodes collective bitmask.
extern int kernelgen_runmodes_mask;

// Platforms handles and count.
#ifdef HAVE_OPENCL
extern cl_platform_id* kernelgen_platforms;
#endif
extern int kernelgen_platforms_count;
extern char** kernelgen_platforms_names;

// Devices handles and count.
#ifdef HAVE_OPENCL
extern cl_device_id** kernelgen_devices;
#endif
extern int* kernelgen_devices_count;

// Devices contexts and command queues.
#ifdef HAVE_OPENCL
extern cl_context** kernelgen_contexts;
extern cl_command_queue** kernelgen_queues;
#endif

// Defines memory region properties.
struct kernelgen_memory_region_t
{
	// Reference to the parent kernel dependency.
	struct kernelgen_kernel_symbol_t* symbol;
	
	// Shift from base address.
	unsigned int shift;
	
	// Memory region base address.
	void* base;
	
	// Memory region size.
	size_t size;
	
	// Link to primary memory region - another
	// instance whose mapping the current region reuses
	// (NULL if current is primary iself).
	struct kernelgen_memory_region_t* primary;
	
	// Link to store device-mapped address.
	void* mapping;
};

// Defines kernel dependency symbol.
struct kernelgen_kernel_symbol_t
{
	// Symbol name.
	char* name;

	// Argument index.
	int index;
	
	// The (ref, size, desc) triple defines the kernel
	// argument specification passed from user code side.
	// Additional s* pair - shadowed original pointers
	// (for comparison mode), and dev_* pair - even more
	// space for backups.
	void *ref, *sref, *dev_ref;
	void *desc, *sdesc, *dev_desc;
	size_t size, desc_size;
	
	// Flag indicating if the symbol is allocatable.
	int allocatable;

	// References to memory regions of
	// data vector and descriptor container
	// (for allocatable variables).
	struct kernelgen_memory_region_t *mref, *mdesc;
};

#pragma pack(push, 1)

// Defines kernel launching configuration.
struct kernelgen_launch_config_t
{
	// Parent kernel config.
	struct kernelgen_kernel_config_t* config;
	
	char* kernel_name;

	char* kernel_source; size_t kernel_source_size;
	char* kernel_binary; size_t kernel_binary_size;
	
	// Entire launch config runmode.
	int runmode;

	// Memory regions array.
	struct kernelgen_memory_region_t* regs;

	// Kernel arguments array.
	struct kernelgen_kernel_symbol_t* args;

	// Kernel used modules symbols (dependencies) array.
	struct kernelgen_kernel_symbol_t* deps;
	
	// The number of regions of specific kind.
	int args_nregions, deps_nregions;
	
	// TODO: this is a temporary flag.
	int deps_init;
	
	// Kernel working time in seconds
	// (without accounting time of data transfers).
	double time;
	
	// Pointer to collection of device-specific
	// kernel launch properties.
	kernelgen_specific_config_t specific;
};

#pragma pack(pop)

#include <stdarg.h>

// Merge specified memory regions into non-overlapping regions.
kernelgen_status_t kernelgen_merge_regions(
	struct kernelgen_memory_region_t* regs,
	int count);

// Parse kernel arguments into launch config structure.
kernelgen_status_t kernelgen_parse_args(
	struct kernelgen_launch_config_t* launch,
	int* nargs, va_list list);

// Parse kernel arguments into launch config structure.
// (with memory aligning).
kernelgen_status_t kernelgen_parse_args_aligned(
	struct kernelgen_launch_config_t* launch,
	int* nargs, va_list list);

// Parse module symbols for kernel executed on specific device.
typedef kernelgen_status_t (*kernelgen_parse_modsyms_func_t)(
	struct kernelgen_launch_config_t* launch,
	int* nargs, va_list list);

extern kernelgen_parse_modsyms_func_t* kernelgen_parse_modsyms;

// Load regions into specific device memory space.
typedef kernelgen_status_t (*kernelgen_load_regions_func_t)(
	struct kernelgen_launch_config_t* launch, int* nmapped);

extern kernelgen_load_regions_func_t* kernelgen_load_regions;

// Save regions from specific device memory space.
typedef kernelgen_status_t (*kernelgen_save_regions_func_t)(
	struct kernelgen_launch_config_t* launch, int nmapped);

extern kernelgen_save_regions_func_t* kernelgen_save_regions;

// Build kernel for specific device.
typedef kernelgen_status_t (*kernelgen_build_func_t)(
	struct kernelgen_launch_config_t* launch);

extern kernelgen_build_func_t* kernelgen_build;

// Launch kernel on specific device.
typedef kernelgen_status_t (*kernelgen_launch_func_t)(
	struct kernelgen_launch_config_t* launch,
	int* bx, int* ex, int* by, int* ey, int* bz, int* ez);

extern kernelgen_launch_func_t* kernelgen_launch;

// Reset specific device.
typedef kernelgen_status_t (*kernelgen_reset_func_t)(
	struct kernelgen_launch_config_t* launch);

extern kernelgen_reset_func_t* kernelgen_reset;

// Load contents of the specified text file.
int kernelgen_load_source(
	const char* filename, char** source, size_t* szsource);

// Load the specified ELF image symbol raw data.
int elf_read(const char* filename, const char* symname,
	char** symdata, size_t* symsize);

// Load the specified ELF executable header.
int kernelgen_elf_read_eheader(
	const char* executable, GElf_Ehdr* ehdr);

// Create ELF image containing symbol with the specified name,
// associated data content and its length. Certain ELF properties
// could be taken from the specified reference executable, if not NULL.
int gforsclae_elf_write(const char* filename, GElf_Ehdr* ehdr,
	const char* symname, const char* symdata, size_t length);

// Create ELF image containing multiple symbols with the specified names,
// associated data contents and their lengths. Certain ELF properties
// could be taken from the specified reference executable, if not NULL.
int kernelgen_elf_write_many(const char* filename, GElf_Ehdr* ehdr,
	int count, ...);

#ifdef __cplusplus
}
#endif

#endif // KERNELGEN_INT_H

