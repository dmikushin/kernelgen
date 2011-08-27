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

#ifndef KERNELGEN_H
#define KERNELGEN_H

#define KERNELGEN_RUNMODE_HOST		 1
#define KERNELGEN_RUNMODE_DEVICE_CPU	(1 << 3)
#define KERNELGEN_RUNMODE_DEVICE_CUDA	(1 << 1)
#define KERNELGEN_RUNMODE_DEVICE_OPENCL	(1 << 2)

// Memory page size.
#define SZPAGE 4096

// Maximum allocatable descriptor size
#define SZDESC 128

// Enable memory mapping to shared data between host and GPU.
// If undefined, plain cudaMalloc/cudaMemcpy is used.
#ifdef HAVE_CUDA
//#define HAVE_MAPPING
#endif

// Enable additional aligning before memory mapping
// (unaligned mapping is only supported by yet unreleased CUDA driver).
#ifdef HAVE_CUDA
//#define HAVE_ALIGNED_MAPPING
#endif

#define kernelgen_print_debug(mask, ...) \
{ \
	if (kernelgen_debug_output & mask) \
	{ \
		fprintf(stdout, "%s:%d kernelgen message (debug) ", __FILE__, __LINE__); \
		fprintf(stdout, __VA_ARGS__); \
	} \
}
#define kernelgen_print_error(mask, ...) \
{ \
	if (kernelgen_error_output & mask) \
	{ \
		fprintf(stderr, "%s:%d kernelgen message (error) ", __FILE__, __LINE__); \
		fprintf(stderr, __VA_ARGS__); \
	} \
}

#ifdef __cplusplus
extern "C"
{
#endif

// Defines debug output options bit field.
extern long kernelgen_debug_output;

// Defines error output options bit field.
extern long kernelgen_error_output;

// Defines kernelgen error status structure.
typedef struct
{
	int value;
	int runmode;
}
kernelgen_status_t;

// Defines kernelgen-specific error codes.
enum kernelgen_error
{
	kernelgen_success = 0,
	kernelgen_initialization_failed = 1021,
	kernelgen_error_not_found = 1022,
	kernelgen_error_not_implemented = 1023,
	kernelgen_error_ffi_setup = 1024,
	kernelgen_error_results_mismatch = 1025,
};

#define kernelgen_make_error(var, value_, runmode_) \
	{ (var).value = value_; (var).runmode = runmode_; }

typedef char* kernelgen_specific_config_t;

// Defines kernel configuration structure.
struct kernelgen_kernel_config_t
{
	// The kernel runmode. Either inherits initial per-process
	// runmode or its individual environment setting.
	// Note kernel runmode setting is unchanged during program
	// execution, but actual kernel behavior is affected by the
	// combination of per-kernel runmode setting and per-thread
	// runmode executing entire kernel.
	int runmode;
	
	int compare;

	int iloop, nloops;
	char* routine_name;

	// Kernel configuration instance shared between
	// launching and comparison functions.
	struct kernelgen_launch_config_t* launch;

	// The total number of arguments and dependencies.
	int nargs, nmodsyms;
	
	// The indexes of platform and device kernel was
	// last time executed on.
	int last_platform_index;
	int last_device_index;
	
	// Pointers to device-specific configs.
	kernelgen_specific_config_t* specific;
};

// Initialize kernel routine configuration.
void kernelgen_kernel_init(
	struct kernelgen_kernel_config_t* config,
	int iloop, int nloops, char* name,
	int nargs, int nmodsyms);

// Assign device to the current host thread.
kernelgen_status_t kernelgen_set_device(
	int platform_index, int device_index);

// Get runmode of the entire thread.
int kernelgen_get_runmode();

// Initialize kernel routine static dependencies.
void kernelgen_kernel_init_deps_(
	struct kernelgen_kernel_config_t* config, ...);
extern long kernelgen_kernel_init_deps;

// Release resources used by kernel routine configuration.
void kernelgen_kernel_free(
	struct kernelgen_kernel_config_t* config);

// Release resources used by kernel routine static dependencies.
void kernelgen_kernel_free_deps(
	struct kernelgen_kernel_config_t* config);

// Launch kernel with the specified 3D compute grid, arguments and modules symbols.
void kernelgen_launch_(
	struct kernelgen_kernel_config_t* config,
	int* bx, int* ex, int* by, int* ey, int* bz, int* ez,
	int* nargs, int* nmodsyms, ...);
extern long kernelgen_launch_verbose;

// The type of function performing kernel-specific results comparison.
typedef int (*kernelgen_compare_function_t)(double* maxdiff, ...);

// Compare currently cached kernel results with results of the 
// regular host loop.
void kernelgen_compare_(
	struct kernelgen_kernel_config_t* config,
	kernelgen_compare_function_t compare, double* maxdiff);
extern long kernelgen_compare_verbose;

// Get last kernel loop launching error.
kernelgen_status_t kernelgen_get_last_error();

// Get text message for the specified error code.
const char* kernelgen_get_error_string(kernelgen_status_t error);

#include <stdint.h>

#pragma pack(push, 1)

// The built-in timer value type.
typedef struct
{
	int64_t seconds;
	int64_t nanoseconds;
}
kernelgen_time_t;

#pragma pack(pop)

void kernelgen_get_timer_resolution(kernelgen_time_t* val);

// Get the built-in timer value.
void kernelgen_get_time(kernelgen_time_t* val);

// Get the built-in timer measured values difference.
double kernelgen_get_time_diff(
	kernelgen_time_t* val1, kernelgen_time_t* val2);

// Print the built-in timer measured values difference.
void kernelgen_print_time_diff(
	kernelgen_time_t* val1, kernelgen_time_t* val2);

#ifdef __cplusplus
}
#endif

#endif // KERNELGEN_H

