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

#ifndef GFORSCALE_H
#define GFORSCALE_H

#define GFORSCALE_RUNMODE_HOST		 1
#define GFORSCALE_RUNMODE_DEVICE_CPU	(1 << 3)
#define GFORSCALE_RUNMODE_DEVICE_CUDA	(1 << 1)
#define GFORSCALE_RUNMODE_DEVICE_OPENCL	(1 << 2)

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

#define gforscale_print_debug(mask, ...) \
{ \
	if (gforscale_debug_output & mask) \
	{ \
		fprintf(stdout, "%s:%d ", __FILE__, __LINE__); \
		fprintf(stdout, __VA_ARGS__); \
	} \
}
#define gforscale_print_error(mask, ...) \
{ \
	if (gforscale_error_output & mask) \
	{ \
		fprintf(stderr, "%s:%d ", __FILE__, __LINE__); \
		fprintf(stderr, __VA_ARGS__); \
	} \
}

#ifdef __cplusplus
extern "C"
{
#endif

// Defines default kernel execution mode,
// that is used, if per-kernel execution mode
// is for specific kernel is undefined.
extern int gforscale_runmode;

// Defines debug output options bit field.
extern long gforscale_debug_output;

// Defines error output options bit field.
extern long gforscale_error_output;

// Defines gforscale error status structure.
typedef struct
{
	int value;
	int runmode;
}
gforscale_status_t;

// Defines gforscale-specific error codes.
enum gforscale_error
{
	gforscale_success = 0,
	gforscale_initialization_failed = 1021,
	gforscale_error_not_found = 1022,
	gforscale_error_not_implemented = 1023,
	gforscale_error_ffi_setup = 1024,
	gforscale_error_results_mismatch = 1025
};

#define gforscale_make_error(var, value_, runmode_) \
	{ (var).value = value_; (var).runmode = runmode_; }

typedef char* gforscale_specific_config_t;

// Defines kernel configuration structure.
struct gforscale_kernel_config_t
{
	int runmode, compare;

	int iloop, nloops;
	char* routine_name;

	// Kernel configuration instance shared between
	// launching and comparison functions.
	struct gforscale_launch_config_t* launch;

	// The total number of arguments and dependencies.
	int nargs, nmodsyms;
	
	// Pointers to device-specific configs.
	gforscale_specific_config_t* specific;
};

// Initialize kernel routine configuration.
void gforscale_kernel_init(
	struct gforscale_kernel_config_t* config,
	int iloop, int nloops, char* name,
	int nargs, int nmodsyms);

// Initialize kernel routine static dependencies.
void gforscale_kernel_init_deps_(
	struct gforscale_kernel_config_t* config, ...);
extern long gforscale_kernel_init_deps;

// Release resources used by kernel routine configuration.
void gforscale_kernel_free(
	struct gforscale_kernel_config_t* config);

// Release resources used by kernel routine static dependencies.
void gforscale_kernel_free_deps(
	struct gforscale_kernel_config_t* config);

// Launch kernel with the specified 3D compute grid, arguments and modules symbols.
void gforscale_launch_(
	struct gforscale_kernel_config_t* config,
	int* bx, int* ex, int* by, int* ey, int* bz, int* ez,
	int* nargs, int* nmodsyms, ...);
extern long gforscale_launch_verbose;

// The type of function performing kernel-specific results comparison.
typedef int (*gforscale_compare_function_t)(double* maxdiff, ...);

// Compare currently cached kernel results with results of the 
// regular host loop.
void gforscale_compare_(
	struct gforscale_kernel_config_t* config,
	gforscale_compare_function_t compare, double* maxdiff);
extern long gforscale_compare_verbose;

// Get last kernel loop launching error.
gforscale_status_t gforscale_get_last_error();

// Get text message for the specified error code.
const char* gforscale_get_error_string(gforscale_status_t error);

#ifdef __cplusplus
}
#endif

#endif // GFORSCALE_H

