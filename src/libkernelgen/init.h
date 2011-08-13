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

#ifndef HAVE_INIT_H
#define HAVE_INIT_H

#include "kernelgen_int.h"

#ifdef __cplusplus
extern "C"
{
#endif

// Parse module symbols for kernel executed on CPU.
kernelgen_status_t kernelgen_parse_modsyms_cpu(
	struct kernelgen_launch_config_t* launch,
	int* nargs, va_list list);

// Parse module symbols for kernel executed on CUDA GPU.
kernelgen_status_t kernelgen_parse_modsyms_cuda(
	struct kernelgen_launch_config_t* launch,
	int* nargs, va_list list);

// Parse module symbols for kernel executed on OpenCL device.
kernelgen_status_t kernelgen_parse_modsyms_opencl(
	struct kernelgen_launch_config_t* launch,
	int* nargs, va_list list);

// Load regions into CPU device memory space.
kernelgen_status_t kernelgen_load_regions_cpu(
	struct kernelgen_launch_config_t* launch, int* nmapped);

// Save regions from CPU device memory space.
kernelgen_status_t kernelgen_save_regions_cpu(
	struct kernelgen_launch_config_t* launch, int nmapped);

// Load regions into CUDA device memory space.
kernelgen_status_t kernelgen_load_regions_cuda(
	struct kernelgen_launch_config_t* launch, int* nmapped);

// Save regions from CUDA device memory space.
kernelgen_status_t kernelgen_save_regions_cuda(
	struct kernelgen_launch_config_t* launch, int nmapped);

// Load regions into OpenCL device memory space.
kernelgen_status_t kernelgen_load_regions_opencl(
	struct kernelgen_launch_config_t* launch, int* nmapped);

// Save regions from OpenCL device memory space.
kernelgen_status_t kernelgen_save_regions_opencl(
	struct kernelgen_launch_config_t* launch, int nmapped);

// Build kernel for the CPU device.
kernelgen_status_t kernelgen_build_cpu(
	struct kernelgen_launch_config_t* launch);

// Build kernel for the CUDA device.
kernelgen_status_t kernelgen_build_cuda(
	struct kernelgen_launch_config_t* launch);

// Build kernel for the OpenCL device.
kernelgen_status_t kernelgen_build_opencl(
	struct kernelgen_launch_config_t* launch);

// Launch kernel on the CPU device.
kernelgen_status_t kernelgen_launch_cpu(
	struct kernelgen_launch_config_t* launch,
	int* bx, int* ex, int* by, int* ey, int* bz, int* ez);

// Launch kernel on the CUDA device.
kernelgen_status_t kernelgen_launch_cuda(
	struct kernelgen_launch_config_t* launch,
	int* bx, int* ex, int* by, int* ey, int* bz, int* ez);

// Launch kernel on the OpenCL device.
kernelgen_status_t kernelgen_launch_opencl(
	struct kernelgen_launch_config_t* launch,
	int* bx, int* ex, int* by, int* ey, int* bz, int* ez);

// Reset CPU device.
kernelgen_status_t kernelgen_reset_cpu(
	struct kernelgen_launch_config_t* launch);

// Reset CUDA device.
kernelgen_status_t kernelgen_reset_cuda(
	struct kernelgen_launch_config_t* launch);

// Reset OpenCL device.
kernelgen_status_t kernelgen_reset_opencl(
	struct kernelgen_launch_config_t* launch);

#ifdef __cplusplus
}
#endif

#endif // HAVE_INIT_H

