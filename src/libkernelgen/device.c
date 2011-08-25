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

#ifdef HAVE_OPENCL
#include <CL/cl.h>
#endif

#ifdef HAVE_CUDA
#include <cuda_runtime.h>
#endif

#include <string.h>

// Get runmode of the entire thread.
int kernelgen_get_runmode()
{
	return kernelgen_thread_runmode;
}

// Assign device to the current host thread.
kernelgen_status_t kernelgen_set_device(int platform_index, int device_index)
{
	// Being quiet optimistic initially...
	kernelgen_status_t result;
	result.value = kernelgen_success;
	result.runmode = KERNELGEN_RUNMODE_HOST;

	// If platform and index unchanged, return now.
	if ((platform_index == kernelgen_thread_platform_index) &&
		(device_index == kernelgen_thread_device_index))
		return result;

	// Platforms info is set during kernelgen global initialization.
	// Check if requested platform index is within bounds.
	if ((platform_index < 0) || (platform_index >= kernelgen_platforms_count))
	{
		result.value = kernelgen_error_not_found;
		kernelgen_print_error(kernelgen_launch_verbose,
			"Invalid platform index: %d (must be 0 <= x < %d)\n",
				platform_index, kernelgen_platforms_count);
		return result;
	}
	
	// Devices info is set during kernelgen global initialization.
	// Check if requested device index is within bounds.
	if ((device_index < 0) || (device_index >= kernelgen_devices_count[platform_index]))
	{
		result.value = kernelgen_error_not_found;
		kernelgen_print_error(kernelgen_launch_verbose,
			"Invalid device index: %d (must be 0 <= x < %d)\n",
				device_index, kernelgen_devices_count[platform_index]);
		return result;
	}
	
	// OK, values seem to be nice.
	kernelgen_thread_platform_index = platform_index;
	kernelgen_thread_device_index = device_index;

	if (!strcmp(kernelgen_platforms_names[platform_index], "NVIDIA"))
	{
#ifdef HAVE_CUDA
		// If platform is NVIDIA, set device, enable memory mapping
		// and enable CUDA runmode.
		cudaSetDevice(device);
	
		cudaGetLastError();
		result.value = cudaSetDeviceFlags(cudaDeviceMapHost);
		if (result.value != cudaSuccess)
		{
			kernelgen_print_error(kernelgen_launch_verbose,
				"Cannot set device flags, status = %d: %s\n",
				result.value, kernelgen_get_error_string(result));
			kernelgen_set_last_error(result);
			return result;
		}

		// Enable CUDA thread runmode, if it is enabled by
		// process runmode.
		kernelgen_thread_runmode = kernelgen_process_runmode;
		kernelgen_thread_runmode |= kernelgen_process_runmode & KERNELGEN_RUNMODE_DEVICE_CUDA;
#else
		// Disable CUDA runmode.
		kernelgen_thread_runmode &= ~KERNELGEN_RUNMODE_DEVICE_CUDA;
#endif
#ifdef HAVE_OPENCL
		// Enable OpenCL thread runmode, if it is enabled by
		// process runmode.
		kernelgen_thread_runmode = kernelgen_process_runmode;
		kernelgen_thread_runmode |= kernelgen_process_runmode & KERNELGEN_RUNMODE_DEVICE_OPENCL;
#else
		// Disable OpenCL runmode.
		kernelgen_thread_runmode &= ~KERNELGEN_RUNMODE_DEVICE_OPENCL;
#endif
	}
	else
	{
#ifdef HAVE_OPENCL
		// Enable OpenCL thread runmode, if it is enabled by
		// process runmode.
		kernelgen_thread_runmode = kernelgen_process_runmode;
		kernelgen_thread_runmode |= kernelgen_process_runmode & KERNELGEN_RUNMODE_DEVICE_OPENCL;
#else
		// Disable OpenCL runmode.
		kernelgen_thread_runmode &= ~KERNELGEN_RUNMODE_DEVICE_OPENCL;
#endif
	}

	return result;
}

