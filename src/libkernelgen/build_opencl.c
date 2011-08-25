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
#include "kernelgen_int_opencl.h"

#include <string.h>

kernelgen_status_t kernelgen_build_opencl(
	struct kernelgen_launch_config_t* l)
{
#ifdef HAVE_OPENCL
	struct kernelgen_opencl_config_t* opencl =
		(struct kernelgen_opencl_config_t*)l->specific;

	// Being quiet optimistic initially...
	kernelgen_status_t result;
	result.value = CL_SUCCESS;
	result.runmode = l->runmode;
	
	int iplatform = kernelgen_thread_platform_index;
	int idevice = kernelgen_thread_device_index;
	
	// If kernel was previously compiled for entire
	// platform/device, return now.
	if ((l->config->last_platform_index == iplatform) &&
		(l->config->last_device_index == idevice))
		return result;
	
	// Otherwise, optimize kernel IR and rebuild it.
	// TODO: optimize IR.
	char* target_source;
	size_t target_source_size = 0;
	if (!strcmp(kernelgen_platforms_names[iplatform], "Advanced Micro Devices, Inc."))
	{
		// Double extension is named differently in case of AMD OpenCL.
		kernelgen_build_llvm(l, "-m32 -D__OPENCL_DEVICE_FUNC__ -Dcl_khr_fp64=cl_amd_fp64",
			&target_source, &target_source_size);
	}
	else
	{
		kernelgen_build_llvm(l, "-m32 -D__OPENCL_DEVICE_FUNC__",
			&target_source, &target_source_size);
	}

	printf("%s\n", target_source);

	/*#
	# Add structure emulating global variables to kernel
	# function prototype (if structure exists).
	#
	if ($code =~ m/\$kernel_name\$_globals_t/)
	{
		$code =~ m/\#ifdef\s__CUDA_DEVICE_FUNC__\n__device__\n#endif\nvoid\s$name\_\((?<PROTO>[^\;]*)\)\;/s;
		my($old_proto) = $+{PROTO};
		my($new_proto) = $old_proto;
		if ($new_proto ne "")
		{
			$new_proto .= ", ";
		}
		$new_proto .= "__global struct $name\_opencl_globals_t* $name\_opencl_globals";
		$old_proto = quotemeta($old_proto);
		$code =~ s/void\s$name\_\($old_proto/void $name\_($new_proto/gs;
	}

	$code =~ s/\#ifdef\s__CUDA_DEVICE_FUNC__\n__device__\n#endif\nvoid\s$name\_\(/#ifdef __OPENCL_DEVICE_FUNC__\n__kernel\n#endif\n#define $name\_ $name\_opencl\nvoid $name\_(/s;
	
	$code =~ s/void\s$name\_blockidx_x\(\n#ifdef\s__OPENCL_DEVICE_FUNC__\n__global\n#endif\s\/\/\s__OPENCL_DEVICE_FUNC__\nunsigned\sint\s\*,\sunsigned\sint\s,\sunsigned int\s\)\;/void $name\_blockidx_x(unsigned int* index, unsigned int start, unsigned int end) { *index = get_group_id(0) + start; }/s;

	$code =~ s/void\s$name\_blockidx_y\(\n#ifdef\s__OPENCL_DEVICE_FUNC__\n__global\n#endif\s\/\/\s__OPENCL_DEVICE_FUNC__\nunsigned\sint\s\*,\sunsigned\sint\s,\sunsigned int\s\)\;/void $name\_blockidx_y(unsigned int* index, unsigned int start, unsigned int end) { *index = get_group_id(1) + start; }/s;

	$code =~ s/void\s$name\_blockidx_z\(\n#ifdef\s__OPENCL_DEVICE_FUNC__\n__global\n#endif\s\/\/\s__OPENCL_DEVICE_FUNC__\nunsigned\sint\s\*,\sunsigned\sint\s,\sunsigned int\s\)\;/void $name\_blockidx_z(unsigned int* index, unsigned int start, unsigned int end) { *index = get_local_id(2) + start; }/s;
	
	#
	# Replace $kernel_name$ with actual name.
	#
	$code =~ s/\$kernel_name\$/$name\_opencl/g;*/
	
	// Grab current device and context and rebuild the kernel.
	cl_device_id device = kernelgen_devices[iplatform][idevice];
	cl_context context = kernelgen_contexts[iplatform][idevice];

	// Load OpenCL program from source.
	cl_int status = CL_SUCCESS;
	/*opencl->program = clCreateProgramWithBinary(
		context, 1, &device, &l->kernel_binary_size,
		(const unsigned char **)&l->kernel_binary, &status, &result.value);*/
	opencl->program = clCreateProgramWithSource(
		context, 1, (const char**)&target_source,
		&target_source_size, &result.value);
	if ((result.value != CL_SUCCESS) || (status != CL_SUCCESS))
	{
		if (result.value != CL_SUCCESS)
		{
			kernelgen_print_error(kernelgen_launch_verbose,
				"Cannot create kernel %s, status = %d: %s\n",
				l->kernel_name, result.value, kernelgen_get_error_string(result));
		}
		if (status != CL_SUCCESS)
		{
			result.value = status;
			kernelgen_print_error(kernelgen_launch_verbose,
				"Cannot create kernel %s, status = %d: %s\n",
				l->kernel_name, result.value, kernelgen_get_error_string(result));
		}
		goto finish;
	}

	result.value = clBuildProgram(opencl->program,
		1, &device, NULL, NULL, NULL);
	if (result.value != CL_SUCCESS)
	{
		kernelgen_print_error(kernelgen_launch_verbose,
			"Cannot build program for kernel %s, status = %d: %s\n",
			l->kernel_name, result.value, kernelgen_get_error_string(result));
		status = clGetProgramBuildInfo(opencl->program, device,
			CL_PROGRAM_BUILD_LOG, l->kernel_source_size,
			l->kernel_source, NULL);
		if (status != CL_SUCCESS)
		{
			result.value = status;
			kernelgen_print_error(kernelgen_launch_verbose,
				"Cannot get kernel %s build log, status = %d: %s\n",
				l->kernel_name, result.value, kernelgen_get_error_string(result));
		}
		kernelgen_print_error(kernelgen_launch_verbose,
			"%s\n", l->kernel_source);
		goto finish;
	}
	
	// Create OpenCL kernel.
	opencl->kernel = clCreateKernel(opencl->program,
		l->kernel_name, &result.value);
	if (result.value != CL_SUCCESS)
	{
		kernelgen_print_error(kernelgen_launch_verbose,
			"Cannot create kernel %s, status = %d: %s\n",
			l->kernel_name, result.value, kernelgen_get_error_string(result));
		goto finish;
	}

	// Mark the precompiled kernel exists for entire
	// platform/device.
	l->config->last_platform_index = iplatform;
	l->config->last_device_index = idevice;
	
	free(target_source);

finish:
	kernelgen_set_last_error(result);
	return result;
#else
	kernelgen_status_t result;
	result.value = kernelgen_error_not_implemented;
	result.runmode = l->runmode;
	return result;
#endif
}

