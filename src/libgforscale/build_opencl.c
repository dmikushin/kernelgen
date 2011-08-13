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

	// Load OpenCL program from binary.
	cl_int status = CL_SUCCESS;
	opencl->program = clCreateProgramWithBinary(
		opencl->context, 1, &opencl->device, &l->kernel_binary_size,
		(const unsigned char **)&l->kernel_binary, &status, &result.value);
	/*opencl->program = clCreateProgramWithSource(
		opencl->context, 1, (const char**)&l->kernel_source,
		&l->kernel_source_size, &result.value);*/
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
		1, &opencl->device, NULL, NULL, NULL);
	if (result.value != CL_SUCCESS)
	{
		kernelgen_print_error(kernelgen_launch_verbose,
			"Cannot build program for kernel %s, status = %d: %s\n",
			l->kernel_name, result.value, kernelgen_get_error_string(result));
		result.value = clGetProgramBuildInfo(opencl->program, opencl->device,
			CL_PROGRAM_BUILD_LOG, l->kernel_source_size,
			&l->kernel_source, NULL);
		if (result.value != CL_SUCCESS)
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
 
