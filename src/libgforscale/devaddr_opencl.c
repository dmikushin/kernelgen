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

#include "gforscale_int.h"
#include "gforscale_int_opencl.h"

char* gforscale_devaddr_opencl_kernel_source =

"__kernel void gforscale_devaddr_opencl_kernel("
"	__global void* host_ptr, __global ptrdiff_t* dev_ptr)"
"{"
"	*dev_ptr = (ptrdiff_t)(host_ptr);"
"}";

// Get the device representation for the specified host container
// of device address.
gforscale_status_t gforscale_devaddr_opencl(
	struct gforscale_launch_config_t* l,
	void* host_ptr, size_t offset, void** dev_ptr)
{
#ifdef HAVE_OPENCL
	static cl_kernel kernel;
	static int kernel_compiled = 0;
	static cl_mem dev_ptr_dev;

	struct gforscale_opencl_config_t* opencl =
		(struct gforscale_opencl_config_t*)l->specific;

	// Being quiet optimistic initially...
	gforscale_status_t result;
	result.value = CL_SUCCESS;
	result.runmode = l->runmode;
	
	const char* kernel_name = "gforscale_devaddr_opencl_kernel";
	
	if (!kernel_compiled)
	{
		// Load OpenCL program from source.
		cl_int status = CL_SUCCESS;
		cl_program program = clCreateProgramWithSource(
			opencl->context, 1,
			(const char**)&gforscale_devaddr_opencl_kernel_source,
			NULL, &result.value);
		if ((result.value != CL_SUCCESS) || (status != CL_SUCCESS))
		{
			if (result.value != CL_SUCCESS)
			{
				gforscale_print_error(gforscale_launch_verbose,
					"Cannot create kernel %s, status = %d: %s\n",
					kernel_name, result.value, gforscale_get_error_string(result));
			}
			if (status != CL_SUCCESS)
			{
				result.value = status;
				gforscale_print_error(gforscale_launch_verbose,
					"Cannot create kernel %s, status = %d: %s\n",
					kernel_name, result.value, gforscale_get_error_string(result));
			}
			goto finish;
		}

		result.value = clBuildProgram(program,
			1, &opencl->device, NULL, NULL, NULL);
		if (result.value != CL_SUCCESS)
		{
			gforscale_print_error(gforscale_launch_verbose,
				"Cannot build program for kernel %s, status = %d: %s\n",
				kernel_name, result.value, gforscale_get_error_string(result));
			result.value = clGetProgramBuildInfo(opencl->program, opencl->device,
				CL_PROGRAM_BUILD_LOG, l->kernel_source_size,
				&l->kernel_source, NULL);
			if (result.value != CL_SUCCESS)
			{
				result.value = status;
				gforscale_print_error(gforscale_launch_verbose,
					"Cannot get kernel %s build log, status = %d: %s\n",
					kernel_name, result.value, gforscale_get_error_string(result));
			}
			gforscale_print_error(gforscale_launch_verbose,
				"%s\n", l->kernel_source);
			goto finish;
		}
		
		// Create OpenCL kernel.
		kernel = clCreateKernel(program,
			kernel_name, &result.value);
		if (result.value != CL_SUCCESS)
		{
			gforscale_print_error(gforscale_launch_verbose,
				"Cannot create kernel %s, status = %d: %s\n",
				kernel_name, result.value, gforscale_get_error_string(result));
			goto finish;
		}
		
		// Create device buffer.
		dev_ptr_dev = clCreateBuffer(opencl->context,
			CL_MEM_READ_WRITE, sizeof(ptrdiff_t), NULL, &result.value);
		if (result.value != CL_SUCCESS)
		{
			gforscale_print_error(gforscale_launch_verbose,
				"Cannot allocate device memory segment of size = %zu on device, status = %d: %s\n",
				sizeof(ptrdiff_t), result.value, gforscale_get_error_string(result));
			goto finish;
		}
		
		kernel_compiled = 1;
	}
	
	// Submit arguments.
	result.value = clSetKernelArg(
		kernel, 0, sizeof(void*), &host_ptr);
	if (result.value != CL_SUCCESS)
	{
		gforscale_print_error(gforscale_launch_verbose,
			"Cannot setup kernel argument, status = %d: %s\n",
			result.value, gforscale_get_error_string(result));
		goto finish;
	}
	/*long loffset = offset;
	result.value = clSetKernelArg(
		kernel, 1, sizeof(long), &loffset);
	if (result.value != CL_SUCCESS)
	{
		gforscale_print_error(gforscale_launch_verbose,
			"Cannot setup kernel argument, status = %d: %s\n",
			result.value, gforscale_get_error_string(result));
		goto finish;
	}*/
	result.value = clSetKernelArg(
		kernel, 1, sizeof(void*), &dev_ptr_dev);
	if (result.value != CL_SUCCESS)
	{
		gforscale_print_error(gforscale_launch_verbose,
			"Cannot setup kernel argument, status = %d: %s\n",
			result.value, gforscale_get_error_string(result));
		goto finish;
	}

	// Launch kernel.
	cl_event sync;
	result.value = clEnqueueTask(
		opencl->command_queue, kernel, 0, NULL, &sync);
	if (result.value != CL_SUCCESS)
	{
		gforscale_print_error(gforscale_launch_verbose,
			"Cannot launch kernel %s, status = %d: %s\n",
			kernel_name, result.value, gforscale_get_error_string(result));
		goto finish;
	}
	result.value = clWaitForEvents(1, &sync);
	if (result.value != CL_SUCCESS)
	{
		gforscale_print_error(gforscale_launch_verbose,
			"Cannot synchronize device running kernel %s, status = %d: %s\n",
			kernel_name, result.value, gforscale_get_error_string(result));
		goto finish;
	}
	
	// Copy back the result.
	result.value = clEnqueueReadBuffer(
		opencl->command_queue, dev_ptr_dev, CL_FALSE,
		0, sizeof(void*), dev_ptr, 0, NULL, &sync);
	if (result.value != CL_SUCCESS)
	{
		gforscale_print_error(gforscale_launch_verbose,
			"Cannot copy data from device to host, status = %d: %s\n",
			result.value, gforscale_get_error_string(result));
		goto finish;
	}
	
finish:
	gforscale_set_last_error(result);
	return result;
#else
	gforscale_status_t result;
	result.value = gforscale_error_not_implemented;
	result.runmode = l->runmode;
	gforscale_set_last_error(result);
	return result;
#endif
}

