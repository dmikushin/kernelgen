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

#ifdef HAVE_CUDA
#include <cuda_runtime.h>
#endif

#ifdef HAVE_OPENCL
#include <CL/cl.h>
#endif

#include <stack>

// Defines gforscale errors stack.
static std::stack<gforscale_status_t> gforscale_error_stack;

// Get the last gforscale error.
extern "C" gforscale_status_t gforscale_get_last_error()
{
	if (gforscale_error_stack.size())
		return gforscale_error_stack.top();
	
	gforscale_status_t success;
	success.value = gforscale_success;
	success.runmode = 0;
	return success;
}

// Pop the last gforscale error.
extern "C" gforscale_status_t gforscale_pop_last_error()
{
	gforscale_status_t status;
	status.value = gforscale_success;
	status.runmode = 0;
	
	if (gforscale_error_stack.size())
	{
		status = gforscale_error_stack.top();
		gforscale_error_stack.pop();
	}
	
	return status;
}

// Get text message for the specified error code.
extern "C" const char* gforscale_get_error_string(gforscale_status_t error)
{
	switch (error.value)
	{
		case gforscale_success :			return "No error";
		case gforscale_initialization_failed :		return "Initialization failed";
		case gforscale_error_not_found :		return "Entity not found";
		case gforscale_error_not_implemented :		return "Used functionality is not implemented";
		case gforscale_error_ffi_setup :		return "Error in FFI setup";
		case gforscale_error_results_mismatch :		return "Kernel and control results mismatch";
	}

#ifdef HAVE_CUDA
	if (error.runmode == GFORSCALE_RUNMODE_DEVICE_CUDA)
		return cudaGetErrorString((cudaError_t)error.value);
#endif

#ifdef HAVE_OPENCL
	if (error.runmode == GFORSCALE_RUNMODE_DEVICE_OPENCL)
	{
		switch(error.value)
		{
		case CL_SUCCESS :				return "No error";
		case CL_DEVICE_NOT_FOUND :			return "Device not found";
		case CL_DEVICE_NOT_AVAILABLE :			return "Device not available";
		case CL_COMPILER_NOT_AVAILABLE :		return "Compiler not available";
		case CL_MEM_OBJECT_ALLOCATION_FAILURE :		return "Memory object allocation failure";
		case CL_OUT_OF_RESOURCES :			return "Out of resources";
		case CL_OUT_OF_HOST_MEMORY :			return "Out of host memory";
		case CL_PROFILING_INFO_NOT_AVAILABLE :		return "Profiling information not available";
		case CL_MEM_COPY_OVERLAP :			return "Memory copy overlap";
		case CL_IMAGE_FORMAT_MISMATCH :			return "Image format mismatch";
		case CL_IMAGE_FORMAT_NOT_SUPPORTED :		return "Image format not supported";
		case CL_BUILD_PROGRAM_FAILURE :			return "Program build failure";
		case CL_MAP_FAILURE :				return "Map failure";
		case CL_INVALID_VALUE :				return "Invalid value";
		case CL_INVALID_DEVICE_TYPE :			return "Invalid device type";
		case CL_INVALID_PLATFORM :			return "Invalid platform";
		case CL_INVALID_DEVICE :			return "Invalid device";
		case CL_INVALID_CONTEXT :			return "Invalid context";
		case CL_INVALID_QUEUE_PROPERTIES :		return "Invalid queue properties";
		case CL_INVALID_COMMAND_QUEUE :			return "Invalid command queue";
		case CL_INVALID_HOST_PTR :			return "Invalid host pointer";
		case CL_INVALID_MEM_OBJECT :			return "Invalid memory object";
		case CL_INVALID_IMAGE_FORMAT_DESCRIPTOR :	return "Invalid image format descriptor";
		case CL_INVALID_IMAGE_SIZE :			return "Invalid image size";
		case CL_INVALID_SAMPLER :			return "Invalid sampler";
		case CL_INVALID_BINARY :			return "Invalid binary";
		case CL_INVALID_BUILD_OPTIONS :			return "Invalid build options";
		case CL_INVALID_PROGRAM :			return "Invalid program";
		case CL_INVALID_PROGRAM_EXECUTABLE :		return "Invalid program executable";
		case CL_INVALID_KERNEL_NAME :			return "Invalid kernel name";
		case CL_INVALID_KERNEL_DEFINITION :		return "Invalid kernel definition";
		case CL_INVALID_KERNEL :			return "Invalid kernel";
		case CL_INVALID_ARG_INDEX :			return "Invalid argument index";
		case CL_INVALID_ARG_VALUE :			return "Invalid argument value";
		case CL_INVALID_ARG_SIZE :			return "Invalid argument size";
		case CL_INVALID_KERNEL_ARGS :			return "Invalid kernel arguments";
		case CL_INVALID_WORK_DIMENSION :		return "Invalid work dimension";
		case CL_INVALID_WORK_GROUP_SIZE :		return "Invalid work group size";
		case CL_INVALID_WORK_ITEM_SIZE :		return "Invalid work item size";
		case CL_INVALID_GLOBAL_OFFSET :			return "Invalid global offset";
		case CL_INVALID_EVENT_WAIT_LIST :		return "Invalid event wait list";
		case CL_INVALID_EVENT :				return "Invalid event";
		case CL_INVALID_OPERATION :			return "Invalid operation";
		case CL_INVALID_GL_OBJECT :			return "Invalid OpenGL object";
		case CL_INVALID_BUFFER_SIZE :			return "Invalid buffer size";
		case CL_INVALID_MIP_LEVEL :			return "Invalid mip-map level";
		}
	}
#endif
	return "Unknown error";
}

// Push the last gforscale error.
extern "C" void gforscale_set_last_error(gforscale_status_t error)
{
	if (error.value != gforscale_success)
		gforscale_error_stack.push(error);
}

