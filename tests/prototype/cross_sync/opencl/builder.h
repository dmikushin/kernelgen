/*
 * KernelGen - the LLVM-based compiler with GPU kernels generation over C backend.
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

#include <CL/cl.h>
#include <CL/cl_ext.h>

typedef struct
{
	int count;
	cl_device_id device;
	cl_context context;
	cl_program program;
	cl_kernel* kernels;
}
device_config_t;

typedef struct
{
	cl_platform_id id;
	device_config_t cpu, gpu;
}
builder_config_t;

builder_config_t* builder_init(
	const char* filename, const char* options, int nkernels);

int builder_deinit(builder_config_t* config);

