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

#ifndef KERNELGEN_INT_CUDA_H
#define KERNELGEN_INT_CUDA_H

#ifdef HAVE_CUDA
#include <cuda_runtime.h>

#ifdef __cplusplus
extern "C"
{
#endif

// Defines CUDA-specific kernel configuration.
struct kernelgen_cuda_config_t
{
	// Flag indicating if memory regions must be aligned
	// before mapping onto device.
	int aligned;
};

#ifdef __cplusplus
}
#endif

#endif // HAVE_CUDA

#endif // KERNELGEN_INT_CUDA_H
