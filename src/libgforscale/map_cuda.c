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
#include "kernelgen_int_cuda.h"

kernelgen_status_t kernelgen_save_regions_cuda(
	struct kernelgen_launch_config_t* l, int nmapped);

#include "map_cuda.h"

#define HAVE_MAPPING
#define kernelgen_load_regions_cuda kernelgen_map_regions_cuda
#define kernelgen_save_regions_cuda kernelgen_unmap_regions_cuda

kernelgen_status_t kernelgen_save_regions_cuda(
	struct kernelgen_launch_config_t* l, int nmapped);

#include "map_cuda.h"

