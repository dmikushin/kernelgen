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

#include "Settings.h"
#include "Verbose.h"

#include "throw.h"

#include <stdlib.h>

kernelgen::Settings::Settings() : runmode(-1), verbose(0), debug(0)
{
	char* crunmode = getenv("kernelgen_runmode");
	if (crunmode) {
		runmode = atoi(crunmode);

		// Load verbose level.
		char* cverbose = getenv("kernelgen_verbose");
		if (cverbose)
			verbose = atoi(cverbose);

		// Load debug level.
		char* cdebug = getenv("kernelgen_debug");
		if (cdebug)
			debug = atoi(cdebug);

		// CUDA target specific: default subarchitecture.
		subarch = getenv("kernelgen_subarch");

		// Check the valid runmode.
		switch (runmode) {
		case KERNELGEN_RUNMODE_NATIVE:
			VERBOSE("Using KernelGen/NATIVE");
			break;
		case KERNELGEN_RUNMODE_CUDA:
			VERBOSE("Using KernelGen/CUDA");
			break;
		case KERNELGEN_RUNMODE_OPENCL:
			VERBOSE("Using KernelGen/OpenCL");
			break;
		default:
			THROW("Unknown runmode " << runmode);
		}
	}
}
