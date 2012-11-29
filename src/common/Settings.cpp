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

#include "KernelGen.h"
#include "Settings.h"
#include "Verbose.h"

#include "throw.h"

#include <iostream>
#include <cstdlib>

kernelgen::Settings::Settings() : verbose(0), debug(0), subarch("")
{
	char* crunmode = getenv("kernelgen_runmode");
	if (crunmode) {
		runmode = atoi(crunmode);

		// Load verbose level.
		char* cverbose = getenv("kernelgen_verbose");
		if (cverbose)
			verbose.setMode(Verbose::Mode(atoi(cverbose)));

		// Load debug level.
		char* cdebug = getenv("kernelgen_debug");
		if (cdebug)
			debug = atoi(cdebug);

		// CUDA target specific: default subarchitecture.
		const char* csubarch = getenv("kernelgen_subarch");
		if (csubarch)
			subarch = csubarch;

		// Check the valid runmode.
		switch (runmode) {
		case KERNELGEN_RUNMODE_NATIVE:
			VERBOSE("Using KernelGen/NATIVE\n");
			break;
		case KERNELGEN_RUNMODE_CUDA:
			VERBOSE("Using KernelGen/CUDA\n");
			break;
		case KERNELGEN_RUNMODE_OPENCL:
			VERBOSE("Using KernelGen/OpenCL\n");
			break;
		default:
			THROW("Unknown runmode " << RUNMODE, RUNMODE);
		}
	}
}
