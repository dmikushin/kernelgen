//===- Settings.cpp - core KernelGen API implementation -------------------===//
//
//     KernelGen -- A prototype of LLVM-based auto-parallelizing Fortran/C
//        compiler for NVIDIA GPUs, targeting numerical modeling code.
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements KernelGen configuration settings API.
//
//===----------------------------------------------------------------------===//

#include "KernelGen.h"
#include "Settings.h"
#include "Verbose.h"

#include <iostream>
#include <cstdlib>

kernelgen::Settings::Settings() : runmode(KERNELGEN_RUNMODE_UNDEF), verbose(0), debug(0)
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
