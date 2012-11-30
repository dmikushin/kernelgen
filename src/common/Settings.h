//===- Settings.h - core KernelGen API implementation ---------------------===//
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

#ifndef SETTINGS_H
#define SETTINGS_H

#include <Verbose.h>

#include <string>

#define RUNMODE              kernelgen::settings.getRunmode()
#define VERBOSE(val)    do { kernelgen::settings.getVerbose() << val; } while (0)
#define KDEBUG               kernelgen::settings.getDebug()
#define SUBARCH              kernelgen::settings.getSubarch()
#define THROW(msg, ...) do { std::cerr << __FILE__ << ":" << __LINE__ << " " << msg << std::endl; \
                             throw __VA_ARGS__; } while (0)

#define KERNELGEN_RUNMODE_UNDEF   (-1)
#define KERNELGEN_RUNMODE_NATIVE	0
#define KERNELGEN_RUNMODE_CUDA		1
#define KERNELGEN_RUNMODE_OPENCL	2
#define KERNELGEN_RUNMODE_COUNT		3

namespace kernelgen {

	class Settings
	{
		// Kernels runmode (target).
		int runmode;

		// Verbose output.
		Verbose verbose;

		// Debug mode.
		int debug;

		// Subarchitecture setting, e.g. "sm_20" for sm_20 CUDA target.
		// This setting overrides the value returned by device CC query
		// (if less or equal).
		std::string subarch;

	public :

		inline int getRunmode() const { return runmode; }
		inline Verbose& getVerbose() { return verbose; }
		inline Verbose::Mode getVerboseMode() const { return verbose.getMode(); }
		inline int getDebug() const { return debug; }
		inline std::string getSubarch() const { return subarch; }

		Settings();
	};

} // kernelgen

#endif // SETTINGS_H
