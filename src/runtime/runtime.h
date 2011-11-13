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

#include "llvm/Module.h"

#include <cstdarg>
#include <map>
#include <string>

#ifndef KERNELGEN_RUNTIME_H
#define KERNELGEN_RUNTIME_H

#define KERNELGEN_RUNMODE_COUNT		3
#define KERNELGEN_RUNMODE_NATIVE	0
#define KERNELGEN_RUNMODE_CUDA		1
#define KERNELGEN_RUNMODE_OPENCL	2

// Launch kernel from the spepcified source code address.
extern "C" int kernelgen_launch(char* kernel, int* args);

namespace kernelgen {

// Kernels runmode (target).
extern int runmode;

// Verbose output.
extern bool verbose;

// The prototype of kernel function.
// Thanks to arguments aggregation, all
// kernels could have the same prototype.
typedef void (*kernel_func_t)(int* args);

// Kernel configuration structure
// containing pointer for original source code
// and space to store specialized source and
// binary variants for each target.
typedef struct
{
	// Kernel name.
	std::string name;

	// Kernel module.
	llvm::Module* module;
	
	// Kernel function.
	llvm::Function* function;

	// Kernel LLVM IR source code.
	std::string source;

	// Target-specific
	struct
	{
		// References to tables of compiled kernels
		// for each supported runmode.
		// Each source may have multiple binaries identified
		// by hash stamps, optimized for different combinations
		// of kernel arguments values.
		std::map<std::string, char*> binary;
		
		// Kernel source version, more close to specific target.
		std::string source;
	}
	target[KERNELGEN_RUNMODE_COUNT];
}
kernel_t;

// The pool of already loaded kernels.
// After kernel is loaded, we pin it here
// for futher references.
extern std::map<std::string, kernel_t*> kernels;

namespace runtime {

void compile(int runmode, kernel_t* kernel, int* args);

} }

#endif // KERNELGEN_RUNTIME_H

