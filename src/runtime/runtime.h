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

#ifndef KERNELGEN_RUNTIME_H
#define KERNELGEN_RUNTIME_H

#include "llvm/Instructions.h"
#include "llvm/Module.h"
#include "llvm/Target/TargetMachine.h"

#include <cstdarg>
#include <map>
#include <string>

#include "kernelgen_interop.h"

#define KERNELGEN_RUNMODE_COUNT		3
#define KERNELGEN_RUNMODE_NATIVE	0
#define KERNELGEN_RUNMODE_CUDA		1
#define KERNELGEN_RUNMODE_OPENCL	2

// Unified kernel or hostcall arguments descriptor.
struct kernelgen_callback_data_t
{
	// The LLVM function type.
	llvm::FunctionType* FunctionTy;
	
	// The LLVM structure type containing function
	// arguments in aggregated form (including
	// FunctionTy and StructTy themselves). If
	// function is not void, last structure field
	// is a placeholder for the return value.
	llvm::StructType* StructTy;
	
	// The structure of kernel arguments.
	// Contrary to the arguments order defined by
	// function type, arguments are REORDERED in a way
	// that all integer values go first.
	void* args;
};

namespace kernelgen {

// Kernels runmode (target).
extern int runmode;

// Verbose output.
extern bool verbose;

// The prototype of kernel function.
// Thanks to arguments aggregation, all
// kernels could have the same prototype.
typedef void (*kernel_func_t)(void* args);

// The type of binaries map.
typedef std::map<std::string, kernel_func_t> binaries_map_t;

// Kernel configuration structure
// containing pointer for original source code
// and space to store specialized source and
// binary variants for each target.
struct kernel_t
{
	// Kernel name.
	std::string name;

	// Kernel module.
	llvm::Module* module;
	
	// Kernel function.
	llvm::Function* function;

	// Kernel LLVM IR source code.
	std::string source;

	// Target-specific configuration.
	struct
	{
		// Indicates if entire runmode is supported
		// for specific kernel (it could be unsupported,
		// for example, in case of CUDA target and kernel loop,
		// that performs hostcalls).
		bool supported;
		
		// Kernel source version, more close to specific target.
		std::string source;

		// References to tables of compiled kernels
		// for each supported runmode.
		// Each source may have multiple binaries identified
		// by hash stamps, optimized for different combinations
		// of kernel arguments values.
		binaries_map_t binaries;
		
		// The universal binary, that is not optimized for
		// any kernel arguments.
		kernel_func_t binary;
		
		// Monitoring kernel (applicable for some targets).
		kernel_func_t monitor_kernel_func;
		
		// Kernel callback structure.
		kernelgen_callback_t* callback;
		
		// Streams for work and monitor kernels.
		void* monitor_kernel_stream;
		void* kernel_stream;
	}
	target[KERNELGEN_RUNMODE_COUNT];
};

// The pool of already loaded kernels.
// After kernel is loaded, we pin it here
// for futher references.
extern std::map<std::string, kernel_t*> kernels;

// Target machines for runmodes.
extern std::auto_ptr<llvm::TargetMachine> targets[KERNELGEN_RUNMODE_COUNT];

namespace runtime {

// Compile kernel with the specified arguments,
// and return its handle.
kernel_func_t compile(int runmode, kernel_t* kernel, llvm::Module* module = NULL);

// Compile C source to PTX using NVISA-enabled
// Open64 compiler variant.
kernel_func_t nvopencc(std::string source, std::string name);

// Setup the device global memory pool initial configuration.
kernelgen_memory_t* init_memory_pool(size_t szpool);

// Wrap call instruction into host function call wrapper.
llvm::CallInst* wrapCallIntoHostcall(llvm::CallInst* call);

} }

// Launch the specified kernel.
extern "C" int kernelgen_launch(kernelgen::kernel_t* kernel,
	unsigned long long szdata, unsigned long long szdatai,
	kernelgen_callback_data_t* data);

// Finish kernel execution.
extern "C" void kernelgen_finish();

// Launch function defined by the specified entry
// point on host.
extern "C" void kernelgen_hostcall(kernelgen::kernel_t* kernel,
	unsigned long long szdata, unsigned long long szdatai,
	kernelgen_callback_data_t* data);

#endif // KERNELGEN_RUNTIME_H

