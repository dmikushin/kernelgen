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

#include "cuda.h"
#include "llvm/Instructions.h"
#include "llvm/Module.h"
#include "llvm/Support/raw_os_ostream.h"
#include "llvm/Target/TargetMachine.h"

#include <cstdarg>
#include <map>
#include <ostream>
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
struct Size3 {
public:
	int64_t x,y,z;
	Size3():x(-1),y(-1),z(-1) { }
	Size3(int64_t ar[3]) {
		x = (int) ar[0];
		y = (int) ar[1];
		z = (int) ar[2];
	}
	void reset() {
		x=y=z=-1;
	}
	Size3(int64_t _x,int64_t _y,int64_t _z)
	:x(_x),y(_y),z(_z) {}
	void writeToArray(int64_t arr[3])
	{
		arr[0] = x;
		arr[1] = y;
		arr[2] = z;
	}
    int getNumOfDimensions()
	{
		if(x == -1) return 0;
		if(y == -1) return 1;
		if(z == -1) return 2;
		return 3;
	}

};

struct dim3 { 
	unsigned int x, y, z;
	friend std::ostream & operator<<(std::ostream &stream, dim3 & dim)
	{
		stream << "{ " << dim.x << ", " << dim.y << ", " << dim.z << " }";
		return stream;
	} 
	friend llvm::raw_ostream & operator<<(llvm::raw_ostream &stream, dim3 & dim)
	{
		stream << "{ " << dim.x << ", " << dim.y << ", " << dim.z << " }";
		return stream;
	}
	dim3(unsigned int _x=1,unsigned int _y=1,unsigned int _z=1)
	:x(_x),y(_y),z(_z) {}
	};

namespace kernelgen {

// Kernels runmode (target).
extern int runmode;

// Verbose output.
extern int verbose;

// Debug mode.
extern bool debug;

#define KERNELGEN_VERBOSE_DISABLE	0
#define KERNELGEN_VERBOSE_SUMMARY	1 << 0
#define KERNELGEN_VERBOSE_SOURCES	1 << 1
#define KERNELGEN_VERBOSE_ISA		1 << 2
#define KERNELGEN_VERBOSE_DATAIO	1 << 3
#define KERNELGEN_VERBOSE_HOSTCALL	1 << 4
#define KERNELGEN_VERBOSE_POLLYGEN	1 << 5
#define KERNELGEN_VERBOSE_TIMEPERF	1 << 6

// Define to load kernels lazily. Means instead of reading and verifying
// modules during application startup, they are instead loaded in the place
// of first use.
#define KERNELGEN_LOAD_KERNELS_LAZILY

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
#ifdef KERNELGEN_LOAD_KERNELS_LAZILY
	// Kernel loaded marker (for lazy processing).
	bool loaded;
#endif
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
		
		// Kernel callback structure.
		kernelgen_callback_t* callback;
		
		// Kernel launch parameters
		dim3 gridDim, blockDim;
		
		// Streams for work and monitor kernels.
		CUstream monitor_kernel_stream;
		CUstream kernel_stream;

		// Kernel execution statistics: min/max/avg time and
		// the number of launches.
		struct
		{
			double min, max, avg;
			unsigned long long nlaunches;
		}
		stats;
	}
	target[KERNELGEN_RUNMODE_COUNT];
};

// The pool of already loaded kernels.
// After kernel is loaded, we pin it here
// for futher references.
extern std::map<std::string, kernel_t*> kernels;

// The array contains addresses of globalVatiables
extern uint64_t *addressesOfGlobalVariables;
extern int numberOfGlobalVariables;

// order of globals in which they were stored in addressesOfGlobalVariables
extern std::map<llvm::StringRef, uint64_t> orderOfGlobals;

// Target machines for runmodes.
extern std::auto_ptr<llvm::TargetMachine> targets[KERNELGEN_RUNMODE_COUNT];

namespace runtime {

// Compile kernel with the specified arguments,
// and return its handle.
kernel_func_t compile(int runmode, kernel_t* kernel, llvm::Module* module = NULL, void* data = NULL, int szdata = 0, int szdatai = 0);

// Compile C source to x86 binary or PTX assembly,
// using the corresponding LLVM backends.
kernel_func_t codegen(int runmode, kernel_t* kernel, llvm::Module* module);

// CUDA runtime context.
extern kernelgen::bind::cuda::context* cuda_context;

// Setup the device global memory pool initial configuration.
kernelgen_memory_t* init_memory_pool(size_t szpool);

// Wrap call instruction into host function call wrapper.
llvm::CallInst* wrapCallIntoHostcall(llvm::CallInst* call, kernel_t* kernel);

// Monitoring module and kernel (applicable for some targets).
extern llvm::Module* monitor_module;
extern kernel_func_t monitor_kernel;

// Runtime module (applicable for some targets).
extern llvm::Module* runtime_module;

// CUDA module (applicable for some targets).
extern llvm::Module* cuda_module;

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
	llvm::FunctionType* FTy, llvm::StructType* StructTy, void* params);

#endif // KERNELGEN_RUNTIME_H

