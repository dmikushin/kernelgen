//===- Runtime.h - KernelGen runtime API ----------------------------------===//
//
//     KernelGen -- A prototype of LLVM-based auto-parallelizing Fortran/C
//        compiler for NVIDIA GPUs, targeting numerical modeling code.
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines KernelGen runtime API.
//
//===----------------------------------------------------------------------===//

#ifndef KERNELGEN_RUNTIME_H
#define KERNELGEN_RUNTIME_H

#include "Cuda.h"
#include "llvm/Instructions.h"
#include "llvm/Module.h"
#include "llvm/Support/raw_os_ostream.h"
#include "llvm/Target/TargetMachine.h"

#include <cstdarg>
#include <map>
#include <ostream>
#include <string>

#include "KernelGen.h"

#include "kernelgen_interop.h"

// Unified kernel or hostcall arguments descriptor.
struct CallbackData
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

// The prototype of kernel function.
// Thanks to arguments aggregation, all
// kernels could have the same prototype.
typedef void (*KernelFunc)(void* args);

// The type of binaries map.
typedef std::map<std::string, KernelFunc> binaries_map_t;

// Kernel configuration structure
// containing pointer for original source code
// and space to store specialized source and
// binary variants for each target.
struct Kernel
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
		KernelFunc binary;
		
		// Kernel callback structure.
		kernelgen_callback_t* callback;
		
		// Kernel launch parameters
		dim3 gridDim, blockDim;

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
extern std::map<std::string, Kernel*> kernels;

// The array contains addresses of globalVatiables
extern uint64_t *AddressesOfGVars;
extern int NumOfGVars;

// order of globals in which they were stored in addressesOfGlobalVariables
extern std::map<llvm::StringRef, uint64_t> orderOfGlobals;

// Target machines for runmodes.
extern std::auto_ptr<llvm::TargetMachine> targets[KERNELGEN_RUNMODE_COUNT];

namespace runtime {

// Compile kernel with the specified arguments,
// and return its handle.
KernelFunc Compile(int runmode, Kernel* kernel, llvm::Module* module = NULL, void* data = NULL, int szdata = 0, int szdatai = 0);

// Compile C source to x86 binary or PTX assembly,
// using the corresponding LLVM backends.
KernelFunc Codegen(int runmode, Kernel* kernel, llvm::Module* module);

// CUDA runtime context.
extern kernelgen::bind::cuda::context* cuda_context;

// Setup the device global memory pool initial configuration.
kernelgen_memory_t* init_memory_pool(size_t szpool);

// Wrap call instruction into host function call wrapper.
llvm::CallInst* WrapCallIntoHostcall(llvm::CallInst* call, Kernel* kernel);

// Monitoring module and kernel (applicable for some targets).
extern llvm::Module* monitor_module;
extern KernelFunc monitor_kernel;

// Runtime module (applicable for some targets).
extern llvm::Module* runtime_module;

// CUDA module (applicable for some targets).
extern llvm::Module* cuda_module;

} }

// Launch the specified kernel.
extern "C" int kernelgen_launch(kernelgen::Kernel* kernel,
	unsigned long long szdata, unsigned long long szdatai,
	CallbackData* data);

// Start kernel execution.
extern "C" void kernelgen_start();

// Finish kernel execution.
extern "C" void kernelgen_finish();

// Launch function defined by the specified entry
// point on host.
extern "C" void kernelgen_hostcall(kernelgen::Kernel* kernel,
	llvm::FunctionType* FTy, llvm::StructType* StructTy, void* params);

// Synchronize GPU data modifications made on host
// during kernelgen_hostcall.
extern "C" void kernelgen_hostcall_memsync();

#endif // KERNELGEN_RUNTIME_H

