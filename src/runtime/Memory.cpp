//===- Memory.cpp - Device memory pool for NVIDIA GPUs --------------------===//
//
//     KernelGen -- A prototype of LLVM-based auto-parallelizing Fortran/C
//        compiler for NVIDIA GPUs, targeting numerical modeling code.
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains functions to support the dynamic memory heap in GPU
// global memory. The rationale is to replace builtin malloc/free calls, since
// they are incompatible with concurrent kernels execution.
//
//===----------------------------------------------------------------------===//

#include "Runtime.h"

#include "kernelgen_interop.h"

#include "llvm/IR/LLVMContext.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/IRReader/IRReader.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Program.h"

using namespace kernelgen;
using namespace kernelgen::bind::cuda;
using namespace kernelgen::runtime;
using namespace llvm;
using namespace llvm::sys;
using namespace llvm::sys::fs;
using namespace std;

// Setup the device global memory pool initial configuration.
kernelgen_memory_t* kernelgen::runtime::InitMemoryPool(size_t szpool)
{
	szpool += 4096;

	// First, fill config on host.
	kernelgen_memory_t config_host;

	// Allocate pool and flush it to zero.
	int err = cuMemAlloc((void**)&config_host.pool, szpool);
	if (err) THROW("Error in cuMemAlloc: " << err);
	err = cuMemsetD8(config_host.pool, 0, szpool);
	if (err) THROW("Error in cuMemsetD8: " << err);

	config_host.szused = 0;
	config_host.szpool = szpool;
	config_host.count = 0;
	config_host.pool += 4096 - (size_t)config_host.pool % 4096;

	// Copy the resulting config to the special
	// device variable.
	kernelgen_memory_t* config_device = NULL;
	err = cuMemAlloc((void**)&config_device, szpool);
	if (err) THROW("Error in cuMemAlloc: " << err);
	err = cuMemcpyHtoD(config_device, &config_host, sizeof(kernelgen_memory_t));
	if (err) THROW("Error in cuMemcpyH2D: " << err);
	return config_device;
}

