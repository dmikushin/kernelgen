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

#include "llvm/LLVMContext.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/IRReader.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Program.h"

using namespace kernelgen;
using namespace kernelgen::bind::cuda;
using namespace kernelgen::runtime;
using namespace llvm;
using namespace llvm::sys;
using namespace llvm::sys::fs;
using namespace std;

extern "C" int cudaMalloc(void** ptr, size_t size)
{
	struct cudaMalloc_t
	{
		void** ptr;
		size_t size;
		void* ptrv;
	};

	static Module* malloc_module = NULL;
	static KernelFunc malloc_kernel;
	static cudaMalloc_t* args_dev = NULL;

	CUstream stream = cuda_context->getSecondaryStream();

	if (!malloc_module)
	{
		// Load LLVM IR for kernelgen_malloc function.
		Path kernelgenSimplePath(Program::FindProgramByName("kernelgen-simple"));
		if (kernelgenSimplePath.empty())
			THROW("Cannot locate kernelgen binaries folder, is it included into $PATH ?");
		string kernelgenPath = kernelgenSimplePath.getDirname().str();
		string monitorModulePath = kernelgenPath + "/../include/cuda/malloc.bc";
		std::ifstream tmp_stream(monitorModulePath.c_str());
		tmp_stream.seekg(0, std::ios::end);
		string monitor_source = "";
		monitor_source.reserve(tmp_stream.tellg());
		tmp_stream.seekg(0, std::ios::beg);

		monitor_source.assign(
				std::istreambuf_iterator<char>(tmp_stream),
				std::istreambuf_iterator<char>());
		tmp_stream.close();

		string err;
		MemoryBuffer* buffer = MemoryBuffer::getMemBuffer(
				monitor_source);
		malloc_module = ParseBitcodeFile(buffer, getGlobalContext(), &err);
		if (!malloc_module)
			THROW("Cannot load KernelGen monitor kernel module: " << err);

		// Codegen and load kernelgen_malloc kernel onto GPU.	
		Kernel malloc;
		malloc.name = "kernelgen_malloc";
		malloc_kernel = kernelgen::runtime::Codegen(KERNELGEN_RUNMODE_CUDA,
				&malloc, malloc_module);

		// Allocate space for args.
		CU_SAFE_CALL(cuMemAlloc((void**)&args_dev, sizeof(cudaMalloc_t)));
	}

	// Pack call arguments.
	{
		cudaMalloc_t args;
		args.ptr = &args_dev->ptrv;
		args.size = size;
		CU_SAFE_CALL(cuMemcpyHtoDAsync(args_dev, &args, sizeof(cudaMalloc_t), stream));
	}

	struct {
		unsigned int x, y, z;
	} gridDim, blockDim;
	gridDim.x = 1;
	gridDim.y = 1;
	gridDim.z = 1;
	blockDim.x = 1;
	blockDim.y = 1;
	blockDim.z = 1;
	size_t szshmem = 0;
	char args[256];
	memcpy(args, &args_dev, sizeof(void*));
	memcpy(args + sizeof(void*), &memory_pool, sizeof(void*));
	CU_SAFE_CALL(cudyLaunch((CUDYfunction) malloc_kernel, gridDim.x,
			gridDim.y, gridDim.z, blockDim.x, blockDim.y,
			blockDim.z, szshmem, args,
			cuda_context->getSecondaryStream(), NULL));

	// Wait for malloc kernel completion.
	CU_SAFE_CALL(cuStreamSynchronize(cuda_context->getSecondaryStream()));

	// Unpack call result.
	{
		CU_SAFE_CALL(cuMemcpyDtoHAsync(ptr, &args_dev->ptrv, sizeof(void*), stream));
	}

	return 0;
}

extern "C" int cudaFree(void* ptr)
{
	if (!ptr) return 0;

	return 0;
}

#define cudaMemcpyHostToDevice 1
#define cudaMemcpyDeviceToHost 2

extern "C" int cudaMemcpy(void* dst, void* src, size_t size, int kind)
{
	switch (kind)
	{
	case cudaMemcpyHostToDevice :
		CU_SAFE_CALL(cuMemcpyHtoDAsync(dst, src, size,
			cuda_context->getSecondaryStream()));
		return 0;
	case cudaMemcpyDeviceToHost :
		CU_SAFE_CALL(cuMemcpyDtoHAsync(dst, src, size,
			cuda_context->getSecondaryStream()));
		return 0;
	default :
		THROW("Unsupported cudaMemcpy wrapper copying kind: " << kind);
	}
}

