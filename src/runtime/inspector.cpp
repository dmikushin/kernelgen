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

#include "llvm/Constants.h"
#include "llvm/Instructions.h"
#include "llvm/LLVMContext.h"
#include "llvm/Module.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/IRReader.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Program.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/ToolOutputFile.h"

#include "elf.h"
#include "util.h"
#include "runtime.h"
#include "kernelgen_interop.h"

#include <ffi.h>
#include <fstream>
#include <iostream>
#include <cstdlib>

#include "TrackedPassManager.h"

// Regular main entry.
extern "C" int __regular_main(int argc, char* argv[])
{
	return 0;
}

using namespace kernelgen;
using namespace kernelgen::runtime;
using namespace kernelgen::bind::cuda;
using namespace llvm;
using namespace llvm::sys;
using namespace llvm::sys::fs;
using namespace std;
using namespace util::elf;

int main(int argc, char* argv[])
{
	tracker = new PassTracker("codegen", NULL, NULL);

	LLVMContext& context = getGlobalContext();

	if (argc != 2)
	{
		cout << argv[0] << " - a short-circuit loader for inspecting the failing LLVM IR kernels" << endl;
		cout << "Usage: kernelgen_runmode=N" << argv[0] << " <object_name>" << endl;
		return 0;
	}

	char* crunmode = getenv("kernelgen_runmode");
	if (!crunmode)
	{
		cout << "The kernelgen_runmode environment variable must be set, or otherwise the tool is useless" << endl;
		return 1;
	}
	
	runmode = atoi(crunmode);

	// Check requested verbosity level.
	char* cverbose = getenv("kernelgen_verbose");
	if (cverbose) verbose = atoi(cverbose);
	
	if (verbose)
	{
		switch (runmode)
		{
		case KERNELGEN_RUNMODE_NATIVE :
			cout << "Using KernelGen/NATIVE" << endl;
			break;
		case KERNELGEN_RUNMODE_CUDA :
			cout << "Using KernelGen/CUDA" << endl;
			break;
		case KERNELGEN_RUNMODE_OPENCL :
			cout << "Using KernelGen/OpenCL" << endl;
			break;
		}
		cout << endl;
	}

	// Build kernels index.
	if (verbose) cout << "Building kernels index ..." << endl;
	celf e(argv[1], "");
	cregex regex("^__kernelgen_.*$", REG_EXTENDED | REG_NOSUB);
	vector<csymbol*> symbols = e.getSymtab()->find(regex);
	for (vector<csymbol*>::iterator i = symbols.begin(),
		ie = symbols.end(); i != ie; i++)
	{
		csymbol* symbol = *i;
		const char* data = symbol->getData();
		const string& name = symbol->getName();
		kernel_t* kernel =  new kernel_t();
		kernel->name = name;
		kernel->source = data;

		// Initially, all targets are supported.
		for (int ii = 0; ii < KERNELGEN_RUNMODE_COUNT; ii++)
		{
			kernel->target[ii].supported = true;
			kernel->target[ii].binary = NULL;
		}
		
		kernels[name] = kernel;
		if (verbose) cout << name << endl;
	}
	if (verbose) cout << endl;

	if (!symbols.size())
	{
		cout << "No kernels found in " << argv[1] << endl;
		return 1;
	}

	kernel_t* kernel = kernels.begin()->second;
	if (symbols.size() != 1)
	{
		// TODO: select the kernel to load.
		cout << "The specified image contains multiple objects" << endl;
		return 1;
	}
	
	// Load arguments and their size for the selected kernel.
	kernelgen_callback_data_t* args = NULL;
	int szargs = 0, szargsi = 0;
	{
		cregex regex("^args" + kernel->name, REG_EXTENDED | REG_NOSUB);
		symbols = e.getSymtab()->find(regex);
		if (!symbols.size())
		{
			cout << "Cannot find args for " << kernel->name << endl;
			return 1;
		}
		args = (kernelgen_callback_data_t*)symbols[0]->getData();
	}
	{
		cregex regex("^szargs" + kernel->name, REG_EXTENDED | REG_NOSUB);
		symbols = e.getSymtab()->find(regex);
		if (!symbols.size())
		{
			cout << "Cannot find size of args for " << kernel->name << endl;
			return 1;
		}
		memcpy(&szargs, symbols[0]->getData(), sizeof(int));
	}
	{
		cregex regex("^szargsi" + kernel->name, REG_EXTENDED | REG_NOSUB);
		symbols = e.getSymtab()->find(regex);
		if (!symbols.size())
		{
			cout << "Cannot find size of integer args for " << kernel->name << endl;
			return 1;
		}
		memcpy(&szargsi, symbols[0]->getData(), sizeof(int));
	}

	// Execute kernel.
	switch (runmode)
	{
		case KERNELGEN_RUNMODE_NATIVE :
		{
			return kernelgen_launch(kernel, szargs, szargsi, args);
		}
		case KERNELGEN_RUNMODE_CUDA :
		{
			// Initialize dynamic kernels loader.
			kernelgen::runtime::cuda_context =
				kernelgen::bind::cuda::context::init(8192);

			// Create streams where monitoring and target kernels
			// will be executed.
			int err = cuStreamCreate(
				&kernel->target[runmode].monitor_kernel_stream, 0);
			if (err) THROW("Error in cuStreamCreate " << err);
			err = cuStreamCreate(
				&kernel->target[runmode].kernel_stream, 0);
			if (err) THROW("Error in cuStreamCreate " << err);
			
			// Load LLVM IR for kernel monitor, if not yet loaded.
			if (!monitor_module)
			{
				string monitor_source = "";
				std::ifstream tmp_stream("/opt/kernelgen/include/kernelgen_monitor.bc");
				tmp_stream.seekg(0, std::ios::end);
				monitor_source.reserve(tmp_stream.tellg());
				tmp_stream.seekg(0, std::ios::beg);

				monitor_source.assign(
					std::istreambuf_iterator<char>(tmp_stream),
					std::istreambuf_iterator<char>());
				tmp_stream.close();

			        SMDiagnostic diag;
			        MemoryBuffer* buffer1 = MemoryBuffer::getMemBuffer(
					monitor_source);
			        monitor_module = ParseIR(buffer1, diag, context);

				monitor_kernel = kernelgen::runtime::codegen(KERNELGEN_RUNMODE_CUDA,
					monitor_module, "kernelgen_monitor", 0);
			}

			// Load LLVM IR for KernelGen runtime functions, if not yet loaded.
			if (!runtime_module)
			{
				string runtime_source = "";
				std::ifstream tmp_stream("/opt/kernelgen/include/kernelgen_runtime.bc");
				tmp_stream.seekg(0, std::ios::end);
				runtime_source.reserve(tmp_stream.tellg());
				tmp_stream.seekg(0, std::ios::beg);

				runtime_source.assign(
					std::istreambuf_iterator<char>(tmp_stream),
					std::istreambuf_iterator<char>());
				tmp_stream.close();

				SMDiagnostic diag;
				MemoryBuffer* buffer1 = MemoryBuffer::getMemBuffer(
					runtime_source);
				runtime_module = ParseIR(buffer1, diag, context);
                        }

			// Load LLVM IR for CUDA runtime functions, if not yet loaded.
			if (!cuda_module)
			{
				string cuda_source = "";
				std::ifstream tmp_stream("/opt/kernelgen/include/kernelgen_cuda.bc");
				tmp_stream.seekg(0, std::ios::end);
				cuda_source.reserve(tmp_stream.tellg());
				tmp_stream.seekg(0, std::ios::beg);

				cuda_source.assign(
					std::istreambuf_iterator<char>(tmp_stream),
					std::istreambuf_iterator<char>());
				tmp_stream.close();

				SMDiagnostic diag;
				MemoryBuffer* buffer1 = MemoryBuffer::getMemBuffer(
					cuda_source);
				cuda_module = ParseIR(buffer1, diag, context);

				// Mark all module functions as device functions.
		                for (Module::iterator F = cuda_module->begin(), FE = cuda_module->end(); F != FE; F++)
                			F->setCallingConv(CallingConv::PTX_Device);
                        }

			// Initialize callback structure.
			// Initial lock state is "locked". It will be dropped
			// by GPU side monitor that must be started *before*
			// target GPU kernel.
			kernelgen_callback_t callback;
			callback.lock = 1;
			callback.state = KERNELGEN_STATE_INACTIVE;
			callback.kernel = NULL;
			callback.data = NULL;
			callback.szdata = szargs;
			callback.szdatai = szargsi;
			kernelgen_callback_t* callback_dev = NULL;
			err = cuMemAlloc((void**)&callback_dev, sizeof(kernelgen_callback_t));
			if (err) THROW("Error in cuMemAlloc " << err);
			err = cuMemcpyHtoD(callback_dev, &callback, sizeof(kernelgen_callback_t));
			if (err) THROW("Error in cuMemcpyHtoD " << err);
			kernel->target[runmode].callback = callback_dev;
			
			// Setup argerator structure and fill it with the main
			// entry arguments.
			char* args_host = (char*)malloc(szargs + sizeof(kernelgen_callback_t*));
			memcpy(args_host, args, szargs);
			memcpy(args_host + szargs, &callback_dev, sizeof(kernelgen_callback_t*));
			kernelgen_callback_data_t* args_dev = NULL;
			err = cuMemAlloc((void**)&args_dev, szargs + sizeof(kernelgen_callback_t*));
			if (err) THROW("Error in cuMemAlloc " << err);
			err = cuMemcpyHtoD(args_dev, &args_host, szargs + sizeof(kernelgen_callback_t*));
			if (err) THROW("Error in cuMemcpyHtoD " << err);
			int ret = kernelgen_launch(kernel, szargs, szargsi, args_dev);

			// Release device memory buffers.
			err = cuMemFree(callback_dev);
			if (err) THROW("Error in cuMemFree " << err);
			err = cuMemFree(args_dev);
			if (err) THROW("Error in cuMemFree " << err);

			delete kernelgen::runtime::cuda_context;

			return ret;
		}
		case KERNELGEN_RUNMODE_OPENCL :
		{
			THROW("Unsupported runmode" << runmode);
		}
		default :
			THROW("Unknown runmode " << runmode);
	}

	return 0;
}

