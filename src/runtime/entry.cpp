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

//#include "TrackedPassManager.h"

// Regular main entry.
extern "C" int __regular_main(int argc, char* argv[]);

using namespace kernelgen;
using namespace kernelgen::runtime;
using namespace kernelgen::bind::cuda;
using namespace llvm;
using namespace llvm::sys;
using namespace llvm::sys::fs;
using namespace std;
using namespace util::elf;

// Kernels runmode (target).
int kernelgen::runmode = -1;

// Verbose output.
int kernelgen::verbose = 0;

// Debug mode.
bool kernelgen::debug = true;

// The pool of already loaded kernels.
// After kernel is loaded, we pin it here
// for futher references.
std::map<string, kernel_t*> kernelgen::kernels;

// The array contains addresses of globalVatiables
uint64_t *kernelgen::addressesOfGlobalVariables;
int kernelgen::numberOfGlobalVariables;

// CUDA runtime context.
// TODO: sort out how to turn it into auto_ptr.
kernelgen::bind::cuda::context* kernelgen::runtime::cuda_context = NULL;

// Monitoring module and kernel (applicable for some targets).
Module* kernelgen::runtime::monitor_module = NULL;
kernel_func_t kernelgen::runtime::monitor_kernel;

// Runtime module (applicable for some targets).
Module* kernelgen::runtime::runtime_module = NULL;

// CUDA module (applicable for some targets).
Module* kernelgen::runtime::cuda_module = NULL;

int main(int argc, char* argv[], char* envp[])
{
	//tracker = new PassTracker("codegen", NULL, NULL);

	LLVMContext& context = getGlobalContext();

	// Retrieve the regular main entry function prototype out of
	// the internal table.
	Function* regular_main = NULL;
	celf e("/proc/self/exe", "");
	//celf e("/RHM/users/work/dmikushin/forge/kernelgen/tests/behavior/hello_c/hello_c", "");
	cregex regex("^__kernelgen_main$", REG_EXTENDED | REG_NOSUB);
	vector<csymbol*> symbols = e.getSymtab()->find(regex);
	if (!symbols.size())
	{
		THROW("Cannot find the __kernelgen_main symbol");
	}
	else
	{
		// Load the regular main function prototype.
		// It is needed to correctly select the argument list.
		MemoryBuffer* buffer = MemoryBuffer::getMemBuffer(symbols[0]->getData());
		SMDiagnostic diag;
		Module* m = ParseIR(buffer, diag, context);
		if (!m)
			THROW("__kernelgen_regular_main:" << diag.getLineNo() << ": " <<
				diag.getLineContents() << ": " << diag.getMessage());
		regular_main = m->getFunction("__kernelgen_regular_main");
		if (!regular_main)
			THROW("Cannot find the __kernelgen_regular_main function");
	}

	FunctionType* mainFuncTy = regular_main->getFunctionType();
	Type* mainRetTy = regular_main->getReturnType();

	// Structure defining aggregate form of the main entry
	// parameters. Return value is also packed, since
	// in CUDA and OpenCL kernels must return void.
	// Also structure aggregates the callback record containing
	// parameters of host-device communication state and the
	// head of memory pool.
	// Note the prototype of main function may vary, and in
	// structure we define the maximum possible parameter list.
	struct main_args_t
	{
		uint64_t* addressesOfGlobalVariables;
		kernelgen_callback_t* callback;
		kernelgen_memory_t* memory;
		int argc;
		char** argv;
		char** envp;
		int ret;
	};

	char* crunmode = getenv("kernelgen_runmode");
	if (crunmode)
	{	
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
#ifdef KERNELGEN_LOAD_KERNELS_LAZILY
			kernel->loaded = false;
#endif
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

		// Check whether the internal table contains a main entry.
		kernel_t* kernel = kernels["__kernelgen_main"];
		if (!kernel)
		{
			THROW("Cannot find the __kernelgen_main symbol");
		}
#ifndef KERNELGEN_LOAD_KERNELS_LAZILY
		// Walk through kernel index and replace
		// all names with kernel structure addresses
		// for each kernelgen_launch call.
		SMDiagnostic diag;
		for (map<string, kernel_t*>::iterator i = kernels.begin(),
			e = kernels.end(); i != e; i++)
		{
			kernel_t* kernel = (*i).second;
			
			if (!kernel) THROW("Invalid kernel item");
			
			// Load IR from source.
			MemoryBuffer* buffer = MemoryBuffer::getMemBuffer(kernel->source);
			Module* m = ParseIR(buffer, diag, context);
			if (!m)
				THROW(kernel->name << ":" << diag.getLineNo() << ": " <<
					diag.getLineContents() << ": " << diag.getMessage());
			m->setModuleIdentifier(kernel->name + "_module");
		
			for (Module::iterator fi = m->begin(), fe = m->end(); fi != fe; fi++)
				for (Function::iterator bi = fi->begin(), be = fi->end(); bi != be; bi++)
					for (BasicBlock::iterator ii = bi->begin(), ie = bi->end(); ii != ie; ii++)
					{
						// Check if instruction in focus is a call.
						CallInst* call = dyn_cast<CallInst>(cast<Value>(ii));
						if (!call) continue;
					
						// Check if function is called (needs -instcombine pass).
						Function* callee = call->getCalledFunction();
						if (!callee) continue;
						if (!callee->isDeclaration()) continue;
						if (callee->getName() != "kernelgen_launch") continue;

						// Get the called function name from the metadata node.
						MDNode* nameMD = call->getMetadata("kernelgen_launch");
						if (!nameMD)
							THROW("Cannot find kernelgen_launch metadata");
						if (nameMD->getNumOperands() != 1)
							THROW("Unexpected kernelgen_launch metadata number of operands");
						ConstantDataArray* nameArray = dyn_cast<ConstantDataArray>(
							nameMD->getOperand(0));
						if (!nameArray)
							THROW("Invalid kernelgen_launch metadata operand");
						if (!nameArray->isCString())
							THROW("Invalid kernelgen_launch metadata operand");
						string name = "__kernelgen_" + (string)nameArray->getAsCString();
						if (verbose)
							cout << "Launcher invokes kernel " << name << endl;
						
						// Permanently assign launcher first argument with the address
						// of the called kernel function structure (for fast access).
						kernel_t* kernel = kernels[name];
						if (!kernel)
							THROW("Cannot get the name of kernel invoked by kernelgen_launch");
						call->setArgOperand(0, ConstantExpr::getIntToPtr(
							ConstantInt::get(Type::getInt64Ty(context), (uint64_t)kernel),
							Type::getInt8PtrTy(context)));
					}

			kernel->source = "";
			raw_string_ostream ir(kernel->source);
			ir << (*m);
			
			//m->dump();
		}
#endif
		kernel = kernels["__kernelgen_main"];
        if (!kernel->module) {
		// Load LLVM IR source into module.
		SMDiagnostic diag;
		MemoryBuffer* buffer = MemoryBuffer::getMemBuffer(kernel->source);
		kernel->module = ParseIR(buffer, diag, context);
		if (!kernel->module)
			THROW(kernel->name << ":" << diag.getLineNo() << ": " <<
			      diag.getLineContents() << ": " << diag.getMessage());
		kernel->module->setModuleIdentifier(kernel->name + "_module");
	}
		
		assert(kernel->module && "main module  must be loaded!");
		assert(sizeof(void *) == sizeof(uint64_t));

		numberOfGlobalVariables = kernel->module->getGlobalList().size();
        addressesOfGlobalVariables=NULL;
		
		// Load arguments, depending on the target runmode
		// and invoke the entry point kernel.
		switch (runmode)
		{
			case KERNELGEN_RUNMODE_NATIVE :
			{
				addressesOfGlobalVariables =(uint64_t *)(calloc(numberOfGlobalVariables, sizeof(void*)));
				main_args_t args;
				args.addressesOfGlobalVariables = addressesOfGlobalVariables;
				args.argc = argc;
				args.argv = argv;
				args.envp = envp;
				kernelgen_launch(kernel, sizeof(main_args_t),
					sizeof(int), (kernelgen_callback_data_t*)&args);
				return args.ret;
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

					kernel_t kernel;
					kernel.name = "kernelgen_monitor";
					kernel.target[KERNELGEN_RUNMODE_CUDA].monitor_kernel_stream = NULL;
					monitor_kernel = kernelgen::runtime::codegen(
						KERNELGEN_RUNMODE_CUDA, &kernel, monitor_module);
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
				callback.szdata = sizeof(main_args_t);
				callback.szdatai = sizeof(int);
				kernelgen_callback_t* callback_dev = NULL;
				err = cuMemAlloc((void**)&callback_dev, sizeof(kernelgen_callback_t));
				if (err) THROW("Error in cuMemAlloc " << err);
				err = cuMemcpyHtoD(callback_dev, &callback, sizeof(kernelgen_callback_t));
				if (err) THROW("Error in cuMemcpyHtoD " << err);
				kernel->target[runmode].callback = callback_dev;
				
				// Setup device dynamic memory heap.
				int szheap = 16 * 1024 * 1024;
		                char* cszheap = getenv("kernelgen_szheap");
				if (cszheap) szheap = atoi(cszheap);
				kernelgen_memory_t* memory = init_memory_pool(szheap);

				// Duplicate argv into device memory.
				// Note in addition to actiual arguments we must pass pass
				// argv[argc] = NULL.
				char** argv_dev = NULL;
				if (mainFuncTy->getNumParams() > 1)
				{
					{
						size_t size = sizeof(char*) * (argc + 1);
						for (int i = 0; i < argc; i++)
							size += strlen(argv[i]) + 1;
						int err = cuMemAlloc((void**)&argv_dev, size);
						if (err) THROW("Error in cuMemAlloc " << err);
					}
					for (int i = 0, offset = 0; i < argc; i++)
					{
						char* arg = (char*)(argv_dev + argc + 1) + offset;
						err = cuMemcpyHtoD(argv_dev + i, &arg, sizeof(char*));
						if (err) THROW("Error in cuMemcpyHtoD " << err);
						size_t length = strlen(argv[i]) + 1;
						err = cuMemcpyHtoD(arg, argv[i], length);
						if (err) THROW("Error in cuMemcpyDtoH " << err);
						offset += length;
					}
					err = cuMemsetD8(argv_dev + argc, 0, sizeof(char*));
					if (err) THROW("Error in cuMemsetD8 " << err);
				}

				// Duplicate envp into device memory.
				// Note in addition to actual arguments we must pass
				// envp[argc] = NULL.
				char** envp_dev = NULL;
				if (mainFuncTy->getNumParams() > 2)
				{
					int envc = 0;
					{
						while (envp[envc]) envc++;
						size_t size = sizeof(char*) * (envc + 1);
						for (int i = 0; i < envc; i++)
							size += strlen(envp[i]) + 1;
						int err = cuMemAlloc((void**)&envp_dev, size);
						if (err) THROW("Error in cuMemAlloc " << err);
					}
					for (int i = 0, offset = 0; i < envc; i++)
					{
						char* env = (char*)(envp_dev + envc + 1) + offset;
						err = cuMemcpyHtoD(envp_dev + i, &env, sizeof(char*));
						if (err) THROW("Error in cuMemcpyHtoD " << err);
						size_t length = strlen(envp[i]) + 1;
						err = cuMemcpyHtoD(env, envp[i], length);
						if (err) THROW("Error in cuMemcpyDtoH " << err);
						offset += length;
					}
					err = cuMemsetD8(envp_dev + envc, 0, sizeof(char*));
					if (err) THROW("Error in cuMemsetD8 " << err);
				}
                
                // page locks memory for globals addresses
				{
					
					CUresult err = cuMemAllocHost((void **)&addressesOfGlobalVariables,numberOfGlobalVariables*sizeof(void*) );
				    if (err) THROW("Error in cuMemAllocHost " << err);
					CUdeviceptr pointerOnDevice = (CUdeviceptr)addressesOfGlobalVariables;
                    err = cuMemsetD8 (pointerOnDevice, 0, numberOfGlobalVariables*sizeof(void*));
					if (err) THROW("Error in cuMemsetD8 " << err);
                    
					/*#define CU_MEMHOSTREGISTER_DEVICEMAP   0x02
                	CUresult err = cuMemHostRegister(addressesOfGlobalVariables, numberOfGlobalVariables*sizeof(void*),CU_MEMHOSTREGISTER_DEVICEMAP);
				    if (err) THROW("Error in cuMemHostRegister " << err);
					err = cuMemHostGetDevicePointer(&pointerOnDevice,addressesOfGlobalVariables,0);
					if (err) THROW("Error in cuMemHostGetDevicePointer " << err);
					assert(pointerOnDevice);
					//addressesOfGlobalVariables = (uint64_t *)pointerOnDevice;*/
				}
				
				// Setup argerator structure and fill it with the main
				// entry arguments.
				main_args_t args_host;
				args_host.addressesOfGlobalVariables = addressesOfGlobalVariables;
				args_host.argc = argc;
				args_host.argv = argv_dev;
				args_host.callback = callback_dev;
				args_host.memory = memory;
				main_args_t* args_dev = NULL;
				err = cuMemAlloc((void**)&args_dev, sizeof(main_args_t));
				if (err) THROW("Error in cuMemAlloc " << err);
				err = cuMemcpyHtoD(args_dev, &args_host, sizeof(main_args_t));
				if (err) THROW("Error in cuMemcpyHtoD " << err);
				kernelgen_launch(kernel, sizeof(main_args_t), sizeof(int),
					(kernelgen_callback_data_t*)args_dev);

				// Store back to host the return value, if present.
				int ret = EXIT_SUCCESS;
				if (!mainRetTy->isVoidTy())
				{
					err = cuMemcpyDtoH(&ret, &args_dev->ret, sizeof(int));
					if (err) THROW("Error in cuMemcpyDtoH " << err);
				}

				// Release device memory buffers.
				if (argv_dev)
				{
					err = cuMemFree(argv_dev);
					if (err) THROW("Error in cuMemFree " << err);
				}
				if (envp_dev)
				{
					err = cuMemFree(envp_dev);
					if (err) THROW("Error in cuMemFree " << err);
				}
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
	}
	
	// Chain to entry point of the regular binary.
	// Since the main entry prototype may vary, it is
	// invoked over FFI using the previously discovered
	// actual argument list.

	ffi_type* rtype = &ffi_type_sint32;
	if (mainRetTy->isVoidTy()) rtype = &ffi_type_void;

	ffi_type* argsTy[] = {
			&ffi_type_sint32,
			&ffi_type_pointer,
			&ffi_type_pointer
	};

	ffi_cif cif;
	if (ffi_prep_cif(&cif, FFI_DEFAULT_ABI,
		mainFuncTy->getNumParams(), rtype, argsTy) != FFI_OK)
		THROW("Error in ffi_prep_cif");

	main_args_t args;
	args.argc = argc;
	args.argv = argv;
	args.envp = envp;

	void* pargs[] = {
			&args.argc,
			&args.argv,
			&args.envp
	};

	int ret;
	typedef void (*func_t)();
	func_t func = (func_t)__regular_main;
	ffi_call(&cif, func, &ret, (void**)&pargs);

	return ret;
}

