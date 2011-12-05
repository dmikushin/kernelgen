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
#include "llvm/Support/IRReader.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/raw_ostream.h"

#include "elf.h"
#include "util.h"
#include "runtime.h"
#include "bind.h"

#include <iostream>
#include <cstdlib>

// Regular main entry.
extern "C" int __regular_main(int argc, char* argv[]);

using namespace kernelgen;
using namespace kernelgen::bind::cuda;
using namespace llvm;
using namespace std;
using namespace util::elf;

// GPU monitoring kernel source.
string cuda_monitor_kernel_source =
	"extern __attribute__((device)) int __iAtomicCAS(int *address, int compare, int val);"
	"\n"
	"__attribute__((global)) __attribute__((used)) void kernelgen_monitor(int* lock)\n"
	"{\n"
	"	// Unlock blocked gpu kernel associated\n"
	"	// with lock. It simply waits for lock\n"
	"	// to be dropped to zero.\n"
	"	__iAtomicCAS(lock, 1, 0);\n"
	"\n"
	"	// Wait for lock to be set.\n"
	"	// When lock is set this thread exits,\n"
	"	// and CPU monitor thread gets notified\n"
	"	// by synchronization.\n"
	"	while (!__iAtomicCAS(lock, 1, 1)) continue;\n"
	"}\n";

// Kernels runmode (target).
int kernelgen::runmode = -1;

// Verbose output.
bool kernelgen::verbose = false;

// The pool of already loaded kernels.
// After kernel is loaded, we pin it here
// for futher references.
std::map<string, kernel_t*> kernelgen::kernels;

vector<kernel_t> kernels_array;

int main(int argc, char* argv[])
{
	char* crunmode = getenv("kernelgen_runmode");
	if (crunmode)
	{	
		runmode = atoi(crunmode);

		// Check requested verbosity level.
		char* cverbose = getenv("kernelgen_verbose");
		if (cverbose) verbose = (bool)atoi(cverbose);
		
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
		celf e("/proc/self/exe", "");
		cregex regex("^__kernelgen_.*$", REG_EXTENDED | REG_NOSUB);
		vector<csymbol*> symbols = e.getSymtab()->find(regex);
		kernels_array.reserve(symbols.size());
		int ii = 0;
		for (vector<csymbol*>::iterator i = symbols.begin(),
			ie = symbols.end(); i != ie; i++, ii++)
		{
			csymbol* symbol = *i;
			const char* data = symbol->getData();
			const string& name = symbol->getName();
			kernel_t kernel;
			kernel.name = name;
			kernel.source = data;
			kernels_array.push_back(kernel);
			kernels[name] = &kernels_array.back();
			if (verbose) cout << name << endl;
		}
		if (verbose) cout << endl;

		// Walk through kernel index and replace
		// all names with kernel structure addresses
		// for each kernelgen_launch call.
		LLVMContext &context = getGlobalContext();
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
						if (!callee && !callee->isDeclaration()) continue;
						if (callee->getName() != "kernelgen_launch") continue;
						
						// Get the called function name.
						string name = "";
						GetElementPtrInst* namePtr = 
							dyn_cast<GetElementPtrInst>(call->getArgOperand(0));
						if (!namePtr)
							THROW("Cannot load GEP from kernelgen_launch argument");
						AllocaInst* nameAlloc =
							dyn_cast<AllocaInst>(namePtr->getPointerOperand());
						if (!nameAlloc)
							THROW("Cannot load AllocInst from kernelgen_launch argument");
						for (Value::use_iterator i = nameAlloc->use_begin(),
							ie = nameAlloc->use_end(); i != ie; i++)
						{
							StoreInst* nameInit = dyn_cast<StoreInst>(*i);
							if (nameInit)
							{
								ConstantArray* nameArray = dyn_cast<ConstantArray>(
									nameInit->getValueOperand());
								if (nameArray && nameArray->isCString())
									name = nameArray->getAsCString();
								nameInit->eraseFromParent();
							}
						}
						if (name == "")
							THROW("Cannot get the name of kernel invoked by kernelgen_launch");

						kernel_t* kernel = kernels["__kernelgen_" + name];
						call->setArgOperand(0, ConstantExpr::getIntToPtr(
							ConstantInt::get(Type::getInt64Ty(context), (uint64_t)kernel),
							Type::getInt8PtrTy(context)));

						// Delete occasional users, like lifetime.start/end.
						for (Value::use_iterator i = namePtr->use_begin(),
							ie = namePtr->use_end(); i != ie; i++)
						{
							Instruction* inst = dyn_cast<Instruction>(*i);
							if (inst) inst->eraseFromParent();
						}
						
						namePtr->eraseFromParent();
						nameAlloc->eraseFromParent();
					}

			kernel->source = "";
			raw_string_ostream ir(kernel->source);
			ir << (*m);
			
			//m->dump();
		}

		// Structure defining aggregate form of main entry
		// parameters. Return value is also packed, since
		// in CUDA and OpenCL kernels must return void.
		struct __attribute__((packed)) args_t
		{
			int64_t size;
			int argc;
			char** argv;
			int ret;
		};
		
		// Load arguments, depending on the target runmode
		// and invoke the entry point kernel.
		kernel_t* kernel = kernels["__kernelgen_main"];
		switch (runmode)
		{
			case KERNELGEN_RUNMODE_NATIVE :
			{
				args_t args;
				args.size = sizeof(int);
				args.argc = argc;
				args.argv = argv;
				kernelgen_launch((char*)kernel, (int*)&args);
				return args.ret;
			}
			case KERNELGEN_RUNMODE_CUDA :
			{
				kernelgen::bind::cuda::init();
				
				// Initialize thread locker variable.
				// Initial state is "locked". It will be dropped
				// by gpu side monitor that must be started *before*
				// target GPU kernel.
				void* lock = NULL;
				int err = cuMemAlloc(&lock, sizeof(int));
				if (err) THROW("Error in cuMemAlloc " << err);
				int one = 1;
				err = cuMemcpyHtoD(lock, &one, sizeof(int));
				if (err) THROW("Error in cuMemcpyHtoD " << err);
				kernel->target[runmode].monitor_lock = lock;

				// Create streams where monitoring and target kernels
				// will be executed.
				err = cuStreamCreate(
					&kernel->target[runmode].monitor_kernel_stream, 0);
				if (err) THROW("Error in cuStreamCreate " << err);
				err = cuStreamCreate(
					&kernel->target[runmode].kernel_stream, 0);
				if (err) THROW("Error in cuStreamCreate " << err);
				
				// Compile GPU monitoring kernel.
				kernel->target[runmode].monitor_kernel_func =
					kernelgen::runtime::nvopencc(cuda_monitor_kernel_source,
					"kernelgen_monitor");
	
				char** argv_dev = NULL;
				{
					size_t size = sizeof(char*) * argc;
					for (int i = 0; i < argc; i++)
						size += strlen(argv[i]) + 1;
					int err = cuMemAlloc((void**)&argv_dev, size);
					if (err) THROW("Error in cuMemAlloc " << err);
				}
				for (int i = 0, offset = 0; i < argc; i++)
				{
					char* arg = (char*)(argv_dev + argc) + offset;
					err = cuMemcpyHtoD(argv_dev + i, &arg, sizeof(char*));
					if (err) THROW("Error in cuMemcpyHtoD " << err);
					size_t length = strlen(argv[i]) + 1;
					err = cuMemcpyHtoD(arg, argv[i], length);
					if (err) THROW("Error in cuMemcpyDtoH " << err);
					offset += length;
				}
				args_t args_host;
				args_host.size = sizeof(int);
				args_host.argc = argc;
				args_host.argv = argv_dev;
				args_t* args_dev = NULL;
				err = cuMemAlloc((void**)&args_dev, sizeof(args_t));
				if (err) THROW("Error in cuMemAlloc " << err);
				err = cuMemcpyHtoD(args_dev, &args_host, sizeof(args_t));
				if (err) THROW("Error in cuMemcpyHtoD " << err);
				kernelgen_launch((char*)kernel, (int*)args_dev);
				err = cuMemcpyDtoH(&args_host.ret, &args_dev->ret, sizeof(int));
				if (err) THROW("Error in cuMemcpyDtoH " << err);
				err = cuMemFree(args_dev);
				if (err) THROW("Error in cuMemFree " << err);
				err = cuMemFree(argv_dev);
				if (err) THROW("Error in cuMemFree " << err);
				return args_host.ret;
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
	return __regular_main(argc, argv);
}

