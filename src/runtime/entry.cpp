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

#include <iostream>
#include <cstdlib>

// Regular main entry.
extern "C" int __regular_main(int argc, char* argv[]);

using namespace kernelgen;
using namespace llvm;
using namespace std;
using namespace util::elf;

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
		//celf e("/proc/self/exe", "");
		celf e("/home/marcusmae/Programming/kernelgen/branches/accurate/tests/behavior/sincos/64/sincos", "");
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
						
						// OK, it's a call to kernelgen_launch, get its first argument.
						ConstantExpr* ce =
							dyn_cast<ConstantExpr>(call->getArgOperand(0));
						if (!ce)
							THROW("Expected contant value in kernelgen_launch first operand");
						GlobalVariable* gvar =
							dyn_cast<GlobalVariable>(ce->getOperand(0));
						if (!gvar)
							THROW("Expected global variable in kernelgen_launch first operand");
						ConstantArray* ca =
							dyn_cast<ConstantArray>(gvar->getInitializer());
						if (!ca || !ca->isCString())
							THROW("Expected constant array in kernelgen_launch first operand");
				
						string name = ca->getAsCString();
						kernel_t* kernel = kernels["__kernelgen_" + name];
						call->setArgOperand(0, ConstantExpr::getIntToPtr(
							ConstantInt::get(Type::getInt64Ty(context), (uint64_t)kernel),
							Type::getInt8PtrTy(context)));
					}

			kernel->source = "";
			raw_string_ostream ir(kernel->source);
			ir << (*m);
			
			//m->dump();
		}

		// Invoke entry point kernel.
		int szargs[2] = { sizeof(int), sizeof(char**) };
		kernel_t* kernel = kernels["__kernelgen_main"];
		struct __attribute__((packed)) args_t
		{
			int64_t* size;
			int argc;
			char** argv;
		}
		args;
		int64_t size = sizeof(int);
		args.size = &size;
		args.argc = argc;
		args.argv = argv;
		return kernelgen_launch((char*)kernel, (int*)&args);
	}
	
	// Chain to entry point of the regular binary.
	return __regular_main(argc, argv);
}

