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

#include "kernelgen.h"
#include "util.h"

#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <fcntl.h>
#include <iostream>
#include <limits.h>
#include <string.h>
#include <unistd.h>

#include "llvm/ADT/StringMap.h"
#include "llvm/ValueSymbolTable.h"
#include "llvm/Constants.h"
#include "llvm/Instructions.h"
#include "llvm/Linker.h"
#include "llvm/LLVMContext.h"
#include "llvm/Module.h"
#include "llvm/PassManager.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/IRReader.h"
#include "llvm/Support/PassManagerBuilder.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/TypeBuilder.h"
#include "llvm/Target/TargetData.h"
#include "llvm/Transforms/IPO.h"
#include "llvm/Transforms/Scalar.h"

using namespace llvm;
using namespace std;

int link(list<string> args, list<string> kgen_args,
	string input, string output, int verbose, int arch,
	string host_compiler)
{
	//
	// 1) Check if there is "-c" option around. In this
	// case there is just compilation, not linking, but
	// in a way we do not know how to handle.
	//
	for (list<string>::iterator it = args.begin(); it != args.end(); it++)
	{
		const char* arg = (*it).c_str();
		if (!strcmp(arg, "-c"))
			return execute(host_compiler, args, "", NULL, NULL);
	}

	//
	// 2) Extract all object files out of the command line.
	// From each object file extract LLVM IR modules and merge
	// them into single composite module.
	//
	LLVMContext &context = getGlobalContext();
	SMDiagnostic diag;
	Module composite("composite", context);
	for (list<string>::iterator it1 = args.begin(); it1 != args.end(); it1++)
	{
		const char* arg = (*it1).c_str();
		if (strcmp(arg + strlen(arg) - 2, ".o"))
			continue;
		
		if (verbose)
			cout << "Linking " << arg << " ..." << endl;

		char** symnames = NULL;
		char** symdatas = NULL;
		size_t* symsizes = NULL;

		// Load all symbols names starting with __kernelgen_.
		int count = 0;
		int fd = open(arg, O_RDONLY);
		int status = util_elf_find(fd, "^__kernelgen_.*$", &symnames, &count);
		if (status) goto finish;

		// Load data for discovered symbols names.
		symdatas = (char**)malloc(sizeof(char*) * count);
		symsizes = (size_t*)malloc(sizeof(size_t) * count);
		status = util_elf_read_many(fd, count, 
			(const char**)symnames, symdatas, symsizes);
		if (status) goto finish;
		
		// For each symbol name containing module, link into
		// the composite module.
		for (int i = 0; i < count; i++)
		{
			MemoryBuffer* buffer =
				MemoryBuffer::getMemBuffer(symdatas[i]);
			Module* m = ParseIR(buffer, diag, context);

			if (!m)
			{
				cerr << "Error loading module " << symnames[i] << endl;
				delete buffer;
				status = 1;
				goto finish;
			}
		
			if (verbose)
				cout << "Linking " << symnames[i] << endl;
		
			string err;
			if (Linker::LinkModules(&composite, m, &err))
			{
				cerr << "Error linking module " << symnames[i] <<
					" : " << err << endl;
				delete buffer, m;
				status = 1;
				goto finish;
			}
			
			delete m;
		}
finish:
		if (fd >= 0) close(fd);

		if (symnames) free(symnames);
		if (symdatas) free(symdatas);
		if (symsizes) free(symsizes);
		if (status) return status;
	}
	
	//
	// 3) Apply optimization passes to the resulting common
	// module.
	//
	{
		PassManager manager;
		manager.add(new TargetData(&composite));
		manager.add(createLowerSetJmpPass());
		PassManagerBuilder builder;
		builder.Inliner = createFunctionInliningPass(2000);
		builder.OptLevel = 3;
		builder.DisableSimplifyLibCalls = true;
		builder.populateModulePassManager(manager);
		manager.run(composite);
	}

	//raw_ostream* Out = &dbgs();
	//(*Out) << (composite);

	ValueSymbolTable& VST = composite.getValueSymbolTable();
	for (ValueSymbolTable::const_iterator I = VST.begin(), E = VST.end(); I != E; ++I)
		cout << I->getKeyData() << endl;

	//
	// 4) Extract functions called over launcher to the separate
	// modules.
	//
	std::vector<GlobalValue*> kernels;
	for (Module::iterator F = composite.begin(); F != composite.end(); F++)
		for (Function::iterator BB = F->begin(); BB != F->end(); BB++)
			for (BasicBlock::iterator I = BB->begin(); I != BB->end(); I++)
			{
				// Check if instruction in focus is a call.
				CallInst* call = dyn_cast<CallInst>(cast<Value>(I));
				if (!call) continue;
				
				// Check if function is called (needs -instcombine pass).
				Function* callee = call->getCalledFunction();
				if (!callee && !callee->isDeclaration()) continue;
				if (callee->getName() != "kernelgen_launch") continue;
				
				// Get value of the function pointer argument.
				StringRef name;
				const ConstantExpr* ce;
				GlobalValue* gval = NULL;
				const GlobalVariable* gvar;
				const ConstantArray* ca;
				if (!call->getNumArgOperands()) continue;
				ce = dyn_cast<ConstantExpr>(call->getArgOperand(0));
				if (!ce) goto failure;
				gvar = dyn_cast<GlobalVariable>(ce->getOperand(0));
				if (!gvar) goto failure;
				ca = dyn_cast<ConstantArray>(gvar->getInitializer());
				if (!ca || !ca->isCString()) goto failure;
				
				name = ca->getAsCString();
				if (verbose)
					cout << "Launcher invokes kernel " << name.data() << endl;

				gval = composite.getFunction(name);
				if (!gval)
				{
					cerr << "Cannot find function " << name.data() << endl;
					continue;
				}
				kernels.push_back(gval);
				continue;
			failure:
				cerr << "Cannot get the name of kernel invoked by kernelgen_launch" << endl;
			}
	
	//
	// 5) Remove all functions called through launcher.
	//
	{
		PassManager manager;
		manager.add(new TargetData(&composite));
		
		// Delete functions called through launcher.
		manager.add(createGVExtractionPass(kernels, true));
		
		// Remove dead debug info.
		manager.add(createStripDeadDebugInfoPass());
		
		// Remove dead func decls.
		manager.add(createStripDeadPrototypesPass());

		manager.run(composite);
	}

	composite.dump();

	//
	// 5) Embed new modules to the resulting executable.
	//

	//
	// 6) Link code using regular linker, but specifying alternate
	// entry point to switch between regular code and LLVM.
	//
	{
		list<string> args_ext = args;
		//args_ext.push_back("-LKGEN_PREFIX/lib");
		//args_ext.push_back("-LKGEN_PREFIX/lib64");
		//args_ext.push_back("-lkernelgen");
		//args_ext.push_back("-lstdc++");
		if (verbose)
		{
			cout << host_compiler;
			for (list<string>::iterator it = args_ext.begin();
				it != args_ext.end(); it++)
				cout << " " << *it;
			cout << endl;
		}
		int status = execute(host_compiler, args_ext, "", NULL, NULL);
		if (status) return status;
	}
	
	return 0;
}

