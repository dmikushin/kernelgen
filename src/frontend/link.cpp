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

		// Load data for found symbols names.
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
		manager.add(createLowerSetJmpPass());
		PassManagerBuilder builder;
		builder.Inliner = createFunctionInliningPass(numeric_limits<int>::max());
		builder.OptLevel = 3;
		builder.DisableSimplifyLibCalls = true;
		builder.populateModulePassManager(manager);
		manager.run(composite);
	}

	raw_ostream* Out = &dbgs();
	(*Out) << (composite);
	
	//
	// 4) Extract functions called over launcher to the separate
	// modules.
	//
	
	//
	// 5) Embed new modules to the resulting executable.
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

