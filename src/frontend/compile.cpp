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
#include <iostream>
#include <limits.h>
#include <string.h>

#include "llvm/Instructions.h"
#include "llvm/LLVMContext.h"
#include "llvm/Module.h"
#include "llvm/PassManager.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/IRReader.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/TypeBuilder.h"
#include "llvm/Transforms/IPO.h"
#include "llvm/Transforms/Scalar.h"

using namespace llvm;
using namespace std;

int compile(list<string> args, list<string> kgen_args,
	string input, string output, int verbose, int arch,
	string host_compiler)
{
	//
	// The LLVM compiler to emit IR.
	//
	const char* llvm_compiler = "kernelgen-gfortran";

	//
	// Linker used to merge multiple objects into single one.
	//
	string merge = "ld";
	list<string> merge_args;

	//
	// Temporary files location prefix.
	//
	string fileprefix = "/tmp/";

	//
	// Interpret kernelgen compile options.
	//
	for (list<string>::iterator it = kgen_args.begin(); it != kgen_args.end(); it++)
	{
		const char* arg = (*it).c_str();
		
		if (!strncmp(arg, "-Wk,--llvm-compiler=", 20))
			llvm_compiler = arg + 20;
		if (!strcmp(arg, "-Wk,--keep"))
			fileprefix = "";
	}

	if (arch == 32)
		merge_args.push_back("-melf_i386");
	merge_args.push_back("--unresolved-symbols=ignore-all");
	merge_args.push_back("-r");
	merge_args.push_back("-o");

	//
	// Generate temporary output file.
	// Check if output file is specified in the command line.
	// Replace or add output to the temporary file.
	//
	string bin_output;
	{
		char* c_bin_output = NULL;
		bin_output = fileprefix + "XXXXXX";
		c_bin_output = new char[bin_output.size() + 1];
		strcpy(c_bin_output, bin_output.c_str());
		int fd = mkstemp(c_bin_output);
		bin_output = c_bin_output;
		delete[] c_bin_output;
	}
	bool output_specified = false;
	for (list<string>::iterator it = args.begin(); it != args.end(); it++)
	{
		const char* arg = (*it).c_str();
		if (!strcmp(arg, "-o"))
		{
			it++;
			*it = bin_output;
			output_specified = true;
			break;
		}
	}
	if (!output_specified)
	{
		args.push_back("-o");
		args.push_back(bin_output);
	}

	//
	// 1) Compile source code using regular host compiler.
	//
	{
		if (verbose)
		{
			cout << host_compiler;
			for (list<string>::iterator it = args.begin();
				it != args.end(); it++)
				cout << " " << *it;
			cout << endl;
		}
		int status = execute(host_compiler, args, "", NULL, NULL);
		if (status) return status;
	}

	//
	// 2) Emit LLVM IR.
	//
	string out = "";
	{
		list<string> emit_ir_args;
		for (list<string>::iterator it = args.begin(); it != args.end(); it++)
		{
			const char* arg = (*it).c_str();
			if (!strcmp(arg, "-c") || !strcmp(arg, "-o"))
			{
				it++;
				continue;
			}
			emit_ir_args.push_back(*it);
		}
		emit_ir_args.push_back("-fplugin=/opt/kernelgen/lib/dragonegg.so");
		emit_ir_args.push_back("-fplugin-arg-dragonegg-emit-ir");
		emit_ir_args.push_back("-S");
		emit_ir_args.push_back(input);
		emit_ir_args.push_back("-o");
		emit_ir_args.push_back("-");
		if (verbose)
		{
			cout << llvm_compiler;
			for (list<string>::iterator it = emit_ir_args.begin();
				it != emit_ir_args.end(); it++)
				cout << " " << *it;
			cout << endl;
		}
		int status = execute(llvm_compiler, emit_ir_args, "", &out, NULL);
		if (status) return status;
	}

	//
	// 3) Record existing module functions.
	//
	LLVMContext &context = getGlobalContext();
	SMDiagnostic diag;
	MemoryBuffer* buffer1 = MemoryBuffer::getMemBuffer(out);
	Module* m1 = ParseIR(buffer1, diag, context);
	const Module::FunctionListType& funcList1 = m1->getFunctionList();
  	for (Module::const_iterator it = funcList1.begin();
		it != funcList1.end(); it++)
	{
		const Function &func = *it;
		if (!func.isDeclaration())
			printf("%s\n", func.getName().data());
	}

	//
	// 4) Inline calls and extract loops into new functions.
	//
	MemoryBuffer* buffer2 = MemoryBuffer::getMemBuffer(out);
	Module* m2 = ParseIR(buffer2, diag, context);
	{
		PassManager manager;
		manager.add(createInstructionCombiningPass());
		manager.add(createFunctionInliningPass(numeric_limits<int>::max()));
		manager.run(*m2);
	}
	{
		PassManager manager;
		manager.add(createLoopExtractorPass());
		manager.run(*m2);
	}

	//
	// 5) Replace call to loop functions with call to launcher.
	//
	Function* launch = Function::Create(
		TypeBuilder<int(const char*, int, ...), false>::get(context),
		GlobalValue::ExternalLinkage, "kernelgen_launch_", m2);
	Module::FunctionListType& funcList2 = m2->getFunctionList();
	for (Module::iterator it2 = funcList2.begin();
		it2 != funcList2.end(); it2++)
	{
		Function &func2 = *it2;
		if (func2.isDeclaration()) continue;

		// Search for the current function in original
		// module functions list.
		bool found = false;
		for (Module::const_iterator it1 = funcList1.begin();
			it1 != funcList1.end(); it1++)
		{
			const Function &func1 = *it1;
			if (func1.isDeclaration()) continue;

			if (func1.getName() == func2.getName())
			{
				found = true;
				break;
			}
		}

		// If function is not in list of original module,
		// then it is generated by the loop extractor.
		if (found) continue;

		// Each such function must be extracted to the
		// standalone module and packed into resulting
		// object file data section.
		printf("Preparing loop function %s ...\n", func2.getName().data());

		// Replace call to this function in module with call to launcher.
		found = false;
		for (Module::iterator F = m2->begin(); (F != m2->end()) && !found; F++)
			for (Function::iterator BB = F->begin(); (BB != F->end()) && !found; BB++)
				for (BasicBlock::iterator I = BB->begin(); I != BB->end(); I++)
				{
					CallInst* call = dyn_cast<CallInst>(cast<Value>(I));
					if (!call) continue;
					Function* callee = call->getCalledFunction();
					if (!callee && !callee->isDeclaration()) continue;
					if (callee->getName() != func2.getName()) continue;

					//ArrayRef<Value*> args;
					//CallInst* newcall = CallInst::Create(launch, args);
					//*call = *newcall;
					
					found = true;
					break;
				}
	}
	
	//
	// 6) Embed the resulting module into object file.
	//
	string ir_output;
	{
		string ir_string;
		raw_string_ostream ir(ir_string);
		ir << (*m2);
		ir_output = fileprefix + "XXXXXX";
		char* c_ir_output = new char[ir_output.size() + 1];
		strcpy(c_ir_output, ir_output.c_str());
		int fd = mkstemp(c_ir_output);
		ir_output = c_ir_output;
		delete[] c_ir_output;
		string ir_symname = "__kernelgen_" + string(input);
		util_elf_write(fd, arch, ir_symname.c_str(), ir_string.c_str(), ir_string.size());
	}
	
	//
	// 7) Merge object files with binary code and IR.
	//
	{
		merge_args.push_back(output);
		merge_args.push_back(bin_output);
		merge_args.push_back(ir_output);
		if (verbose)
		{
			cout << merge;
			for (list<string>::iterator it = merge_args.begin();
				it != merge_args.end(); it++)
				cout << " " << *it;
			cout << endl;
		}
		int status = execute(merge, merge_args, "", NULL, NULL);
		if (status) return status;
	}

	delete m1, m2, buffer1, buffer2;

	return 0;
}

