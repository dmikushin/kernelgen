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

#include "util.h"

#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <list>
#include <malloc.h>
#include <sstream>
#include <string.h>

// Construct the command line from the specified format
// and parameter list.
static int build_command(const char* fmt, char** cmd, ...)
{
	if (!cmd || !fmt) return 0;
	va_list list;
	va_start(list, cmd);
	int length = vsnprintf(NULL, 0, fmt, list);
	va_end(list);
	if (length < 0) return 1;
	*cmd = (char*)malloc(length + 1);
	va_start(list, cmd);
	vsprintf(*cmd, fmt, list);
	va_end(list);
	return 0;
}

#include "llvm/Instructions.h"
#include "llvm/LLVMContext.h"
#include "llvm/Module.h"
#include "llvm/PassManager.h"
#include "llvm/Support/CallSite.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/IRReader.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/TypeBuilder.h"
#include "llvm/Target/TargetData.h"
#include "llvm/Transforms/IPO.h"
#include "llvm/Transforms/Scalar.h"
#include "llvm/Transforms/Utils/Cloning.h"

using namespace llvm;
using namespace std;

int compile(const char* command, list<string> arguments)
{
	return 1;
}

int main(int argc, char* argv[])
{
	//
	// Behave like compiler if no arguments.
	//
	if (argc == 1)
	{
		cout << "kernelgen: no input files" << endl;
		return 0;
	}

	//
	// Enable or disable verbose output.
	//
	int verbose = 1;

	//
	// Switch to bypass the kernelgen pipe and use regular compiler only.
	//
	int bypass = 0;

	//
	// The regular compiler used for host-side source code.
	//
	const char* host_compiler = "gfortran";

	//
	// The LLVM compiler to emit IR.
	//
	const char* llvm_compiler = "kernelgen-gfortran";
	
	//
	// Target architecture.
	//
	int arch = 64;

	//
	// Supported source code files extensions.
	//
	list<string> source_ext;
	source_ext.push_back(".c");
	source_ext.push_back(".f");
	source_ext.push_back(".f90");
	source_ext.push_back(".F");
	source_ext.push_back(".F90");
	
	//
	// Temporary files location prefix.
	//
	string fileprefix = "/tmp/";

	//
	// Linker used to merge multiple objects into single one.
	//
	string merge = "ld";
	list<string> merge_args;

	//
	// Split kgen args from other args in the command line.
	//
	list<string> args, kgen_args;
	for (int i = 1; i < argc; i++)
	{
		char* arg = argv[i];
		if (!strncmp(arg, "-Wk,", 4))
			kgen_args.push_back(arg);
		else
			args.push_back(arg);

		// In case of 32-bit compilation on 64-bit,
		// invoke object mergering command with 32-bit flag.		
		if (!strcmp(arg, "-m32"))
		{
			arch = 32;
			merge_args.push_back("-melf_i386");
		}
	}
	merge_args.push_back("--unresolved-symbols=ignore-all");
	merge_args.push_back("-r");
	merge_args.push_back("-o");

	//
	// Interpret kgen args.
	//
	for (list<string>::iterator it = kgen_args.begin(); it != kgen_args.end(); it++)
	{
		const char* arg = (*it).c_str();
		
		if (!strcmp(arg, "-Wk,--bypass"))
			bypass = 1;
		if (!strncmp(arg, "-Wk,--llvm-compiler=", 20))
			llvm_compiler = arg + 20;
		if (!strncmp(arg, "-Wk,--host-compiler=", 20))
			host_compiler = arg + 20;
		if (!strncmp(arg, "-Wk,--kernel-target=", 20))
		{
			const char* targets = arg + 20;
			/*foreach $target (@targets)
			{
				if (($target ne "cpu") and ($target ne "cuda") and ($target ne "opencl"))
				{
					print STDERR BOLD, RED, "Unknown target $target\n", RESET;
					print RESET, "";
					exit 1;
				}
			}*/
		}
		if (!strcmp(arg, "-Wk,--keep"))
			fileprefix = "";
		if (!strcmp(arg, "-Wk,--verbose"))
			verbose = 1;
	}

	//
	// Find source code input.
	// FIXME There could be multiple source files supplied,
	// currently this case is unhandled.
	//
	const char* input = NULL;
	for (list<string>::iterator it1 = args.begin(); (it1 != args.end()) && !input; it1++)
	{
		const char* arg = (*it1).c_str();
		for (list<string>::iterator it2 = source_ext.begin(); it2 != source_ext.end(); it2++)
		{
			const char* ext = (*it2).c_str();
			if (!strcmp(arg + strlen(arg) - strlen(ext), ext))
			{
				input = arg;
				break;
			}
		}
	}

	//
	// Find output file in args.
	// There could be "-c" or "-o" option or both.
	// With "-c" source is compiled only, producing by default
	// an object file with same basename as source.
	// With "-o" source could either compiled only (with additional
	// "-c") or fully linked, but in both cases output is sent to
	// explicitly defined file after "-o" option.
	//
	char* c_bin_output = NULL;
	{
		string bin_output = fileprefix + "XXXXXX";
		c_bin_output = new char[bin_output.size() + 1];
		strcpy(c_bin_output, bin_output.c_str());
		int fd = mkstemp(c_bin_output);
	}
	char* output = NULL;
	for (list<string>::iterator it = args.begin(); it != args.end(); it++)
	{
		const char* arg = (*it).c_str();
		if (!strcmp(arg, "-o"))
		{
			it++;
			arg = (*it).c_str();

			output = new char[strlen(arg) + 1];
			strcpy(output, arg);
			
			// Replace output with temporary output.
			if (input) *it = c_bin_output;
			break;
		}
	}
	if (input && !output)
	{
		args.push_back("-o");
		args.push_back(c_bin_output);
	}
	for (list<string>::iterator it = args.begin(); (it != args.end()) && !output; it++)
	{
		const char* arg = (*it).c_str();
		if (!strcmp(arg, "-c"))
		{
			it++;
			arg = (*it).c_str();
		
			// Trim path.
			output = new char[strlen(arg) + 1];
			strcpy(output, arg);
			for (int i = strlen(output); i >= 0; i--)
			{
				if (output[i] == '/')
				{
					memcpy(output, arg, i);
					output[i] = '\0';
					break;
				}
			}

			// Replace source extension with object extension.
			for (int i = strlen(output); i >= 0; i--)
			{
				if (output[i] == '.')
				{
					output[i + 1] = 'o';
					output[i + 2] = '\0';
					break;
				}
			}
		}
	}

	//
	// Only execute the regular host compiler, if required or
	// do only regular compilation for file extensions
	// we do not know. Also should cover the case of linking.
	//
	if (bypass)
		return execute(host_compiler, args, "", NULL, NULL);
	else
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

	if (!input) return 0;

	//
	// 1) Emit LLVM IR.
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
		if (status) return 1;
	}

	//
	// 2) Record existing module functions.
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
	// 3) Inline calls and extract loops into new functions.
	//
	MemoryBuffer* buffer2 = MemoryBuffer::getMemBuffer(out);
	Module* m2 = ParseIR(buffer2, diag, context);
	{
		PassManager manager;
		manager.add(createInstructionCombiningPass());
		manager.add(createFunctionInliningPass());
		manager.run(*m2);
	}
	{
		PassManager manager;
		manager.add(createLoopExtractorPass());
		manager.run(*m2);
	}

	//
	// 4) Replace call to loop functions with call to launcher.
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
	// 4) Embed the resulting module into object file.
	//
	char* c_ir_output = NULL;
	{
		string ir_string;
		raw_string_ostream ir(ir_string);
		ir << (*m2);
		string ir_output = fileprefix + "XXXXXX";
		c_ir_output = new char[ir_output.size() + 1];
		strcpy(c_ir_output, ir_output.c_str());
		int fd = mkstemp(c_ir_output);
		string ir_symname = "__kernelgen_" + string(input);
		util_elf_write(fd, arch, ir_symname.c_str(), ir_string.c_str(), ir_string.size());
	}
	
	//
	// 5) Merge object files with binary code and IR.
	//
	{
		merge_args.push_back(output);
		merge_args.push_back(c_bin_output);
		merge_args.push_back(c_ir_output);
		if (verbose)
		{
			cout << merge;
			for (list<string>::iterator it = merge_args.begin();
				it != merge_args.end(); it++)
				cout << " " << *it;
			cout << endl;
		}
		execute(merge, merge_args, "", NULL, NULL);
	}
	delete[] c_bin_output, c_ir_output, output;

	//raw_ostream* Out = &dbgs();
	//(*Out) << (*m2);

	delete m1, m2, buffer1, buffer2;

	return 0;
}

