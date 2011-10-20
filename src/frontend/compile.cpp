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
#include "util/util.h"

#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <limits.h>
#include <string.h>

#include "llvm/Constants.h"
#include "llvm/Instructions.h"
#include "llvm/LLVMContext.h"
#include "llvm/Module.h"
#include "llvm/PassManager.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/IRReader.h"
#include "llvm/Support/Host.h"
#include "llvm/Support/PassManagerBuilder.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/TypeBuilder.h"
#include "llvm/Target/TargetData.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetRegistry.h"
#include "llvm/Target/TargetSelect.h"
#include "llvm/Transforms/IPO.h"
#include "llvm/Transforms/Scalar.h"

using namespace llvm;
using namespace std;

int compile(list<string> args, list<string> kgen_args,
	string merge, list<string> merge_args,
	string input, string output, int verbose, int arch,
	string host_compiler, string fileprefix)
{
	//
	// The LLVM compiler to emit IR.
	//
	const char* llvm_compiler = "kernelgen-gfortran";

	//
	// Interpret kernelgen compile options.
	//
	for (list<string>::iterator iarg = kgen_args.begin(),
		iearg = kgen_args.end(); iarg != iearg; iarg++)
	{
		const char* arg = (*iarg).c_str();		
		if (!strncmp(arg, "-Wk,--llvm-compiler=", 20))
			llvm_compiler = arg + 20;
	}

	//
	// Generate temporary output file.
	// Check if output file is specified in the command line.
	// Replace or add output to the temporary file.
	//
	int bin_fd = -1;
	string bin_output;
	{
		char* c_bin_output = NULL;
		bin_output = fileprefix + "XXXXXX";
		c_bin_output = new char[bin_output.size() + 1];
		strcpy(c_bin_output, bin_output.c_str());
		bin_fd = mkstemp(c_bin_output);
		bin_output = c_bin_output;
		delete[] c_bin_output;
	}
	bool output_specified = false;
	for (list<string>::iterator iarg = args.begin(),
		iearg = args.end(); iarg != iearg; iarg++)
	{
		const char* arg = (*iarg).c_str();
		if (!strcmp(arg, "-o"))
		{
			iarg++;
			*iarg = bin_output;
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
			for (list<string>::iterator iarg = args.begin(),
				iearg = args.end(); iarg != iearg; iarg++)
				cout << " " << *iarg;
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
		for (list<string>::iterator iarg = args.begin(),
			iearg = args.end(); iarg != iearg; iarg++)
		{
			const char* arg = (*iarg).c_str();
			if (!strcmp(arg, "-c") || !strcmp(arg, "-o"))
			{
				iarg++;
				continue;
			}
			emit_ir_args.push_back(*iarg);
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
			for (list<string>::iterator iarg = emit_ir_args.begin(),
				iearg = emit_ir_args.end(); iarg != iearg; iarg++)
				cout << " " << *iarg;
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

	//
	// 4) Inline calls and extract loops into new functions.
	//
	MemoryBuffer* buffer2 = MemoryBuffer::getMemBuffer(out);
	Module* m2 = ParseIR(buffer2, diag, context);
	{
		PassManager manager;
		manager.add(createInstructionCombiningPass());
		manager.run(*m2);
	}
	{
		PassManager manager;
		manager.add(createLoopExtractorPass());
		manager.run(*m2);
	}

	//
	// 5) Create target machine and get its target data.
	//
	Triple triple(m2->getTargetTriple());
	if (triple.getTriple().empty())
		triple.setTriple(sys::getHostTriple());
	string err;
	InitializeAllTargets();
	const Target* target = TargetRegistry::lookupTarget(triple.getTriple(), err);
	if (!target)
	{
		cerr << "Error auto-selecting target for module '" << err << "'." << endl;
		cerr << "Please use the -march option to explicitly pick a target.\n" << endl;
		return 1;
	}
	TargetMachine* machine = target->createTargetMachine(
		triple.getTriple(), "", "", Reloc::Default, CodeModel::Default);
	if (!machine)
	{
		cerr << "Could not allocate target machine" << endl;
		return 1;
	}
	const TargetData* tdata = machine->getTargetData();

	//
	// 6) Replace call to loop functions with call to launcher.
	// Append "always inline" attribute to all other functions.
	//
	Function* launch = Function::Create(
		TypeBuilder<void(types::i<8>*, types::i<32>, types::i<32>*, ...), true>::get(context),
		GlobalValue::ExternalLinkage, "kernelgen_launch", m2);
	for (Module::iterator f1 = m2->begin(), fe1 = m2->end(); f1 != fe1; f1++)
	{
		Function* func = f1;
		if (func->isDeclaration()) continue;

		// Search for the current function in original module
		// functions list.		
		// If function is not in list of original module, then
		// it is generated by the loop extractor.
		// Append "always inline" attribute to all other functions.
		if (m1->getFunction(func->getName()))
		{
			const AttrListPtr attr = func->getAttributes();
			const AttrListPtr attr_new = attr.addAttr(~0U, Attribute::AlwaysInline);
			func->setAttributes(attr_new);
			continue;
		}

		// Each such function must be extracted to the
		// standalone module and packed into resulting
		// object file data section.
		printf("Preparing loop function %s ...\n", func->getName().data());
		
		// Reset to default visibility.
		func->setVisibility(GlobalValue::DefaultVisibility);

		// Replace call to this function in module with call to launcher.
		bool found = false;
		for (Module::iterator f2 = m2->begin(), fe2 = m2->end(); (f2 != fe2) && !found; f2++)
			for (Function::iterator bb = f2->begin(); (bb != f2->end()) && !found; bb++)
				for (BasicBlock::iterator i = bb->begin(); i != bb->end(); i++)
				{
					// Check if instruction in focus is a call.
					CallInst* call = dyn_cast<CallInst>(cast<Value>(i));
					if (!call) continue;
					
					// Check if function is called (needs -instcombine pass).
					Function* callee = call->getCalledFunction();
					if (!callee && !callee->isDeclaration()) continue;
					if (callee->getName() != func->getName()) continue;

					// Start forming new function call argument list
					// by copying the list of original function call.
					SmallVector<Value*, 16> call_args(call->op_begin(), call->op_end());
					
					// Insert first extra argument - the number of
					// original call arguments.
					int nargs = call->getNumArgOperands();
					Type* int32Ty = Type::getInt32Ty(context);
					call_args.insert(call_args.begin(),
						ConstantInt::get(int32Ty, nargs));
					
					// Create a constant array holding original called
					// function name.
					Constant* name = ConstantArray::get(
						context, callee->getName(), true);
					
					// Create global variable to hold the function name
					// string.
					GlobalVariable* GV1 = new GlobalVariable(*m2, name->getType(),
						true, GlobalValue::PrivateLinkage, name,
						callee->getName(), 0, false);
					
					// Convert array to pointer using GEP construct.
					std::vector<Constant*> gep_args(2, Constant::getNullValue(int32Ty));
				        
					// Insert second extra argument - the pointer to the
					// original function string name.
					call_args.insert(call_args.begin(),
						ConstantExpr::getGetElementPtr(GV1, gep_args));

					// Insert third extra argument - an array of original
					// function arguments sizes.
					std::vector<Constant*> sizes;
					for (int i = 0; i != nargs; i++)
					{
						Value* arg = call->getArgOperand(i);
						int size = tdata->getTypeStoreSize(arg->getType());
						sizes.push_back(ConstantInt::get(int32Ty, size));
					}
					Constant* csizes = ConstantArray::get(
						ArrayType::get(sizes[0]->getType(), sizes.size()), sizes);
					GlobalVariable* GV2 = new GlobalVariable(*m2, csizes->getType(),
						true, GlobalValue::PrivateLinkage, csizes,
						callee->getName(), 0, false);					
					call_args.insert(call_args.begin() + 2,
						ConstantExpr::getGetElementPtr(GV2, gep_args));
					
					// Create new function call with new call arguments
					// and copy old call properties.
					CallInst* newcall = CallInst::Create(launch, call_args, "", call);
					newcall->takeName(call);
					newcall->setCallingConv(call->getCallingConv());
					newcall->setAttributes(call->getAttributes());
					newcall->setDebugLoc(call->getDebugLoc());
					
					// Replace old call with new one.
					call->eraseFromParent();
					
					found = true;
					break;
				}
	}
	
	//
	// 7) Apply optimization passes to the resulting common
	// module.
	//
	{
		PassManager manager;
		manager.add(createLowerSetJmpPass());
		PassManagerBuilder builder;
		builder.Inliner = createFunctionInliningPass();
		builder.OptLevel = 3;
		builder.DisableSimplifyLibCalls = true;
		builder.populateModulePassManager(manager);
		manager.run(*m2);
	}

	//
	// 8) Embed the resulting module into object file.
	//
	int ir_fd = -1;
	string ir_output;
	{
		string ir_string;
		raw_string_ostream ir(ir_string);
		ir << (*m2);
		ir_output = fileprefix + "XXXXXX";
		char* c_ir_output = new char[ir_output.size() + 1];
		strcpy(c_ir_output, ir_output.c_str());
		ir_fd = mkstemp(c_ir_output);
		ir_output = c_ir_output;
		delete[] c_ir_output;
		string ir_symname = "__kernelgen_" + string(input);
		util_elf_write(ir_fd, arch, ir_symname.c_str(), ir_string.c_str(), ir_string.size() + 1);
	}
	
	//
	// 9) Merge object files with binary code and IR.
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
	
	if (bin_fd >= 0) close(bin_fd);
	if (ir_fd >= 0) close(ir_fd);

	delete m1, m2, buffer1, buffer2;

	return 0;
}

