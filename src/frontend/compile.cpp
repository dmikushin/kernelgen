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
#include "runtime/elf.h"
#include "runtime/runtime.h"
#include "runtime/util.h"

#include <cstdarg>
#include <cstdlib>
#include <iostream>

#include "llvm/Constants.h"
#include "llvm/Instructions.h"
#include "llvm/LLVMContext.h"
#include "llvm/Module.h"
#include "llvm/PassManager.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/IRReader.h"
#include "llvm/Support/PassManagerBuilder.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/TypeBuilder.h"
#include "llvm/Transforms/IPO.h"
#include "llvm/ADT/SetVector.h"

#include "BranchedLoopExtractor.h"

using namespace kernelgen;
using namespace llvm;
using namespace std;
using namespace util::elf;
using namespace util::io;

int compile(list<string> args, list<string> kgen_args,
	string merge, list<string> merge_args,
	string input, string output, int arch,
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
	cfiledesc tmp_output = cfiledesc::mktemp(fileprefix);
	bool output_specified = false;
	for (list<string>::iterator iarg = args.begin(),
		iearg = args.end(); iarg != iearg; iarg++)
	{
		const char* arg = (*iarg).c_str();
		if (!strcmp(arg, "-o"))
		{
			iarg++;
			*iarg = tmp_output.getFilename();
			output_specified = true;
			break;
		}
	}
	if (!output_specified)
	{
		args.push_back("-o");
		args.push_back(tmp_output.getFilename());
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
			if (!strcmp(arg, "-g"))
			{
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
	auto_ptr<Module> m1;
	m1.reset(ParseIR(buffer1, diag, context));
	
	//m1.get()->dump();

	//
	// 4) Inline calls and extract loops into new functions.
	//
	MemoryBuffer* buffer2 = MemoryBuffer::getMemBuffer(out);
	auto_ptr<Module> m2;
	m2.reset(ParseIR(buffer2, diag, context));
	{
		PassManager manager;
		manager.add(createInstructionCombiningPass());
		manager.run(*m2.get());
	}
	std::vector<CallInst *> LoopFuctionCalls;
	{
		PassManager manager;
		manager.add(createBranchedLoopExtractorPass(LoopFuctionCalls));
		manager.run(*m2.get());
	}

	//m2.get()->dump();

	//
	// 5) Replace call to loop functions with call to launcher.
	// Append "always inline" attribute to all other functions.
	//
	Type* int32Ty = Type::getInt32Ty(context);
	Function* launch = Function::Create(
		TypeBuilder<types::i<32>(types::i<8>*, types::i<64>, types::i<32>*), true>::get(context),
		GlobalValue::ExternalLinkage, "kernelgen_launch", m2.get());
	for (Module::iterator f1 = m2.get()->begin(), fe1 = m2.get()->end(); f1 != fe1; f1++)
	{
		Function* func = f1;
		if (func->isDeclaration()) continue;

		// Search for the current function in original module
		// functions list.		
		// If function is not in list of original module, then
		// it is generated by the loop extractor.
		// Append "always inline" attribute to all other functions.
		if (m1.get()->getFunction(func->getName()))
		{
			const AttrListPtr attr = func->getAttributes();
			const AttrListPtr attr_new = attr.addAttr(~0U, Attribute::AlwaysInline);
			func->setAttributes(attr_new);
			continue;
		}

		// Each such function must be extracted to the
		// standalone module and packed into resulting
		// object file data section.
		if (verbose)
			cout << "Preparing loop function " << func->getName().data() <<
				" ..." << endl;
		
		// Reset to default visibility.
		func->setVisibility(GlobalValue::DefaultVisibility);
		
		// Reset to default linkage.
		func->setLinkage(GlobalValue::ExternalLinkage);

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
					if (!callee) continue;
					if (callee->isDeclaration()) continue;
					if (callee->getName() != func->getName()) continue;
					
					// Create a constant array holding original called
					// function name.
					Constant* name = ConstantArray::get(
						context, callee->getName(), true);

					// Create and initialize the memory buffer for name.
					ArrayType* nameTy = cast<ArrayType>(name->getType());
					AllocaInst* nameAlloc = new AllocaInst(nameTy, "", call);
					StoreInst* nameInit = new StoreInst(name, nameAlloc, "", call);
					Value* Idx[2];
					Idx[0] = Constant::getNullValue(Type::getInt32Ty(context));
					Idx[1] = ConstantInt::get(Type::getInt32Ty(context), 0);
					GetElementPtrInst* namePtr = GetElementPtrInst::Create(nameAlloc, Idx, "", call);

					// Add pointer to the original function string name.
					SmallVector<Value*, 16> call_args;
					call_args.push_back(namePtr);

					// Add size of the aggregated arguments structure.
					{
						BitCastInst* BC = new BitCastInst(
							call->getArgOperand(0), Type::getInt64PtrTy(context),
							"", call);

						LoadInst* LI = new LoadInst(BC, "", call);
						call_args.push_back(LI);
					}	

					// Add original aggregated structure argument.
					call_args.push_back(call->getArgOperand(0));

					// Create new function call with new call arguments
					// and copy old call properties.
					CallInst* newcall = CallInst::Create(launch, call_args, "", call);
					//newcall->takeName(call);
					newcall->setCallingConv(call->getCallingConv());
					newcall->setAttributes(call->getAttributes());
					newcall->setDebugLoc(call->getDebugLoc());
					
					// Replace old call with new one.
					call->replaceAllUsesWith(newcall);
					call->eraseFromParent();
					
					found = true;
					break;
				}
	}

	//m2.get()->dump();
	
	//
	// 6) Apply optimization passes to the resulting common
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
		manager.run(*m2.get());
	}
	
	//m2.get()->dump();

	//
	// 7) Embed the resulting module into object file.
	//
	{
		string ir_string;
		raw_string_ostream ir(ir_string);
		ir << (*m2.get());
		celf e(tmp_output.getFilename(), output);
		e.getSection(".data")->addSymbol(
			"__kernelgen_" + string(input),
			ir_string.c_str(), ir_string.size() + 1);
	}

	return 0;
}

