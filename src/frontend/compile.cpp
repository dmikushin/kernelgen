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
using namespace kernelgen::runtime;
using namespace llvm;
using namespace std;
using namespace util::elf;
using namespace util::io;

static Function* transformToVoid(Function* &oldFunction)
{
	LLVMContext & context = getGlobalContext();
	Type* oldReturnType = oldFunction->getReturnType();
	if (oldReturnType == Type::getVoidTy(context))
		return oldFunction;

	FunctionType* oldFunctionType = oldFunction->getFunctionType();
	std::vector<Type*> argTypes;
	int numParams = oldFunctionType->getNumParams();
	for(int i = 0; i< numParams; i++)
		argTypes.push_back(oldFunctionType->getParamType(i));

	FunctionType* newFunctionType = FunctionType::get(Type::getVoidTy(context), argTypes, false);
	Function* newFunction = Function::Create(
		newFunctionType, GlobalValue::InternalLinkage, "", oldFunction->getParent());
	newFunction->takeName(oldFunction);
	vector<BasicBlock*> Blocks;
	Function::BasicBlockListType& newBBList = newFunction->getBasicBlockList();
	for (Function::iterator BB = oldFunction->begin(),
		BB_End = oldFunction->end(); BB != BB_End; BB++)
		Blocks.push_back(BB);
	for (int i = 0; i < Blocks.size(); i++)
	{
		Blocks[i]->removeFromParent();
		newBBList.push_back(Blocks[i]);
		ReturnInst * TI ;
		if (TI = dyn_cast<ReturnInst>(Blocks[i]->getTerminator()))
		{
			TI->eraseFromParent();
			ReturnInst::Create(context,Blocks[i]);
		}
	}
	for (Function::arg_iterator AI = oldFunction->arg_begin(),
		NewAI = newFunction->arg_begin(), AI_End = oldFunction->arg_end();
		AI != AI_End; AI++,NewAI++)
	{
		NewAI->setName(AI->getName());
		AI->replaceAllUsesWith(NewAI);
	}

	return newFunction;
}

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
			/*if (!strcmp(arg, "-g"))
			{
				continue;
			}*/
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
	// 3) Append "always inline" attribute to all existing functions.
	//
	LLVMContext &context = getGlobalContext();
	SMDiagnostic diag;
	MemoryBuffer* buffer1 = MemoryBuffer::getMemBuffer(out);
	auto_ptr<Module> m;
	m.reset(ParseIR(buffer1, diag, context));
	for (Module::iterator f = m.get()->begin(), fe = m.get()->end(); f != fe; f++)
	{
		Function* func = f;
		if (func->isDeclaration()) continue;

		const AttrListPtr attr = func->getAttributes();
		const AttrListPtr attr_new = attr.addAttr(~0U, Attribute::AlwaysInline);
		func->setAttributes(attr_new);
	}
	
	//m.get()->dump();

	//
	// 4) Inline calls and extract loops into new functions.
	//
	{
		std::vector<CallInst*> LoopFuctionCalls;
		PassManager manager;
		manager.add(createInstructionCombiningPass());
		manager.add(createBranchedLoopExtractorPass(LoopFuctionCalls));
		manager.run(*m.get());
	}

	//m.get()->dump();

	//
	// 5) Apply optimization passes to the resulting common
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
		manager.run(*m.get());
	}
	
	//m.get()->dump();

	//
	// 6) Embed the resulting module into object file.
	//
	{
		string ir_string;
		raw_string_ostream ir(ir_string);
		ir << (*m.get());
		celf e(tmp_output.getFilename(), output);
		e.getSection(".data")->addSymbol(
			"__kernelgen_" + string(input),
			ir_string.c_str(), ir_string.size() + 1);
	}

	return 0;
}

