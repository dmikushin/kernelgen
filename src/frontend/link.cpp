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

#include "runtime/runtime.h"

#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <fcntl.h>
#include <iostream>
#include <memory>
#include <string.h>
#include <unistd.h>

#include "llvm/Constants.h"
#include "llvm/Instructions.h"
#include "llvm/Linker.h"
#include "llvm/LLVMContext.h"
#include "llvm/Module.h"
#include "llvm/PassManager.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/IRReader.h"
#include "llvm/Transforms/IPO/PassManagerBuilder.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/TypeBuilder.h"
#include "llvm/Target/TargetData.h"
#include "llvm/Transforms/IPO.h"
#include "llvm/Transforms/Scalar.h"
#include "llvm/Transforms/Utils/Cloning.h"
#include "llvm/Analysis/Verifier.h"

using namespace llvm;
using namespace std;

extern "C" void kernelgen_link(const char* input, const char* output)
{
	rename(input, output);

	/*//
	// 1) Check if there is "-c" option around. In this
	// case there is just compilation, not linking, but
	// in a way we do not know how to handle.
	//
	for (list<string>::iterator iarg = args.begin(), iearg = args.end();
		iarg != iearg; iarg++)
	{
		const char* arg = (*iarg).c_str();
		if (!strcmp(arg, "-c"))
			return execute(host_compiler, args, "", NULL, NULL);
	}

	//
	// 2) Extract all object files out of the command line.
	// From each object file extract LLVM IR modules and merge
	// them into single composite module.
	// In the meantime, capture an object containing main entry.
	//
	string object = "";
	cfiledesc tmp_object = cfiledesc::mktemp(fileprefix);
	LLVMContext &context = getGlobalContext();
	SMDiagnostic diag;
	Module composite("composite", context);
	for (list<string>::iterator iarg = args.begin(), iearg = args.end();
		iarg != iearg; iarg++)
	{
		const char* arg = (*iarg).c_str();
		if (strcmp(arg + strlen(arg) - 2, ".o"))
			continue;
		
		if (verbose)
			cout << "Linking " << arg << " ..." << endl;

		// Search current object for main entry. It will be
		// used as a container for LLVM IR data.
		// For build process consistency, all activities will
		// be performed on duplicate object.
		{
			celf e(arg, "");
			cregex regex("^main$", REG_EXTENDED | REG_NOSUB);
			vector<csymbol*> symbols = e.getSymtab()->find(regex);
			if (symbols.size())
			{
				object = arg;
				list<string> cp_args;
				cp_args.push_back(arg);
				cp_args.push_back(tmp_object.getFilename());
				int status = execute("cp", cp_args, "", NULL, NULL);
				if (status) return status;
				*iarg = tmp_object.getFilename();
				arg = tmp_object.getFilename().c_str();
			}
		}

		// Load and link together all LLVM IR modules from
		// symbols names starting with __kernelgen_.
		vector<string> names;
		{
			celf e(arg, "");
			cregex regex("^__kernelgen_.*$", REG_EXTENDED | REG_NOSUB);
			vector<csymbol*> symbols = e.getSymtab()->find(regex);

			// Load data for discovered symbols names.
			// For each symbol name containing module, link into
			// the composite module.
			for (vector<csymbol*>::iterator i = symbols.begin(),
				ie = symbols.end(); i != ie; i++)
			{
				csymbol* symbol = *i;
				const char* data = symbol->getData();
				const string& name = symbol->getName();
				names.push_back(name);

				MemoryBuffer* buffer = MemoryBuffer::getMemBuffer(data);
				std::auto_ptr<Module> m;
				m.reset(ParseIR(buffer, diag, context));

				if (!m.get()) THROW("Error loading module " << name);
		
				if (verbose)
					cout << "Linking " << name << endl;
		
				string err;
				if (Linker::LinkModules(&composite, m.get(),Linker::PreserveSource, &err))
					THROW("Error linking module " << name << " : " << err);
			}
		}

		// Remove symbols starting with __kernelgen_.
		// TODO: do not strip symbols here!!
		for (vector<string>::iterator i = names.begin(),
			ie = names.end(); i != ie; i++)
		{
			string& name = *i;
			list<string> objcopy_args;
			objcopy_args.push_back("--strip-symbol=" + name);
			objcopy_args.push_back(arg);
			int status = execute("objcopy", objcopy_args, "", NULL, NULL);
			if (status) return status;
		}
	}
	if (object == "")
	{
		// TODO: In general case this is not an error.
		// Missing main entry only means we are linking
		// a library. In this case every public symbol
		// must be treated as main entry.
		THROW("Cannot find object containing main entry");
	}*/
	
	//
	// 3) Convert global variables into main entry locals and
	// arguments of loops functions.
	//
	/*{
		// Form the type of structure to hold global variables.
		// Constant globals are excluded from structure.
		std::vector<Type*> paramTy;
		for (Module::global_iterator i = composite.global_begin(),
			ie = composite.global_end(); i != ie; i++)
		{
			GlobalVariable* gvar = i;
			if (i->isConstant()) continue;
			paramTy.push_back(i->getType());
		}
		Type* structTy = StructType::get(context, paramTy, true);

		// Allocate globals structure in the beginning of main.
		Instruction* first = composite.getFunction("main")->begin()->begin();
		AllocaInst* structArg = new AllocaInst(structTy, 0, "", first);

		// Fill globals structure with initial values of globals.
		int ii = 0;
		vector<GlobalVariable*> remove;
		for (Module::global_iterator i = composite.global_begin(),
			ie = composite.global_end(); i != ie; i++)
		{
			GlobalVariable* gvar = i;
			if (gvar->isConstant()) continue;

			// Generate index.
			Value *Idx[2];
			Idx[0] = Constant::getNullValue(Type::getInt32Ty(context));
			Idx[1] = ConstantInt::get(Type::getInt32Ty(context), ii);

			// Get address of "globals[i]" in struct.
			GetElementPtrInst *GEP = GetElementPtrInst::Create(
				structArg, Idx, "", first);

			if (gvar->hasInitializer())
			{
				// Store initial value to that address.
				StoreInst *SI = new StoreInst(gvar->getInitializer(), GEP, false, first);
			}

			// TODO: Replace all uses with GEP & LoadInst.
			// gvar->replaceAllUsesWith(GEP);
			for (Value::use_iterator i = gvar->use_begin(),
				ie = gvar->use_end(); i != ie; i++)
			{
				LoadInst* LI = new LoadInst(GEP, "", *i);
				i->replaceAllUsesWith(LI);
			}

			remove.push_back(gvar);
			
			ii++;
		}
		
		// Erase replaced globals.
		for (vector<GlobalVariable*>::iterator i = remove.begin(), ie = remove.end();
			i != ie; i++)
		{
			GlobalVariable* gvar = *i;
			gvar->eraseFromParent();
		}

		//composite.dump();

		// For each loop function add globals structure
		// as an argument.

		// Replace uses of globals with uses of local globals struct.
	}*/

	/*//
	// 4) Rename main entry and insert another one
	// into composite module.
	//
	{
		Function* main_ = composite.getFunction("main");
		main_->setName("main_");
		
		// Create new main(int* args).
		Function* main = Function::Create(
			TypeBuilder<void(types::i<32>*), true>::get(context),
			GlobalValue::ExternalLinkage, "main", &composite);
		main->setHasUWTable();
		main->setDoesNotThrow();

		if (main_->doesNotThrow())
			main->setDoesNotThrow(true);

		// Create basic block in new main.
		BasicBlock* root = BasicBlock::Create(context, "entry");
		main->getBasicBlockList().push_back(root);

		// Add no capture attribute on argument.
		Function::arg_iterator arg = main->arg_begin();
		arg->setName("args");
		arg->addAttr(Attribute::NoCapture);

		// Create and insert GEP to (int*)(args + 3).
		Value *Idx1[1];
		Idx1[0] = ConstantInt::get(Type::getInt64Ty(context), 6);
		GetElementPtrInst *GEP1 = GetElementPtrInst::CreateInBounds(
			arg, Idx1, "", root);

		// Bitcast (int8***)(int*)(args + 3).
		Value* argv1 = new BitCastInst(GEP1, Type::getInt8Ty(context)->
			getPointerTo(0)->getPointerTo(0)->getPointerTo(0), "", root);

		// Load argv from int8***.
		LoadInst* argv2 = new LoadInst(argv1, "", root);
		argv2->setAlignment(1);

		// Create and insert GEP to (int*)(args + 2).
		Value *Idx2[1];
		Idx2[0] = ConstantInt::get(Type::getInt64Ty(context), 4);
		GetElementPtrInst *GEP2 = GetElementPtrInst::CreateInBounds(
			arg, Idx2, "", root);

		// Load argc from (int*)args.
		LoadInst* argc1 = new LoadInst(GEP2, "", root);
		argc1->setAlignment(1);

		// Create argument list and call instruction to
		// call main_(int argc, char** argv).
		SmallVector<Value*, 16> call_args;
		call_args.push_back(argc1);
		call_args.push_back(argv2);
		CallInst* call = CallInst::Create(main_, call_args, "", root);
		call->setTailCall();
		call->setDoesNotThrow();

		// Create and insert GEP to (int*)(args + 5).
		Value *Idx4[1];
		Idx4[0] = ConstantInt::get(Type::getInt64Ty(context), 8);
		GetElementPtrInst *GEP4 = GetElementPtrInst::CreateInBounds(
			arg, Idx4, "", root);

		// Store call ret value to ret.
		StoreInst* ret1 = new StoreInst(call, GEP4, false, root);
		ret1->setAlignment(1);
		
		// Call kernelgen_finish to finalize execution.
		Function* finish = Function::Create(TypeBuilder<void(), true>::get(context),
			GlobalValue::ExternalLinkage, "kernelgen_finish", &composite);
		SmallVector<Value*, 16> finish_args;
		CallInst* finish_call = CallInst::Create(finish, finish_args, "", root);

		// Return the int result of call instruction.
		ReturnInst::Create(context, 0, root);

		if (verifyFunction(*main))
		{
			cerr << "Function verification failed!" << endl;
			return 1;
		}
	}
	
	//
	// 5) Apply optimization passes to the resulting common
	// module.
	//
	{
		PassManager manager;
		manager.add(new TargetData(&composite));
		//manager.add(createLowerSetJmpPass());
		PassManagerBuilder builder;
		builder.Inliner = createFunctionInliningPass();
		builder.OptLevel = 3;
		builder.DisableSimplifyLibCalls = true;
		builder.populateModulePassManager(manager);
		manager.run(composite);
	}

	//
	// 6) Clone composite module and transform it into the
	// "main" kernel, executing serial portions of code on
	// device.
	//
	std::auto_ptr<Module> main;
	main.reset(CloneModule(&composite));
	{
		Instruction* root = NULL;
		main->setModuleIdentifier("main");
		std::vector<GlobalValue*> loops_functions;
		for (Module::iterator f = main.get()->begin(), fe = main.get()->end(); f != fe; f++)
			for (Function::iterator bb = f->begin(); bb != f->end(); bb++)
				for (BasicBlock::iterator i = bb->begin(); i != bb->end(); i++)
				{
					// Check if instruction in focus is a call.
					CallInst* call = dyn_cast<CallInst>(cast<Value>(i));
					if (!call) continue;
				
					// Check if function is called (needs -instcombine pass).
					Function* callee = call->getCalledFunction();
					if (!callee && !callee->isDeclaration()) continue;
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
					string name = nameArray->getAsCString();
					if (verbose)
						cout << "Launcher invokes kernel " << name << endl;

					Function* func = main->getFunction(name);
					if (!func)
					{
						// TODO: not fatal, as soon as function could be defined in
						// linked library. The question is if we should we dump LLVM IR
						// from libraries right now.
						cerr << "Cannot find function " << name << endl;
						continue;
					}
					
					loops_functions.push_back(func);
					main->Dematerialize(func);
				}

		PassManager manager;
		manager.add(new TargetData(main.get()));
		
		// Delete functions called through launcher.
		manager.add(createGVExtractionPass(loops_functions, true));

		// Delete unreachable globals		
		manager.add(createGlobalDCEPass());
		
		// Remove dead debug info.
		manager.add(createStripDeadDebugInfoPass());
		
		// Remove dead func decls.
		manager.add(createStripDeadPrototypesPass());

		manager.run(*main.get());
	}
		
	//
	// 7) Clone composite module and transform it into the
	// "loop" kernels, each one executing single parallel loop.
	//
	{
		std::auto_ptr<Module> loops;
		loops.reset(CloneModule(&composite));
		{
			PassManager manager;
			manager.add(new TargetData(loops.get()));

			std::vector<GlobalValue*> plain_functions;
			for (Module::iterator f = main.get()->begin(), fe = main.get()->end(); f != fe; f++)
				if (!f->isDeclaration())
					plain_functions.push_back(loops->getFunction(f->getName()));
		
			// Delete all plain functions (that are not called through launcher).
			manager.add(createGVExtractionPass(plain_functions, true));
			manager.add(createGlobalDCEPass());
			manager.add(createStripDeadDebugInfoPass());
			manager.add(createStripDeadPrototypesPass());
			manager.run(*loops);
		}

		for (Module::iterator f1 = loops.get()->begin(), fe1 = loops.get()->end(); f1 != fe1; f1++)
		{
			if (f1->isDeclaration()) continue;
		
			auto_ptr<Module> loop;
			loop.reset(CloneModule(loops.get()));
			loop->setModuleIdentifier(f1->getName());
			std::vector<GlobalValue*> remove_functions;
			for (Module::iterator f2 = loop.get()->begin(), fe2 = loop.get()->end(); f2 != fe2; f2++)
			{
				if (f2->isDeclaration()) continue;
				if (f2->getName() != f1->getName())
					remove_functions.push_back(f2);
			}
			
			PassManager manager;
			manager.add(new TargetData(loop.get()));

			// Delete all loops functions, except entire one.
			manager.add(createGVExtractionPass(remove_functions, true));
			manager.add(createGlobalDCEPass());
			manager.add(createStripDeadDebugInfoPass());
			manager.add(createStripDeadPrototypesPass());
			manager.run(*loop);

			// Rename "loop" to "__kernelgen_loop".
			loop->getFunction(f1->getName())->setName(
				"__kernelgen_" + f1->getName());
			f1->setName("__kernelgen_" + f1->getName());

			//loop.get()->dump();

			// Embed "loop" module into object.
			{
				string ir_string;
				raw_string_ostream ir(ir_string);
				ir << (*loop.get());
				celf e(tmp_object.getFilename(), tmp_object.getFilename());
				e.getSection(".data")->addSymbol(
					string(f1->getName()),
					ir_string.c_str(), ir_string.size() + 1);
			}			
		}
	}
	
	//
	// 8) Delete all plain functions, except main out of "main" module.
	//
	{
		PassManager manager;
		manager.add(new TargetData(main.get()));

		std::vector<GlobalValue*> plain_functions;
		for (Module::iterator f = main->begin(), fe = main->end(); f != fe; f++)
			if (!f->isDeclaration() && f->getName() != "main")
				plain_functions.push_back(f);
	
		// Delete all plain functions (that are not called through launcher).
		manager.add(createGVExtractionPass(plain_functions, true));
		manager.add(createGlobalDCEPass());
		manager.add(createStripDeadDebugInfoPass());
		manager.add(createStripDeadPrototypesPass());
		manager.run(*main.get());

		// Rename "main" to "__kernelgen_main".
		Function* kernelgen_main_ = main->getFunction("main");
		kernelgen_main_->setName("__kernelgen_main");

		// Create global variable with pointer to callback structure.
		GlobalVariable* callback1 = new GlobalVariable(
			*main.get(), Type::getInt32PtrTy(context), false,
			GlobalValue::PrivateLinkage,
			Constant::getNullValue(Type::getInt32PtrTy(context)),
			"__kernelgen_callback");
		
		// Assign callback structure pointer with value received
		// from the arguments structure.
		// %struct.callback_t = type { i32, i32, i8*, i32, i8* }
		// %0 = getelementptr inbounds i32* %args, i64 10
		// %1 = bitcast i32* %0 to %struct.callback_t**
		// %2 = load %struct.callback_t** %1, align 8
		// %3 = getelementptr inbounds %struct.callback_t* %2, i64 0, i32 0
		// store i32* %3, i32** @__kernelgen_callback, align 8
		{	
			Instruction* root = kernelgen_main_->begin()->begin();
			Function::arg_iterator arg = kernelgen_main_->arg_begin();
			Value *Idx3[1];
			Idx3[0] = ConstantInt::get(Type::getInt64Ty(context), 10);
			GetElementPtrInst *GEP3 = GetElementPtrInst::CreateInBounds(
				arg, Idx3, "", root);  
			Value* callback2 = new BitCastInst(GEP3,
				Type::getInt32PtrTy(context)->getPointerTo(0), "", root);
			LoadInst* callback3 = new LoadInst(callback2, "", root);
			callback3->setAlignment(8);
			Value *Idx4[1];
			Idx4[0] = ConstantInt::get(Type::getInt64Ty(context), 0);
			GetElementPtrInst *GEP4 = GetElementPtrInst::CreateInBounds(
				callback3, Idx4, "", root);
			StoreInst* callback4 = new StoreInst(GEP4, callback1, true, root); // volatile!
			callback4->setAlignment(8);
		}

		// Create global variable with pointer to memory structure.
		GlobalVariable* memory1 = new GlobalVariable(
			*main.get(), Type::getInt32PtrTy(context), false,
			GlobalValue::PrivateLinkage,
			Constant::getNullValue(Type::getInt32PtrTy(context)),
			"__kernelgen_memory");
		
		// Assign memory structure pointer with value received
		// from the arguments structure.
		// %struct.memory_t = type { i8*, i64, i64, i64 }
		// %4 = getelementptr inbounds i32* %args, i64 12
		// %5 = bitcast i32* %4 to %struct.memory_t**
		// %6 = load %struct.memory_t** %5, align 8
		// %7 = bitcast %struct.memory_t* %6 to i32*
		// store i32* %7, i32** @__kernelgen_memory, align 8
		{	
			Instruction* root = kernelgen_main_->begin()->begin();
			Function::arg_iterator arg = kernelgen_main_->arg_begin();
			Value *Idx3[1];
			Idx3[0] = ConstantInt::get(Type::getInt64Ty(context), 12);
			GetElementPtrInst *GEP3 = GetElementPtrInst::CreateInBounds(
				arg, Idx3, "", root);  
			Value* memory2 = new BitCastInst(GEP3,
				Type::getInt32PtrTy(context)->getPointerTo(0), "", root);
			LoadInst* memory3 = new LoadInst(memory2, "", root);
			memory3->setAlignment(8);
			Value *Idx4[1];
			Idx4[0] = ConstantInt::get(Type::getInt64Ty(context), 0);
			GetElementPtrInst *GEP4 = GetElementPtrInst::CreateInBounds(
				memory3, Idx4, "", root);
			StoreInst* memory4 = new StoreInst(GEP4, memory1, true, root); // volatile!
			memory4->setAlignment(8);
		}

		//main.get()->dump();

		// Embed "main" module into object.
		{
			string ir_string;
			raw_string_ostream ir(ir_string);
			ir << (*main.get());
			celf e(tmp_object.getFilename(), tmp_object.getFilename());
			e.getSection(".data")->addSymbol(
				"__kernelgen_main", ir_string.c_str(), ir_string.size() + 1);
		}
	}
	
	//
	// 9) Rename original main entry and insert new main
	// with switch between original main and kernelgen's main.
	//
	{
		list<string> objcopy_args;
		objcopy_args.push_back("--redefine-sym main=__regular_main");
		objcopy_args.push_back(tmp_object.getFilename());
		int status = execute("objcopy", objcopy_args, "", NULL, NULL);
		if (status) return status;

		{
			list<string> merge_args_ext = merge_args;		
			merge_args_ext.push_back(tmp_object.getFilename() + "_");
			merge_args_ext.push_back(tmp_object.getFilename());
			merge_args_ext.push_back("--whole-archive");
			if (arch == 32)
				merge_args_ext.push_back("/opt/kernelgen/lib/libkernelgen.a");
			else
				merge_args_ext.push_back("/opt/kernelgen/lib64/libkernelgen.a");
			if (verbose)
			{
				cout << merge;
				for (list<string>::iterator it = merge_args_ext.begin();
					it != merge_args_ext.end(); it++)
					cout << " " << *it;
				cout << endl;
			}
			int status = execute(merge, merge_args_ext, "", NULL, NULL);
			if (status) return status;
			list<string> mv_args;
			mv_args.push_back(tmp_object.getFilename() + "_");
			mv_args.push_back(tmp_object.getFilename());
			status = execute("mv", mv_args, "", NULL, NULL);
			if (status) return status;
		}
	}

	//
	// 10) Link code using regular linker.
	//
	{
		// Adding -rdynamic to use executable global symbols
		// to resolve dependencies of subsequently loaded kernel objects.
		args.push_back("-rdynamic");
		args.push_back("/opt/kernelgen/lib/libLLVM-3.1svn.so");
		args.push_back("/opt/kernelgen/lib/LLVMPolly.so");
		args.push_back("/opt/kernelgen/lib/libdyloader.so");
		args.push_back("/opt/kernelgen/lib/libasfermi.so");
		args.push_back("-lelf");
		args.push_back("-lrt");
		args.push_back("-lgmp");
		args.push_back("-lmhash");
		args.push_back("-ldl");
		args.push_back("-lffi");
		args.push_back("-lstdc++");
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
	
	return 0;*/
}

