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

#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <elf.h>
#include <fcntl.h>
#include <fstream>
#include <gelf.h>
#include <iostream>
#include <list>
#include <memory>
#include <sstream>
#include <string.h>
#include <unistd.h>
#include <vector>

#include "llvm/Constants.h"
#include "llvm/Instructions.h"
#include "llvm/Linker.h"
#include "llvm/LLVMContext.h"
#include "llvm/Module.h"
#include "llvm/PassManager.h"
#include "llvm/Analysis/Verifier.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/FormattedStream.h"
#include "llvm/Support/Host.h"
#include "llvm/Support/IRReader.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/PluginLoader.h"
#include "llvm/Support/PrettyStackTrace.h"
#include "llvm/Support/Program.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/Signals.h"
#include "llvm/Support/TargetRegistry.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/ToolOutputFile.h"
#include "llvm/Support/TypeBuilder.h"
#include "llvm/Target/TargetData.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetOptions.h"
#include "llvm/Transforms/IPO.h"
#include "llvm/Transforms/IPO/PassManagerBuilder.h"
#include "llvm/Transforms/Scalar.h"
#include "llvm/Transforms/Utils/Cloning.h"

#include "BranchedLoopExtractor.h"
#include "tracker.h"

using namespace llvm;
using namespace llvm::sys;
using namespace llvm::sys::fs;
using namespace std;

static int verbose = 0;

const char* compiler = "kernelgen-gfortran";
const char* linker = "ld";
const char* objcopy = "objcopy";
const char* cp = "cp";

struct fallback_args_t
{
	const char* input;
	const char* output;
	PassTracker* tracker;
};

// A fallback function to be called in case kernelgen-enabled
// link process fails by some reason.
void fallback(void* arg)
{
	fallback_args_t* fallback_args =
			(fallback_args_t*)arg;

	rename(fallback_args->input, fallback_args->output);

	delete fallback_args;
	delete tracker;
	exit(0);
}

extern "C" void kernelgen_link(const char* input, const char* output)
{
	// Enable or disable verbose output.
	char* cverbose = getenv("KERNELGEN_VERBOSE");
	if (cverbose) verbose = atoi(cverbose);

	//
	// 1) Turn on LLVM passes bug tracker.
	//
	fallback_args_t* fallback_args = new fallback_args_t();
	fallback_args->input = input;
	fallback_args->output = output;
	PassTracker* tracker = new PassTracker(input, &fallback, fallback_args);
	PluginLoader loader;
	loader.operator =("libkernelgen-opt.so");
	fallback_args->tracker = tracker;

	//
	// 2) Extract LLVM IR modules from the output file and merge
	// them into single composite module.
	//
	int fd;
	string tmp_mask = "%%%%%%%%";
	SmallString<128> tmp_main_vector;
	string tmp_main_output1 = input;
	if (unique_file(tmp_mask, fd, tmp_main_vector))
	{
		cout << "Cannot generate main output file name" << endl;
		abort();
	}
	string tmp_main_output2 = (StringRef)tmp_main_vector;
	close(fd);
	string err;
	tool_output_file tmp_main_object2(tmp_main_output2.c_str(), err, raw_fd_ostream::F_Binary);
	if (!err.empty())
	{
		cerr << "Cannot open output file" << tmp_main_output2.c_str() << endl;
		abort();
	}
	LLVMContext &context = getGlobalContext();
	SMDiagnostic diag;
	Module composite("composite", context);
	if (elf_version(EV_CURRENT) == EV_NONE)
	{
		cerr << "ELF library initialization failed: " << elf_errmsg(-1) << endl;
		abort();
	}

	Elf* e = NULL;
	try
	{
		vector<char> container;
		char *image = NULL;
		stringstream stream(stringstream::in | stringstream::out |
			stringstream::binary);
		ifstream f(input, ios::in | ios::binary);
		stream << f.rdbuf();
		f.close();
		string str = stream.str();
		container.resize(str.size() + 1);
		image = (char*)&container[0];
		memcpy(image, str.c_str(), str.size() + 1);

		if (strncmp(image, ELFMAG, 4))
		{
			cerr << "Cannot read ELF image from " << input;
			throw;
		}

		// Walk through the ELF image and record the positions
		// of the .kernelgen section.
		e = elf_memory(image, container.size());
		if (!e)
		{
			cerr << "elf_begin() failed: " << elf_errmsg(-1) << endl;
			throw;
		}
		size_t shstrndx;
		if (elf_getshdrstrndx(e, &shstrndx))
		{
			cerr << "elf_getshdrstrndx() failed: " << elf_errmsg(-1) << endl;
			throw;
		}
		Elf_Data* symbols = NULL;
		int nsymbols = 0, ikernelgen = -1;
		int64_t okernelgen = 0;
		Elf_Scn* scn = elf_nextscn(e, NULL);
		GElf_Shdr symtab;
		for (int i = 1; scn != NULL; scn = elf_nextscn(e, scn), i++)
		{
			GElf_Shdr shdr;
			if (!gelf_getshdr(scn, &shdr))
			{
				cerr << "gelf_getshdr() failed for " << elf_errmsg(-1) << endl;
				throw;
			}

			if (shdr.sh_type == SHT_SYMTAB)
			{
				symbols = elf_getdata(scn, NULL);
				if (!symbols)
				{
					cerr << "elf_getdata() failed for " << elf_errmsg(-1);
					throw;
				}
				if (shdr.sh_entsize)
					nsymbols = shdr.sh_size / shdr.sh_entsize;
				symtab = shdr;
			}

			char* name = NULL;
			if ((name = elf_strptr(e, shstrndx, shdr.sh_name)) == NULL)
			{
				cerr << "Cannot read the section " << i << " name" << endl;
				throw;
			}

			if (!strcmp(name, ".kernelgen"))
			{
				ikernelgen = i;

				// Account section address, since in this case the
				// input binary is fully linked.
				okernelgen = shdr.sh_offset - shdr.sh_addr;
			}
		}

		// Early exit if no .kernelgen section.
		// Means we work with an ordinary object file.
		if ((ikernelgen == -1) || !symbols)
		{
			elf_end(e);
			fallback(fallback_args);
			delete fallback_args;
			delete tracker;
			return;
		}

		for (int isymbol = 0; isymbol < nsymbols; isymbol++)
		{
			GElf_Sym symbol;
			gelf_getsym(symbols, isymbol, &symbol);
			char* name = elf_strptr(e, symtab.sh_link, symbol.st_name);

			// Find all symbols belonging to the .kernelgen section and link
			// them together into composite module.
			if ((GELF_ST_TYPE(symbol.st_info) == STT_OBJECT) &&
				(symbol.st_shndx == ikernelgen))
			{
				MemoryBuffer* buffer = MemoryBuffer::getMemBuffer(
						image + okernelgen + symbol.st_value);
				if (!buffer)
				{
					cerr << "Error reading object file symbol " << name << endl;
					abort();
				}
				auto_ptr<Module> m;
				m.reset(ParseIR(buffer, diag, context));
				if (!m.get())
				{
					cerr << "Error parsing LLVM IR module from symbol " << name << endl;
					abort();
				}

				string err;
				if (Linker::LinkModules(&composite, m.get(), Linker::PreserveSource, &err))
				{
					cerr << "Error linking module " << name << " : " << err << endl;
					abort();
				}
			}
		}
	}
	catch (...)
	{
		if (e) elf_end(e);
		abort();
	}
	elf_end(e);
	
	//
	// 3) TODO: Convert global variables into main entry locals and
	// arguments of loops functions.
	//

	//
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
			abort();
		}
	}
	
	//
	// 5) Apply optimization passes to the resulting common
	// module.
	//
	{
		TrackedPassManager manager(tracker);
		manager.add(new TargetData(&composite));
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
					if (!callee) continue;
					if (!callee->isDeclaration()) continue;
					if (callee->getName() != "kernelgen_launch") continue;
				
					// Get the called function name from the metadata node.
					MDNode* nameMD = call->getMetadata("kernelgen_launch");
					if (!nameMD)
					{
						cerr << "Cannot find kernelgen_launch metadata" << endl;
						abort();
					}
					if (nameMD->getNumOperands() != 1)
					{
						cerr << "Unexpected kernelgen_launch metadata number of operands" << endl;
						abort();
					}
					ConstantDataArray* nameArray = dyn_cast<ConstantDataArray>(
						nameMD->getOperand(0));
					if (!nameArray)
					{
						cerr << "Invalid kernelgen_launch metadata operand" << endl;
						abort();
					}
					if (!nameArray->isCString())
					{
						cerr << "Invalid kernelgen_launch metadata operand" << endl;
						abort();
					}
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

		TrackedPassManager manager(tracker);
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

	InitializeAllTargets();
	InitializeAllTargetMCs();
	InitializeAllAsmPrinters();
	InitializeAllAsmParsers();

	Triple triple(composite.getTargetTriple());
	if (triple.getTriple().empty())
		triple.setTriple(sys::getDefaultTargetTriple());
	TargetOptions options;
	const Target* target = TargetRegistry::lookupTarget(triple.getTriple(), err);
	if (!target)
	{
		cerr << "Error auto-selecting target for module '" << err << endl;
		abort();
	}
	auto_ptr<TargetMachine> machine;
	machine.reset(target->createTargetMachine(
	        triple.getTriple(), "", "", options, Reloc::PIC_, CodeModel::Default));
	if (!machine.get())
	{
		cerr <<  "Could not allocate target machine" << endl;
		abort();
	}

	const TargetData* tdata = machine.get()->getTargetData();

	//
	// 7) Clone composite module and transform it into the
	// "loop" kernels, each one executing single parallel loop.
	//
	{
		std::auto_ptr<Module> loops;
		loops.reset(CloneModule(&composite));
		{
			TrackedPassManager manager(tracker);
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

		tmp_main_vector.clear();
		if (unique_file(tmp_mask, fd, tmp_main_vector))
		{
			cout << "Cannot generate temporary main object file name" << endl;
			abort();
		}
		string tmp_loop_output = (StringRef)tmp_main_vector;
		close(fd);

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
			
			TrackedPassManager manager(tracker);
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
				// Put the resulting module into LLVM output file
				// as object binary. Method: create another module
				// with a global variable incorporating the contents
				// of entire module and emit it for X86_64 target.
				string ir_string;
				raw_string_ostream ir(ir_string);
				ir << (*loop.get());
				Module obj_m("kernelgen", context);
				Constant* name = ConstantDataArray::getString(context, ir_string, true);
				GlobalVariable* GV1 = new GlobalVariable(obj_m, name->getType(),
					true, GlobalValue::LinkerPrivateLinkage, name,
					f1->getName(), 0, false);

				PassManager manager;
				manager.add(new TargetData(*tdata));

				tool_output_file tmp_loop_object(tmp_loop_output.c_str(), err, raw_fd_ostream::F_Binary);
				if (!err.empty())
				{
					cerr << "Cannot open output file" << tmp_loop_output.c_str() << endl;
					abort();
				}

				formatted_raw_ostream fos(tmp_loop_object.os());
				if (machine.get()->addPassesToEmitFile(manager, fos,
			        TargetMachine::CGFT_ObjectFile, CodeGenOpt::None))
				{
					cerr << "Target does not support generation of this file type" << endl;
					abort();
				}

				manager.run(obj_m);
				fos.flush();

				vector<const char*> args;
				args.push_back(linker);
				args.push_back("--unresolved-symbols=ignore-all");
				args.push_back("-r");
				args.push_back("-o");
				args.push_back(tmp_main_output2.c_str());
				args.push_back(tmp_main_output1.c_str());
				args.push_back(tmp_loop_output.c_str());
				args.push_back(NULL);
				if (verbose)
				{
					cout << args[0];
					for (int i = 1; args[i]; i++)
						cout << " " << args[i];
					cout << endl;
				}
				int status = Program::ExecuteAndWait(
					Program::FindProgramByName(linker), &args[0],
					NULL, NULL, 0, 0, &err);
				if (status)
				{
					cerr << err;
					abort();
				}

				// Swap tmp_main_output 1 and 2.
				string swap = tmp_main_output1;
				tmp_main_output1 = tmp_main_output2;
				tmp_main_output2 = swap;
			}
		}
	}
	
	//
	// 8) Delete all plain functions, except main out of "main" module.
	// Add wrapper around main to make it compatible with kernelgen_launch.
	//
	{
		TrackedPassManager manager(tracker);
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

		// Embed "main" module into main_output.
		{
			// Put the resulting module into LLVM output file
			// as object binary. Method: create another module
			// with a global variable incorporating the contents
			// of entire module and emit it for X86_64 target.
			string ir_string;
			raw_string_ostream ir(ir_string);
			ir << (*main.get());
			Module obj_m("kernelgen", context);
			Constant* name = ConstantDataArray::getString(context, ir_string, true);
			GlobalVariable* GV1 = new GlobalVariable(obj_m, name->getType(),
				true, GlobalValue::LinkerPrivateLinkage, name,
				"__kernelgen_main", 0, false);

			PassManager manager;
			manager.add(new TargetData(*tdata));

			tmp_main_vector.clear();
			if (unique_file(tmp_mask, fd, tmp_main_vector))
			{
				cout << "Cannot generate temporary main object file name" << endl;
				abort();
			}
			string tmp_main_output = (StringRef)tmp_main_vector;
			close(fd);
			tool_output_file tmp_main_object(tmp_main_output.c_str(), err, raw_fd_ostream::F_Binary);
			if (!err.empty())
			{
				cerr << "Cannot open output file" << tmp_main_output.c_str() << endl;
				abort();
			}

			formatted_raw_ostream fos(tmp_main_object.os());
			if (machine.get()->addPassesToEmitFile(manager, fos,
		        TargetMachine::CGFT_ObjectFile, CodeGenOpt::None))
			{
				cerr << "Target does not support generation of this file type" << endl;
				abort();
			}

			manager.run(obj_m);
			fos.flush();

			vector<const char*> args;
			args.push_back(linker);
			args.push_back("--unresolved-symbols=ignore-all");
			args.push_back("-r");
			args.push_back("-o");
			args.push_back(tmp_main_output2.c_str());
			args.push_back(tmp_main_output1.c_str());
			args.push_back(tmp_main_output.c_str());
			args.push_back(NULL);
			if (verbose)
			{
				cout << args[0];
				for (int i = 1; args[i]; i++)
					cout << " " << args[i];
				cout << endl;
			}
			int status = Program::ExecuteAndWait(
				Program::FindProgramByName(linker), &args[0],
				NULL, NULL, 0, 0, &err);
			if (status)
			{
				cerr << err;
				abort();
			}

			// Swap tmp_main_output 1 and 2.
			string swap = tmp_main_output1;
			tmp_main_output1 = tmp_main_output2;
			tmp_main_output2 = swap;
		}
	}

	//
	// 9) Rename original main entry and insert new main
	// with switch between original main and kernelgen's main.
	//
	{
		vector<const char*> args;
		args.push_back(objcopy);
		args.push_back("--redefine-sym");
		args.push_back("main=__regular_main");
		args.push_back(tmp_main_output1.c_str());
		args.push_back(NULL);
		if (verbose)
		{
			cout << args[0];
			for (int i = 1; args[i]; i++)
				cout << " " << args[i];
			cout << endl;
		}
		int status = Program::ExecuteAndWait(
			Program::FindProgramByName(objcopy), &args[0],
			NULL, NULL, 0, 0, &err);
		if (status)
		{
			cerr << err;
			abort();
		}
	}
	{
		vector<const char*> args;
		args.push_back(linker);
		args.push_back("--unresolved-symbols=ignore-all");
		args.push_back("-r");
		args.push_back("-o");
		args.push_back(tmp_main_output2.c_str());
		args.push_back(tmp_main_output1.c_str());
		args.push_back("--whole-archive");
		args.push_back("/opt/kernelgen/lib/libkernelgen.a");
		args.push_back(NULL);
		if (verbose)
		{
			cout << args[0];
			for (int i = 1; args[i]; i++)
				cout << " " << args[i];
			cout << endl;
		}
		int status = Program::ExecuteAndWait(
			Program::FindProgramByName(linker), &args[0],
			NULL, NULL, 0, 0, &err);
		if (status)
		{
			cerr << err;
			abort();
		}
	}

	//
	// 10) Link code using regular linker.
	//
	{
		// Adding -rdynamic to use executable global symbols
		// to resolve dependencies of subsequently loaded kernel objects.
		vector<const char*> args;
		args.push_back(compiler);
		args.push_back("-o");
		args.push_back(output);
		args.push_back(tmp_main_output2.c_str());
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
		args.push_back(NULL);
		if (verbose)
		{
			cout << args[0];
			for (int i = 1; args[i]; i++)
				cout << " " << args[i];
			cout << endl;
		}
		const char** envp = const_cast<const char **>(environ);
		vector<const char*> env;
		for (int i = 0; envp[i]; i++)
			env.push_back(envp[i]);
		env.push_back("KERNELGEN_FALLBACK=1");
		env.push_back(NULL);
		int status = Program::ExecuteAndWait(
			Program::FindProgramByName(compiler), &args[0],
			&env[0], NULL, 0, 0, &err);
		if (status)
		{
			cerr << err;
			abort();
		}
	}

	delete tracker;
}
