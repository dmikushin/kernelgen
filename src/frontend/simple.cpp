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

#include "runtime/elf.h"
#include "runtime/runtime.h"
#include "runtime/util.h"

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
#include "llvm/Support/FormattedStream.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/IRReader.h"
#include "llvm/Support/Program.h"
#include "llvm/Transforms/IPO/PassManagerBuilder.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/PrettyStackTrace.h"
#include "llvm/Support/Signals.h"
#include "llvm/Support/TargetRegistry.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/ToolOutputFile.h"
#include "llvm/Support/TypeBuilder.h"
#include "llvm/Target/TargetData.h"
#include "llvm/Target/TargetData.h"
#include "llvm/Transforms/IPO.h"
#include "llvm/Transforms/Scalar.h"
#include "llvm/Transforms/Utils/Cloning.h"
#include "llvm/Analysis/Verifier.h"

#include "llvm/Constants.h"
#include "llvm/LLVMContext.h"
#include "llvm/Instructions.h"
#include "llvm/Module.h"
#include "llvm/PassManager.h"
#include "llvm/Analysis/Passes.h"
#include "llvm/Support/FormattedStream.h"
#include "llvm/Support/IRReader.h"
#include "llvm/Support/Host.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Transforms/IPO/PassManagerBuilder.h"
#include "llvm/Support/raw_os_ostream.h"
#include "llvm/Support/ToolOutputFile.h"
#include "llvm/Support/TypeBuilder.h"
#include "llvm/Target/TargetData.h"
#include "llvm/Target/TargetOptions.h"
#include "llvm/Support/TargetRegistry.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Transforms/Scalar.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include "llvm/Transforms/Utils/Cloning.h"
#include "llvm/Transforms/Utils/FunctionUtils.h"
#include "llvm/Transforms/IPO.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/FormattedStream.h"

#include "BugDriver.h"

#include "BranchedLoopExtractor.h"

using namespace kernelgen;
using namespace kernelgen::runtime;
using namespace llvm;
using namespace llvm::sys;
using namespace llvm::sys::fs;
using namespace std;
using namespace util::elf;
using namespace util::io;

// The regular compiler frontend.
const char* compiler = "kernelgen-gfortran";

// The linker program.
const char* linker = "ld";

// Tracker to catch and inspect the crashing LLVM passes.
class PassTracker
{
	list<Pass*> passes;

	Module* module;
	
	string input;

	// Run the same passes in the bugpoint.
	static void handler(void* instance)
	{
		PassTracker* tracker = (PassTracker*)instance;
	
		// Initialize the bug driver
		bool FindBugs = false;
		int TimeoutValue = 300;
		int MemoryLimit = -1;
		bool UseValgrind = false;
		BugDriver D("kernelgen-simple", FindBugs, TimeoutValue, MemoryLimit,
			UseValgrind, getGlobalContext());
		D.setNewProgram(tracker->module);

		// Add currently tracked passes.
		for (list<Pass*>::iterator i = tracker->passes.begin(),
			e = tracker->passes.end(); i != e; i++)
		{
			const void *ID = (*i)->getPassID();
			const PassInfo *PI = PassRegistry::getPassRegistry()->getPassInfo(ID);
			if (PI) 
			{ D.addPass(PI->getPassArgument()); cout << PI->getPassArgument() << endl; }
			else { D.addPass((*i)->getPassName()); cout << (*i)->getPassName() << endl; }
		}

		// Reduce the test case.
		D.debugOptimizerCrash(tracker->input);
	}

public:
	PassTracker() : input("foo"), module(NULL)
	{
		sys::AddSignalHandler(PassTracker::handler, this);
	}

	void reset()
	{
		passes.clear();
		if (module)
		{
			delete module;
			module = NULL;
		}
	}

	void add(Pass *P)
	{
		passes.push_back(P);
	}
	
	void run(Module* M)
	{
		module = CloneModule(M);
	}
}
*tracker;


class TrackedPassManager : public PassManager
{
	PassTracker* tracker;

public:
	TrackedPassManager(PassTracker* tracker) :
		PassManager(), tracker(tracker) { }

	virtual void add(Pass *P)
	{
		tracker->add(P);
		PassManager::add(P);
	}
	
	virtual bool run(Module &M)
	{
		tracker->run(&M);
		return PassManager::run(M);
	}
	
	~TrackedPassManager()
	{
		tracker->reset();
	}
};


static bool a_ends_with_b(const char* a, const char* b)
{
	if (strlen(b) > strlen(a)) return false;
	return equal(a + strlen(a) - strlen(b), a + strlen(a), b);
}

Pass* createFixPointersPass();

static void addKernelgenPasses(const PassManagerBuilder &Builder, PassManagerBase &PM)
{
	PM.add(createInstructionCombiningPass());
	PM.add(createFixPointersPass());
	PM.add(createInstructionCombiningPass());
	PM.add(createBranchedLoopExtractorPass());
}

static int compile(int argc, char** argv, const char* input, const char* output)
{
	//
	// 1) Compile source code using the regular compiler.
	// Place output to the temporary file.
	//
	int fd;
	string tmp_mask = "%%%%%%%%";
	SmallString<128> gcc_output_vector;
	if (unique_file(tmp_mask, fd, gcc_output_vector))
	{
		cout << "Cannot generate gcc output file name" << endl;
		return 1;
	}
	string gcc_output = (StringRef)gcc_output_vector;
	close(fd);
	string err;
	tool_output_file object(gcc_output.c_str(), err, raw_fd_ostream::F_Binary);
	if (!err.empty())
	{
		cerr << "Cannot open output file" << gcc_output << endl;
		return 1;
	}
	{
		// Replace or add temporary output to the command line.
		vector<const char*> args;
		args.reserve(argc);
		for (int i = 0; argv[i]; i++)
		{
			if (!strcmp(argv[i], "-o"))
			{
				i++;
				continue;
			}
			args.push_back(argv[i]);
		}
		args.push_back("-o");
		args.push_back(gcc_output.c_str());
		args.push_back(NULL);
		args[0] = compiler;
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
			cerr << err << endl;
			return status;
		}
	}

	//
	// 2) Emit LLVM IR with DragonEgg.
	//
	LLVMContext &context = getGlobalContext();
	auto_ptr<Module> m;
	{
		SmallString<128> llvm_output_vector;
		if (unique_file(tmp_mask, fd, llvm_output_vector))
		{
			cout << "Cannot generate gcc output file name" << endl;
			return 1;
		}
		string llvm_output = (StringRef)llvm_output_vector;
		close(fd);
		tool_output_file llvm_object(llvm_output.c_str(), err, raw_fd_ostream::F_Binary);
		if (!err.empty())
		{
			cerr << "Cannot open output file" << llvm_output.c_str() << endl;
			return 1;
		}

		vector<const char*> args;
		args.reserve(argc);
		for (int i = 0; argv[i]; i++)
		{
			if (!strcmp(argv[i], "-g")) continue;
			if (!strcmp(argv[i], "-c")) continue;
			if (!strcmp(argv[i], "-o")) {
				i++; // skip next
				continue;
			}
			args.push_back(argv[i]);
		}
		args.push_back("-fplugin=/opt/kernelgen/lib/dragonegg.so");
		args.push_back("-fplugin-arg-dragonegg-emit-ir");
		args.push_back("-fplugin-arg-dragonegg-llvm-ir-optimize=0");
		args.push_back("-D_KERNELGEN");
		args.push_back("-S");
		args.push_back("-o"); 
		args.push_back(llvm_output.c_str());
		args.push_back(NULL);
		args[0] = compiler;
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
			cerr << err << endl;
			return status;
		}

		SMDiagnostic diag;
		m.reset(getLazyIRFileModule(llvm_output.c_str(), diag, context));
	}

	//
	// 3) Append "always inline" attribute to all existing functions.
	//
	for (Module::iterator f = m.get()->begin(), fe = m.get()->end(); f != fe; f++)
	{
		Function* func = f;
		if (func->isDeclaration()) continue;

		const AttrListPtr attr = func->getAttributes();
		const AttrListPtr attr_new = attr.addAttr(~0U, Attribute::AlwaysInline);
		func->setAttributes(attr_new);
	}
	
	//
	// 4) Inline calls and extract loops into new functions.
	// Apply optimization passes to the resulting common module.
	//
	{
		PassManagerBuilder builder;
		builder.Inliner = createFunctionInliningPass();
		builder.OptLevel = 3;
		builder.DisableSimplifyLibCalls = true;
		builder.addExtension(PassManagerBuilder::EP_ModuleOptimizerEarly,
			addKernelgenPasses);
		TrackedPassManager manager(tracker);
		builder.populateModulePassManager(manager);
		manager.run(*m.get());
	}

	//
	// 5) Embed the resulting module into object file.
	//
	{
		SmallString<128> llvm_output_vector;
		if (unique_file(tmp_mask, fd, llvm_output_vector))
		{
			cout << "Cannot generate gcc output file name" << endl;
			return 1;
		}
		string llvm_output = (StringRef)llvm_output_vector;
		close(fd);
		tool_output_file llvm_object(llvm_output.c_str(), err, raw_fd_ostream::F_Binary);
		if (!err.empty())
		{
			cerr << "Cannot open output file" << llvm_output.c_str() << endl;
			return 1;
		}

		// Put the resulting module into LLVM output file
		// as object binary. Method: create another module
		// with a global variable incorporating the contents
		// of entire module and emit it for X86_64 target.
		string ir_string;
		raw_string_ostream ir(ir_string);
		ir << (*m.get());
		Module obj_m("kernelgen", context);
		Constant* name = ConstantDataArray::getString(context, ir_string, true);
		GlobalVariable* GV1 = new GlobalVariable(obj_m, name->getType(),
			true, GlobalValue::LinkerPrivateLinkage, name,
			"__kernelgen_" + string(input), 0, false);

		InitializeAllTargets();
		InitializeAllTargetMCs();
		InitializeAllAsmPrinters();
		InitializeAllAsmParsers();

		Triple triple(m->getTargetTriple());
		if (triple.getTriple().empty())
			triple.setTriple(sys::getDefaultTargetTriple());
		string err;
		TargetOptions options;
		const Target* target = TargetRegistry::lookupTarget(triple.getTriple(), err);
		if (!target)
		{
			cerr << "Error auto-selecting target for module '" << err << endl;
			return 1;
		}
		auto_ptr<TargetMachine> machine;
		machine.reset(target->createTargetMachine(
		        triple.getTriple(), "", "", options, Reloc::PIC_, CodeModel::Default));
		if (!machine.get())
		{
			cerr <<  "Could not allocate target machine" << endl;
			return 1;
		}

		const TargetData* tdata = machine.get()->getTargetData();
		PassManager manager;
		manager.add(new TargetData(*tdata));

		tool_output_file object(llvm_output.c_str(), err, raw_fd_ostream::F_Binary);
		if (!err.empty())
		{
			cerr << "Cannot open output file" << llvm_output.c_str() << endl;
			return 1;
		}

		formatted_raw_ostream fos(object.os());
		if (machine.get()->addPassesToEmitFile(manager, fos,
	        TargetMachine::CGFT_ObjectFile, CodeGenOpt::Aggressive))
		{
			cerr << "Target does not support generation of this file type" << endl;
			return 1;
		}

		manager.run(obj_m);
		fos.flush();

		vector<const char*> args;
		args.push_back(linker);
		args.push_back("--unresolved-symbols=ignore-all");
		args.push_back("-r");
		args.push_back("-o");
		args.push_back(output);
		args.push_back(gcc_output.c_str());
		args.push_back(llvm_output.c_str());
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
			cerr << err << endl;
			return status;
		}
	}

	return 0;
}

int link(list<string>& args, list<string>& kgen_args,
	string merge, list<string> merge_args,
	const char* input, const char* output, int arch,
	string host_compiler, string fileprefix)
{
	//
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
	}
	
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
			return 1;
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
			merge_args_ext.push_back("/opt/kernelgen/lib/libkernelgen.a");
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
	
	return 0;
}

extern "C" int __regular_main(int argc, char* argv[])
{
	llvm::PrettyStackTraceProgram X(argc, argv);

	cout << "kernelgen: note \"simple\" is a development frontend not intended for regular use!" << endl;

	// Behave like compiler if no arguments.
	if (argc == 1)
	{
		cout << "kernelgen: no input files" << endl;
		return 0;
	}

	// Enable or disable verbose output.
	char* cverbose = getenv("kernelgen_verbose");
	if (cverbose) verbose = atoi(cverbose);

	// Supported source code files extensions.
	vector<const char*> ext;
	ext.push_back(".c");
	ext.push_back(".cpp");
	ext.push_back(".f");
	ext.push_back(".f90");
	ext.push_back(".F");
	ext.push_back(".F90");

	// Find source code input.
	// Note simple frontend does not support multiple inputs.
	const char* input = NULL;
	for (int i = 0; argv[i]; i++)
	{
		for (int j = 0; j != ext.size(); j++)
		{
			if (!a_ends_with_b(argv[i], ext[j])) continue;
			
			if (input)
			{
				fprintf(stderr, "Multiple input files are not supported\n");
				fprintf(stderr, "in the kernelgen-simple frontend\n");
				return 1;
			}
			
			input = argv[i];
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
	vector<char> output_vector;
	char* output = NULL;
	for (int i = 0; argv[i]; i++)
	{
		if (!strcmp(argv[i], "-o"))
		{
			i++;
			output = argv[i];
			break;
		}
	}
	if (!output)
	{
		output_vector.reserve(strlen(input + 1));
		output = &output_vector[0];
		strcpy(output, input);

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

		// Trim path.
		for (int i = strlen(output); i >= 0; i--)
		{
			if (output[i] == '/')
			{
				output = output + i + 1;
				break;
			}
		}
	}

	// Linker used to merge multiple objects into single one.
	string merge = "ld";
	list<string> merge_args;

	merge_args.push_back("--unresolved-symbols=ignore-all");
	merge_args.push_back("-r");
	merge_args.push_back("-o");

	tracker = new PassTracker(); 

	// Execute either compiler or linker.
	int result;
	if (input)
		result = compile(argc, argv, input, output);
	else
	{
		list<string> args_list;
		list<string> kgen_args;
		int arch = 64;
		for (int i = 1; argv[i]; i++)
			args_list.push_back(argv[i]);
		string prefix = "/tmp/";
		result = link(args_list, kgen_args,
			merge, merge_args, input, output,
			arch, compiler, prefix);
	}
	delete tracker;
	return result;
}
