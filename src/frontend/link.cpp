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
#include "util/elf.h"
#include "util/util.h"

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
#include "llvm/Support/PassManagerBuilder.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/TypeBuilder.h"
#include "llvm/Target/TargetData.h"
#include "llvm/Transforms/IPO.h"
#include "llvm/Transforms/Scalar.h"
#include "llvm/Transforms/Utils/Cloning.h"

using namespace llvm;
using namespace std;
using namespace util::elf;

int link(list<string> args, list<string> kgen_args,
	string merge, list<string> merge_args,
	string input, string output, int verbose, int arch,
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

		// Search object for main entry.
		{
			int fd = open(arg, O_RDONLY);
			cregex regex("^main$", REG_EXTENDED | REG_NOSUB);
			celf e(fd, ELF_C_READ);
			vector<csymbol*> symbols = e.getSymtab()->find(regex);
			if (symbols.size()) object = arg;
			if (fd >= 0) close(fd);
		}

		// Load and link together all LLVM IR modules from
		// symbols names starting with __kernelgen_.
		vector<string> names;
		{
			int fd = open(arg, O_RDONLY);
			cregex regex("^__kernelgen_.*$", REG_EXTENDED | REG_NOSUB);
			celf e(fd, ELF_C_READ);
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
				names.push_back(symbol->getName());

				MemoryBuffer* buffer = MemoryBuffer::getMemBuffer(data);
				std::auto_ptr<Module> m;
				m.reset(ParseIR(buffer, diag, context));

				if (!m.get()) THROW("Error loading module " << name);
		
				if (verbose)
					cout << "Linking " << name << endl;
		
				string err;
				if (Linker::LinkModules(&composite, m.get(), &err))
					THROW("Error linking module " << name << " : " << err);
			}
		}

		// Remove symbols starting with __kernelgen_.
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
	// 3) Apply optimization passes to the resulting common
	// module.
	//
	{
		PassManager manager;
		manager.add(new TargetData(&composite));
		manager.add(createLowerSetJmpPass());
		PassManagerBuilder builder;
		builder.Inliner = createFunctionInliningPass();
		builder.OptLevel = 3;
		builder.DisableSimplifyLibCalls = true;
		builder.populateModulePassManager(manager);
		manager.run(composite);
	}

	//
	// 4) Clone composite module and transform it into the
	// "main" kernel, executing serial portions of code on
	// device.
	//
	Module* main = CloneModule(&composite);
	{
		main->setModuleIdentifier("main");
		std::vector<GlobalValue*> loops_functions;
		for (Module::iterator f = main->begin(), fe = main->end(); f != fe; f++)
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

					gval = main->getFunction(name);
					if (!gval)
					{
						cerr << "Cannot find function " << name.data() << endl;
						continue;
					}
					loops_functions.push_back(gval);
					main->Dematerialize(gval);
					continue;
failure:
					cerr << "Cannot get the name of kernel invoked by kernelgen_launch" << endl;
				}

		PassManager manager;
		manager.add(new TargetData(main));
		
		// Delete functions called through launcher.
		manager.add(createGVExtractionPass(loops_functions, true));

		// Delete unreachable globals		
		manager.add(createGlobalDCEPass());
		
		// Remove dead debug info.
		manager.add(createStripDeadDebugInfoPass());
		
		// Remove dead func decls.
		manager.add(createStripDeadPrototypesPass());

		manager.run(*main);
	}
		
	//
	// 5) Clone composite module and transform it into the
	// "loop" kernels, each one executing single parallel loop.
	//
	{
		Module* loops = CloneModule(&composite);
		{
			PassManager manager;
			manager.add(new TargetData(loops));

			std::vector<GlobalValue*> plain_functions;
			for (Module::iterator f = main->begin(), fe = main->end(); f != fe; f++)
				if (!f->isDeclaration())
					plain_functions.push_back(loops->getFunction(f->getName()));
		
			// Delete all plain functions (that are not called through launcher).
			manager.add(createGVExtractionPass(plain_functions, true));
			manager.add(createGlobalDCEPass());
			manager.add(createStripDeadDebugInfoPass());
			manager.add(createStripDeadPrototypesPass());
			manager.run(*loops);
		}

		for (Module::iterator f1 = loops->begin(), fe1 = loops->end(); f1 != fe1; f1++)
		{
			if (f1->isDeclaration()) continue;
		
			Module* loop = CloneModule(loops);
			loop->setModuleIdentifier(f1->getName());
			std::vector<GlobalValue*> remove_functions;
			for (Module::iterator f2 = loop->begin(), fe2 = loop->end(); f2 != fe2; f2++)
			{
				if (f2->isDeclaration()) continue;
				if (f2->getName() != f1->getName())
					remove_functions.push_back(f2);
			}
			
			PassManager manager;
			manager.add(new TargetData(loop));

			// Delete all loops functions, except entire one.
			manager.add(createGVExtractionPass(remove_functions, true));
			manager.add(createGlobalDCEPass());
			manager.add(createStripDeadDebugInfoPass());
			manager.add(createStripDeadPrototypesPass());
			manager.run(*loop);
			
			loop->dump();

			// Embed "loop" module into object.
			int ir_fd = -1;
			string ir_output;
			{
				string ir_string;
				raw_string_ostream ir(ir_string);
				ir << (*loop);
				ir_output = fileprefix + "XXXXXX";
				char* c_ir_output = new char[ir_output.size() + 1];
				strcpy(c_ir_output, ir_output.c_str());
				ir_fd = mkstemp(c_ir_output);
				ir_output = c_ir_output;
				delete[] c_ir_output;
				string ir_symname = "__kernelgen_" + string(f1->getName());
				util_elf_write(ir_fd, arch, ir_symname.c_str(),
					ir_string.c_str(), ir_string.size() + 1);
			}
			
			delete loop;

			// Merge object files with binary code and IR.
			list<string> merge_args_ext = merge_args;
			{
				merge_args_ext.push_back(object + "_");
				merge_args_ext.push_back(object);
				merge_args_ext.push_back(ir_output);
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
				mv_args.push_back(object + "_");
				mv_args.push_back(object);
				status = execute("mv", mv_args, "", NULL, NULL);
				if (status) return status;
			}

			if (ir_fd >= 0) close(ir_fd);
		}
		delete loops;
	}
	
	//
	// 6) Delete all plain functions, except main out of "main" module.
	//
	{
		PassManager manager;
		manager.add(new TargetData(main));

		std::vector<GlobalValue*> plain_functions;
		for (Module::iterator f = main->begin(), fe = main->end(); f != fe; f++)
			if (!f->isDeclaration() && f->getName() != "main")
				plain_functions.push_back(main->getFunction(f->getName()));
	
		// Delete all plain functions (that are not called through launcher).
		manager.add(createGVExtractionPass(plain_functions, true));
		manager.add(createGlobalDCEPass());
		manager.add(createStripDeadDebugInfoPass());
		manager.add(createStripDeadPrototypesPass());
		manager.run(*main);

		main->dump();

		// Embed "main" module into object.
		int ir_fd = -1;
		string ir_output;
		{
			string ir_string;
			raw_string_ostream ir(ir_string);
			ir << (*main);
			ir_output = fileprefix + "XXXXXX";
			char* c_ir_output = new char[ir_output.size() + 1];
			strcpy(c_ir_output, ir_output.c_str());
			ir_fd = mkstemp(c_ir_output);
			ir_output = c_ir_output;
			delete[] c_ir_output;
			string ir_symname = "__kernelgen_main";
			util_elf_write(ir_fd, arch, ir_symname.c_str(),
				ir_string.c_str(), ir_string.size() + 1);
		}
		
		delete main;

		// Merge object files with binary code and IR.
		{
			list<string> merge_args_ext = merge_args;
			merge_args_ext.push_back(object + "_");
			merge_args_ext.push_back(object);
			merge_args_ext.push_back(ir_output);
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
			mv_args.push_back(object + "_");
			mv_args.push_back(object);
			status = execute("mv", mv_args, "", NULL, NULL);
			if (status) return status;
		}

		if (ir_fd >= 0) close(ir_fd);		
	}
	
	//
	// 7) Rename original main entry and insert new main
	// with switch between original main and kernelgen's main.
	//
	{
		list<string> objcopy_args;
		objcopy_args.push_back("--redefine-sym main=__regular_main");
		objcopy_args.push_back(object);
		int status = execute("objcopy", objcopy_args, "", NULL, NULL);
		if (status) return status;

		{
			list<string> merge_args_ext = merge_args;		
			merge_args_ext.push_back(object + "_");
			merge_args_ext.push_back(object);
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
			mv_args.push_back(object + "_");
			mv_args.push_back(object);
			status = execute("mv", mv_args, "", NULL, NULL);
			if (status) return status;
		}
	}

	//
	// 8) Link code using regular linker.
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

