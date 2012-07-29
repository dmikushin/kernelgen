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
#include "llvm/LinkAllPasses.h"

#include "BranchedLoopExtractor.h"
#include "TrackedPassManager.h"

using namespace llvm;
using namespace llvm::sys;
using namespace llvm::sys::fs;
using namespace std;

static int verbose = 0;

const char* compiler = "kernelgen-gfortran";
const char* linker = "ld";
const char* objcopy = "objcopy";
const char* cp = "cp";

static bool a_ends_with_b(const char* a, const char* b)
{
	if (strlen(b) > strlen(a)) return false;
	return equal(a + strlen(a) - strlen(b), a + strlen(a), b);
}

Pass *createFixPointersPass();
Pass *createMoveUpCastsPass();

static void addKernelgenPasses(const PassManagerBuilder &Builder, PassManagerBase &PM)
{
	PM.add(createFixPointersPass());
	PM.add(createInstructionCombiningPass());
	PM.add(createMoveUpCastsPass());
	PM.add(createInstructionCombiningPass());
	PM.add(createBasicAliasAnalysisPass());
	PM.add(createGVNPass());  
	PM.add(createBranchedLoopExtractorPass());
	PM.add(createVerifierPass());
}

struct fallback_args_t
{
	int argc;
	char** argv;
	const char* input;
	const char* output;
};

static void fallback(void* arg)
{
	fallback_args_t* args = (fallback_args_t*)arg;
	int argc = args->argc;
	const char** argv = (const char**)(args->argv);

	// Compile source code using the regular compiler.
	if (verbose)
	{
		cout << argv[0];
		for (int i = 1; argv[i]; i++)
			cout << " " << argv[i];
		cout << endl;
	}
	const char** envp = const_cast<const char **>(environ);
	vector<const char*> env;
	for (int i = 0; envp[i]; i++)
		env.push_back(envp[i]);
	env.push_back("KERNELGEN_FALLBACK=1");
	env.push_back(NULL);
	string err;
	int status = Program::ExecuteAndWait(
		Program::FindProgramByName(compiler), argv,
		&env[0], NULL, 0, 0, &err);
	if (status)
	{
		cerr << err;
		exit(1);
	}

	delete args;
	delete tracker;
	exit(0);
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
			cerr << err;
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
			cerr << err;
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
	
	/*//
	// Add noalias for all used functions arguments (dirty hack).
	//
	for(Module::iterator function = m.get()->begin(), function_end = m.get()->end();
	    function != function_end; function++) {
		Function * f = function;
		int i = 1;
		for(Function::arg_iterator arg_iter = f -> arg_begin(), arg_iter_end = f -> arg_end();
		    arg_iter != arg_iter_end; arg_iter++) {
			Argument * arg = arg_iter;
			if(isa<PointerType>(*(arg -> getType())))
				if(arg -> getType() -> getSequentialElementType() -> isSingleValueType() )
					f -> setDoesNotAlias(i);
			i++;
		}
	}*/
	
	//
	// 4) Inline calls and extract loops into new functions.
	// Apply optimization passes to the resulting common module.
	//
	{
		int optLevel = 3;
		
		PassManagerBuilder builder;
		builder.Inliner = createFunctionInliningPass();
		builder.OptLevel = optLevel;
		builder.DisableSimplifyLibCalls = true;
		
		TrackedPassManager manager(tracker);
		
		if (optLevel == 0)
			addKernelgenPasses(builder,manager);
		else
			builder.addExtension(PassManagerBuilder::EP_ModuleOptimizerEarly,
				addKernelgenPasses);
		
		builder.populateModulePassManager(manager);
		manager.run(*m);
	}

	if (verbose) m->dump();

	//
	// 5) Emit the resulting LLVM IR module into temporary
	// object symbol and embed it into the final object file.
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
			input, 0, false);

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
	        TargetMachine::CGFT_ObjectFile, CodeGenOpt::None))
		{
			cerr << "Target does not support generation of this file type" << endl;
			return 1;
		}

		manager.run(obj_m);
		fos.flush();

		// Rename .data section to .kernelgen (for unification
		// with the gcc-based toolchain).
		{
			vector<const char*> args;
			args.push_back(objcopy);
			args.push_back("--rename-section");
			args.push_back(".rodata=.kernelgen");
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
				Program::FindProgramByName(objcopy), &args[0],
				NULL, NULL, 0, 0, &err);
			if (status)
			{
				cerr << err;
				return status;
			}
		}

		{
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
				cerr << err;
				return status;
			}
		}
	}

	return 0;
}

static int link(int argc, char** argv, const char* input, const char* output)
{
	//
	// 1) Check if there is "-c" option around. In this
	// case there is just compilation, not linking, but
	// in a way we do not know how to handle.
	//
	for (int i = 0; argv[i]; i++)
		if (!strcmp(argv[i], "-c"))
		{
			cerr << "Don't know what to do with this:" << endl;
			cerr << argv[0];
			for (int i = 1; argv[i]; i++)
				cerr << " " << argv[i];
			cerr << endl;
			cerr << "Note I'm a SIMPLE frontend! ";
			cerr << "For complex things try to use kernelgen-gcc instead." << endl;
			return 1;
		}

	//
	// 2) Extract all object files out of the command line.
	// From each object file extract LLVM IR modules and merge
	// them into single composite module.
	// In the meantime, find an object containing the main entry.
	//
	const char* main_output = NULL;
	int fd;
	string tmp_mask = "%%%%%%%%";
	SmallString<128> tmp_main_vector;
	if (unique_file(tmp_mask, fd, tmp_main_vector))
	{
		cout << "Cannot generate temporary main object file name" << endl;
		return 1;
	}
	string tmp_main_output1 = (StringRef)tmp_main_vector;
	close(fd);
	string err;
	tool_output_file tmp_main_object1(
		tmp_main_output1.c_str(), err, raw_fd_ostream::F_Binary);
	if (!err.empty())
	{
		cerr << "Cannot open output file" << tmp_main_output1.c_str() << endl;
		return 1;
	}
	if (unique_file(tmp_mask, fd, tmp_main_vector))
	{
		cout << "Cannot generate main output file name" << endl;
		return 1;
	}
	string tmp_main_output2 = (StringRef)tmp_main_vector;
	close(fd);
	tool_output_file tmp_main_object2(
		tmp_main_output2.c_str(), err, raw_fd_ostream::F_Binary);
	if (!err.empty())
	{
		cerr << "Cannot open output file" << tmp_main_output2.c_str() << endl;
		return 1;
	}
	LLVMContext &context = getGlobalContext();
	SMDiagnostic diag;
	Module composite("composite", context);
	if (elf_version(EV_CURRENT) == EV_NONE)
	{
		cerr << "ELF library initialization failed: " << elf_errmsg(-1) << endl;
		return 1;
	}
	for (int i = 0; argv[i]; i++)
	{
		char* arg = argv[i];
		if (!strcmp(arg + strlen(arg) - 2, ".a"))
		{
			cout << "Note kernelgen-simple does not parse objects in .a libraries!" << endl;
			continue;
		}
		if (!strcmp(arg + strlen(arg) - 2, ".so"))
		{
			cout << "Note kernelgen-simple does not parse objects in .so libraries!" << endl;
			continue;
		}
		if (strcmp(arg + strlen(arg) - 2, ".o"))
			continue;
		
		if (verbose)
			cout << "Linking " << arg << " ..." << endl;

		vector<char> container;
		char *image = NULL;
		stringstream stream(stringstream::in | stringstream::out |
			stringstream::binary);
		ifstream f(arg, ios::in | ios::binary);
		stream << f.rdbuf();
		f.close();
		string str = stream.str();
		container.resize(str.size() + 1);
		image = (char*)&container[0];
		memcpy(image, str.c_str(), str.size() + 1);

		if (strncmp(image, ELFMAG, 4))
		{
			cerr << "Cannot read ELF image from " << arg << endl;
			return 1;
		}

		// Walk through the ELF image and record the positions
		// of the .kernelgen section.
		Elf* e = elf_memory(image, container.size());
		if (!e)
		{
			cerr << "elf_begin() failed: " << elf_errmsg(-1) << endl;
			return 1;
		}
		size_t shstrndx;
		if (elf_getshdrstrndx(e, &shstrndx))
		{
			cerr << "elf_getshdrstrndx() failed: " << elf_errmsg(-1) << endl;
			return 1;
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
				return 1;
			}

			if (shdr.sh_type == SHT_SYMTAB)
			{
				symbols = elf_getdata(scn, NULL);
				if (!symbols)
				{
					cerr << "elf_getdata() failed for " << elf_errmsg(-1);
					return 1;
				}
				if (shdr.sh_entsize)
					nsymbols = shdr.sh_size / shdr.sh_entsize;
				symtab = shdr;
			}

			char* name = NULL;
			if ((name = elf_strptr(e, shstrndx, shdr.sh_name)) == NULL)
			{
				cerr << "Cannot read the section " << i << " name" << endl;
				return 1;
			}

			if (!strcmp(name, ".kernelgen"))
			{
				ikernelgen = i;
				okernelgen = shdr.sh_offset;
			}
		}

		// Early exit if no symbols
		// Do not exit on absent .kernelgen section, since the object
		// still may contain the main entry.
		if (!symbols) continue;

		for (int isymbol = 0; isymbol < nsymbols; isymbol++)
		{
			GElf_Sym symbol;
			gelf_getsym(symbols, isymbol, &symbol);
			char* name = elf_strptr(e, symtab.sh_link, symbol.st_name);

			// Search current main_output for main entry. It will be
			// used as a container for LLVM IR data.
			// For build process consistency, all activities will
			// be performed on duplicate main_output.
			if (!strcmp(name, "main"))
			{
				// Clone the main object to the temporary copy.
				vector<const char*> args;
				args.push_back(cp);
				args.push_back(arg);
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
					Program::FindProgramByName(cp), &args[0],
					NULL, NULL, 0, 0, &err);
				if (status)
				{
					cerr << err;
					return status;
				}
				main_output = arg;
			}

			// Find all symbols belonging to the .kernelgen section and link
			// them together into composite module.
			if ((GELF_ST_TYPE(symbol.st_info) == STT_OBJECT) &&
				(symbol.st_shndx == ikernelgen))
			{
				// Since our objects are not fully linked, offset in st_value
				// is relative and must be shifted by section offset to get
				// the absolute value.
				MemoryBuffer* buffer = MemoryBuffer::getMemBuffer(
						image + okernelgen + symbol.st_value);
				if (!buffer)
				{
					cerr << "Error reading object file symbol " << name << endl;
					return 1;
				}
				auto_ptr<Module> m;
				m.reset(ParseIR(buffer, diag, context));
				if (!m.get())
				{
					cerr << "Error parsing LLVM IR module from symbol " << name << endl;
					return 1;
				}

				string err;
				if (Linker::LinkModules(&composite, m.get(), Linker::DestroySource, &err))
				{
					cerr << "Error linking module " << name << " : " << err << endl;
					return 1;
				}

				// TODO: to reduce memory footprint, try:
				// composite.Dematerialize() all globals.
			}
		}
		elf_end(e);
	}
	if (!main_output)
	{
		// In general case this is not an error.
		// Missing main entry only means we are linking
		// a library.
		cerr << "Cannot find object containing main entry" << endl;
		cerr << "Note kernelgen-simple only searches in objects!" << endl;
		return 1;
	}

	//
	// 3) Rename main entry and insert new main entry into the
	// composite module. The new main entry shall have all arguments
	// wrapped into aggregator, similar to loop kernels, and also
	// contain special fields specifically for main entry configuration:
	// the callback structure and the device memory heap pointer.
	//
	FunctionType* mainTy;
	{
		// Get the regular main entry and rename in to
		// __kernelgen_regular_main.
		Function* main_ = composite.getFunction("main");
		main_->setName("_main");
		mainTy = main_->getFunctionType();

		// Check whether the prototype is supported.
		while (1)
		{
			if (mainTy == TypeBuilder<void(), true>::get(context))
				break;
			if (mainTy == TypeBuilder<void(
				types::i<32>, types::i<8>**), true>::get(context))
				break;
			if (mainTy == TypeBuilder<void(
				types::i<32>, types::i<8>**, types::i<8>**), true>::get(context))
				break;

			if (mainTy == TypeBuilder<types::i<32>(), true>::get(context))
				break;
			if (mainTy == TypeBuilder<types::i<32>(
				types::i<32>, types::i<8>**), true>::get(context))
				break;
			if (mainTy == TypeBuilder<types::i<32>(
				types::i<32>, types::i<8>**, types::i<8>**), true>::get(context))
				break;

			cerr << "Unsupported main entry prototype: ";
			mainTy->dump();
			cerr << endl;
			return 1;
		}
		
		// Create new main(int* args).
		Function* main = Function::Create(
			TypeBuilder<void(types::i<32>*), true>::get(context),
			GlobalValue::ExternalLinkage, "main", &composite);
		main->setHasUWTable();
		main->setDoesNotThrow();

		// Create basic block in new main.
		BasicBlock* root = BasicBlock::Create(context, "entry");
		main->getBasicBlockList().push_back(root);

		// Add no capture attribute on argument.
		Function::arg_iterator arg = main->arg_begin();
		arg->setName("args");
		arg->addAttr(Attribute::NoCapture);

		// Create global variable with pointer to callback structure.
		GlobalVariable* callback1 = new GlobalVariable(
			composite, Type::getInt32PtrTy(context), false,
			GlobalValue::PrivateLinkage,
			Constant::getNullValue(Type::getInt32PtrTy(context)),
			"__kernelgen_callback");

		// Assign callback structure pointer with value received
		// from the arguments structure.
		// %struct.callback_t = type { i32, i32, i8*, i32, i8* }
		// %0 = getelementptr inbounds i32* %args, i64 0
		// %1 = bitcast i32* %0 to %struct.callback_t**
		// %2 = load %struct.callback_t** %1, align 8
		// %3 = getelementptr inbounds %struct.callback_t* %2, i64 0, i32 0
		// store i32* %3, i32** @__kernelgen_callback, align 8
		{
			Value *Idx3[1];
			Idx3[0] = ConstantInt::get(Type::getInt64Ty(context), 0);
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
			composite, Type::getInt32PtrTy(context), false,
			GlobalValue::PrivateLinkage,
			Constant::getNullValue(Type::getInt32PtrTy(context)),
			"__kernelgen_memory");

		// Assign memory structure pointer with value received
		// from the arguments structure.
		// %struct.memory_t = type { i8*, i64, i64, i64 }
		// %4 = getelementptr inbounds i32* %args, i64 2
		// %5 = bitcast i32* %4 to %struct.memory_t**
		// %6 = load %struct.memory_t** %5, align 8
		// %7 = bitcast %struct.memory_t* %6 to i32*
		// store i32* %7, i32** @__kernelgen_memory, align 8
		{
			Value *Idx3[1];
			Idx3[0] = ConstantInt::get(Type::getInt64Ty(context), 2);
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

		// Create an argument list for the main_ call.
		SmallVector<Value*, 16> call_args;

		// Load the argc argument value from aggregator.
		if (main_->getFunctionType()->getNumParams() >= 2)
		{
			// Create and insert GEP to (int*)(args + 2).
			Value* Idx[] = { ConstantInt::get(Type::getInt64Ty(context), 4) };
			GetElementPtrInst* GEP = GetElementPtrInst::CreateInBounds(
				arg, Idx, "", root);

			// Load argc from (int*)args.
			LoadInst* argc = new LoadInst(GEP, "", root);
			argc->setAlignment(1);

			call_args.push_back(argc);
		}

		// Load the argv argument value from aggregator.
		if (main_->getFunctionType()->getNumParams() >= 2)
		{
			// Create and insert GEP to (int*)(args + 3).
			Value* Idx[] = { ConstantInt::get(Type::getInt64Ty(context), 6) };
			GetElementPtrInst* GEP = GetElementPtrInst::CreateInBounds(
				arg, Idx, "", root);

			// Bitcast (int8***)(int*)(args + 3).
			Value* argv1 = new BitCastInst(GEP, Type::getInt8Ty(context)->
				getPointerTo(0)->getPointerTo(0)->getPointerTo(0), "", root);

			// Load argv from int8***.
			LoadInst* argv2 = new LoadInst(argv1, "", root);
			argv2->setAlignment(1);

			call_args.push_back(argv2);
		}

		// Load the envp argument value from aggregator.
		// Create and insert GEP to (int*)(args + 4).
		if (main_->getFunctionType()->getNumParams() == 3)
		{
			Value* Idx[] = { ConstantInt::get(Type::getInt64Ty(context), 8) };
			GetElementPtrInst* GEP = GetElementPtrInst::CreateInBounds(
					arg, Idx, "", root);

			// Bitcast (int8***)(int*)(args + 4).
			Value* envp1 = new BitCastInst(GEP, Type::getInt8Ty(context)->
					getPointerTo(0)->getPointerTo(0)->getPointerTo(0), "", root);

			// Load envp from int8***.
			LoadInst* envp2 = new LoadInst(envp1, "", root);
			envp2->setAlignment(1);

			call_args.push_back(envp2);
		}

		// Create a call to main_(int argc, char* argv[], char* envp[]).
		CallInst* call = CallInst::Create(main_, call_args, "", root);
		call->setTailCall();
		call->setDoesNotThrow();

		// Set return value, if present.
		if (!main_->getReturnType()->isVoidTy())
		{
			// Create and insert GEP to (int*)(args + 5).
			Value* Idx[] = { ConstantInt::get(Type::getInt64Ty(context), 10) };
			GetElementPtrInst* GEP = GetElementPtrInst::CreateInBounds(
				arg, Idx, "", root);

			// Store the call return value to ret.
			StoreInst* ret = new StoreInst(call, GEP, false, root);
			ret->setAlignment(1);
		}
		
		// Call kernelgen_finish to finalize execution.
		Function* finish = Function::Create(TypeBuilder<void(), true>::get(context),
			GlobalValue::ExternalLinkage, "kernelgen_finish", &composite);
		SmallVector<Value*, 16> finish_args;
		CallInst* finish_call = CallInst::Create(finish, finish_args, "", root);

		// Return at the end.
		ReturnInst::Create(context, 0, root);

		if (verifyFunction(*main))
		{
			cerr << "Function verification failed!" << endl;
			return 1;
		}
	}

	//
	// 4) Perform inlining pass on the resulting common module.
	// Do not perform agressive optimizations here, or the process
	// would hang infinitely.
	//
	if (verbose)
		cout << "Inlining ..." << endl;
	{
		TrackedPassManager manager(tracker);
		manager.add(new TargetData(&composite));
		manager.add(createInstructionCombiningPass());
		PassManagerBuilder builder;
		builder.Inliner = createFunctionInliningPass();
		builder.OptLevel = 0;
		builder.DisableSimplifyLibCalls = true;
		builder.populateModulePassManager(manager);
		manager.run(composite);
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

	//
	// 5) Transform the composite module into "main" kernel,
	// executing serial portions of code on device.
	// Extract "loop" kernels, each one executing single parallel loop.
	//
	int nloops = 0;
	if (verbose)
		cout << "Prepare main kernel ... " << endl;
	{
		Instruction* root = NULL;
		composite.setModuleIdentifier("main");
		Function* f = composite.getFunction("kernelgen_launch");
		if (f)
		{
			tmp_main_vector.clear();
			if (unique_file(tmp_mask, fd, tmp_main_vector))
			{
				cout << "Cannot generate temporary main object file name" << endl;
				return 1;
			}
			string tmp_loop_output = (StringRef)tmp_main_vector;
			close(fd);

			// Extract "loop" kernels, each one executing single parallel loop.
			for (Value::use_iterator UI = f->use_begin(), UE = f->use_end(); UI != UE; UI++)
			{
				// Check if instruction in focus is a call.
				CallInst* call = dyn_cast<CallInst>(*UI);
				if (!call) continue;
				
				// Get the called function name from the metadata node.
				MDNode* nameMD = call->getMetadata("kernelgen_launch");
				if (!nameMD)
				{
					cerr << "Cannot find kernelgen_launch metadata" << endl;
					return 1;
				}
				if (nameMD->getNumOperands() != 1)
				{
					cerr << "Unexpected kernelgen_launch metadata number of operands" << endl;
					return 1;
				}
				ConstantDataArray* nameArray = dyn_cast<ConstantDataArray>(
					nameMD->getOperand(0));
				if (!nameArray)
				{
					cerr << "Invalid kernelgen_launch metadata operand" << endl;
					return 1;
				}
				if (!nameArray->isCString())
				{
					cerr << "Invalid kernelgen_launch metadata operand" << endl;
					return 1;
				}
				string name = nameArray->getAsCString();
				if (verbose)
					cout << "Launcher invokes kernel " << name << endl;

				Function* func = composite.getFunction(name);
				if (!func) continue;

				if (verbose)
					cout << "Extracting kernel " << func->getName().str() << " ..." << endl;
				
				func->removeFromParent();
				
				// Rename "loop" function to "__kernelgen_loop".
				func->setName("__kernelgen_" + func->getName());
				
				// Create new module and populate it with entire loop function.
				Module loop(func->getName(), context);
				loop.setTargetTriple(composite.getTargetTriple());
				loop.setDataLayout(composite.getDataLayout());
				loop.getFunctionList().push_back(func);
				
				// Also clone all function definitions used by entire
				// loop function to the new module.
				for (Function::iterator bb = func->begin(), be = func->end(); bb != be; bb++)
					for (BasicBlock::iterator i = bb->begin(); i != bb->end(); i++)
					{
						CallInst* call = dyn_cast<CallInst>(i);
						if (!call) continue;
				
						Function* callee = dyn_cast<Function>(
							call->getCalledValue()->stripPointerCasts());
						if (callee) loop.getOrInsertFunction(callee->getName(), callee->getFunctionType());
					}

				// Embed "loop" module into object.
				{
					// Put the resulting module into LLVM output file
					// as object binary. Method: create another module
					// with a global variable incorporating the contents
					// of entire module and emit it for X86_64 target.
					string ir_string;
					raw_string_ostream ir(ir_string);
					ir << loop;
					Module obj_m("kernelgen", context);
					Constant* name = ConstantDataArray::getString(context, ir_string, true);
					GlobalVariable* GV1 = new GlobalVariable(obj_m, name->getType(),
						true, GlobalValue::LinkerPrivateLinkage, name,
						func->getName(), 0, false);

					PassManager manager;
					manager.add(new TargetData(*tdata));

					tool_output_file tmp_loop_object(tmp_loop_output.c_str(),
						err, raw_fd_ostream::F_Binary);
					if (!err.empty())
					{
						cerr << "Cannot open output file" << tmp_loop_output.c_str() << endl;
						return 1;
					}

					formatted_raw_ostream fos(tmp_loop_object.os());
					if (machine.get()->addPassesToEmitFile(manager, fos,
					TargetMachine::CGFT_ObjectFile, CodeGenOpt::None))
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
						return status;
					}

					// Swap tmp_main_output 1 and 2.
					string swap = tmp_main_output1;
					tmp_main_output1 = tmp_main_output2;
					tmp_main_output2 = swap;
				}
				
				nloops++;
			}
		}

		TrackedPassManager manager(tracker);
		manager.add(new TargetData(&composite));
		
		// Delete unreachable globals		
		manager.add(createGlobalDCEPass());
		
		// Remove dead debug info.
		manager.add(createStripDeadDebugInfoPass());
		
		// Remove dead func decls.
		manager.add(createStripDeadPrototypesPass());

		manager.run(composite);
	}

	//
	// 6) Delete all plain functions, except main out of "main" module.
	// Add wrapper around main to make it compatible with kernelgen_launch.
	//
	if (verbose)
		cout << "Extracting kernel main ..." << endl;
	{
		TrackedPassManager manager(tracker);
		manager.add(new TargetData(&composite));

		std::vector<GlobalValue*> plain_functions;
		for (Module::iterator f = composite.begin(), fe = composite.end(); f != fe; f++)
			if (!f->isDeclaration() && f->getName() != "main")
				plain_functions.push_back(f);
	
		// Delete all plain functions (that are not called through launcher).
		manager.add(createGVExtractionPass(plain_functions, true));
		manager.add(createGlobalDCEPass());
		manager.add(createStripDeadDebugInfoPass());
		manager.add(createStripDeadPrototypesPass());
		manager.run(composite);

		// Rename "main" to "__kernelgen_main".
		Function* kernelgen_main_ = composite.getFunction("main");
		kernelgen_main_->setName("__kernelgen_main");

		// Add __kernelgen_regular_main reference.
		Function::Create(mainTy, GlobalValue::ExternalLinkage,
			"__kernelgen_regular_main", &composite);

		//composite.dump();

		// Replace all allocas by one collective alloca
		Function* f = composite.getFunction("kernelgen_launch");
		if (f)
		{
			Module *module = &composite;
			
			//maximum size of aggregated structure with parameters
			unsigned long long maximumSizeOfData=0;
			
			//list of allocas for aggregated structures with parameters
			//list<AllocaInst *> allocasForArgs;
            set<AllocaInst *> allocasForArgs;
			
            allocasForArgs.clear();
			Value * tmpArg=NULL;
			
			//walk on all kernelgen_launch's users
		    for (Value::use_iterator UI = f->use_begin(), UE = f->use_end(); UI != UE; UI++)
			{
				
				CallInst* call = dyn_cast<CallInst>(*UI);
				if (!call) continue;
				
				assert(call->getParent() -> getParent() == kernelgen_main_ &&
				  "by this time, after deleting of all plain functions, "
                  "kernelgen_launch's calls can be only on kernelgen_main");
				
				
				//retrive size of data
				tmpArg = call -> getArgOperand(1);
				assert( isa<ConstantInt>(*tmpArg) && "by this time, after optimization,"
													 "second parameter of kernelgen_launch "
													 "must be ConstantInt");
				
				//get maximum size of data
				uint64_t sizeOfData = ((ConstantInt*)tmpArg)->getZExtValue();
				if(maximumSizeOfData < sizeOfData)
				    maximumSizeOfData=sizeOfData;
				
				//retrive allocas from kernelgen_launches
                tmpArg = call -> getArgOperand(3);
				assert(isa<BitCastInst>(*tmpArg) &&  "4th parameter of kernelgen_launch "
													 "must be BitCast for int32 *");
				BitCastInst *castStructToPtr = (BitCastInst *)tmpArg;
				
				tmpArg = castStructToPtr->getOperand(0);
				assert(isa<AllocaInst>(*tmpArg) && "must be cast of AllocaInst's result");
				AllocaInst *allocaForArgs = (AllocaInst*)tmpArg;

				assert(allocaForArgs->getAllocatedType()->isStructTy() && "must be allocation of structure for args");
				
				//store alloca
				//allocasForArgs.push_back(allocaForArgs);   
                allocasForArgs.insert(allocaForArgs);
			}
			// allocate maximumSizeOfData of i8
			//AllocaInst *collectiveAlloca = new AllocaInst(Type::getInt8Ty(module->getContext()), 
			//                           ConstantInt::get(Type::getInt64Ty(module->getContext()),maximumSizeOfData),
			//						   8, "collectiveAllocaForArgs",
			//						   kernelgen_main_->begin()->getFirstNonPHI());
									   
			// allocate array [i8 x maximumSizeOfData]
			AllocaInst *collectiveAlloca = new AllocaInst(ArrayType::get(Type::getInt8Ty(module->getContext()),maximumSizeOfData),
			          "collectiveAllocaForArgs",
					  kernelgen_main_->begin()->begin());
			
			// walk on all stored allocas
			//for(list<AllocaInst *>::iterator iter=allocasForArgs.begin(), iter_end=allocasForArgs.end();
            for(set<AllocaInst *>::iterator iter=allocasForArgs.begin(), iter_end=allocasForArgs.end();
                  iter!=iter_end;iter++ )
				  {
					 AllocaInst * allocaInst=*iter;
					 
					 //get type of old alloca
					 Type * structPtrType = allocaInst -> getType();
					 
					 //create bit cast of created alloca for specified type
					 BitCastInst * bitcast=new BitCastInst(collectiveAlloca,structPtrType,"ptrToArgsStructure");
					 //insert after old alloca
					 bitcast->insertAfter(allocaInst);
					 //replace uses of old alloca with create bit cast
					 allocaInst -> replaceAllUsesWith(bitcast);
					 //erase old alloca from parent basic block
					 allocaInst -> eraseFromParent();
				  }
		
		}

		// Embed "main" module into main_output.
		{
			// Put the resulting module into LLVM output file
			// as object binary. Method: create another module
			// with a global variable incorporating the contents
			// of entire module and emit it for X86_64 target.
			string ir_string;
			raw_string_ostream ir(ir_string);
			ir << composite;
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
				return 1;
			}
			string tmp_main_output = (StringRef)tmp_main_vector;
			close(fd);
			tool_output_file tmp_main_object(tmp_main_output.c_str(), err, raw_fd_ostream::F_Binary);
			if (!err.empty())
			{
				cerr << "Cannot open output file" << tmp_main_output.c_str() << endl;
				return 1;
			}

			formatted_raw_ostream fos(tmp_main_object.os());
			if (machine.get()->addPassesToEmitFile(manager, fos,
		        TargetMachine::CGFT_ObjectFile, CodeGenOpt::None))
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
				return status;
			}

			// Swap tmp_main_output 1 and 2.
			string swap = tmp_main_output1;
			tmp_main_output1 = tmp_main_output2;
			tmp_main_output2 = swap;
		}
	}
	
	//
	// 7) Rename original main entry and insert new main
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
			return status;
		}
	}

	//
	// 8) Link code using the regular linker.
	//
	if (output)
	{
		// Use cloned main object instead of original one.
		vector<const char*> args;
		args.reserve(argc);
		for (int i = 0; argv[i]; i++)
		{
			if (!strcmp(argv[i], main_output)) {
				args.push_back(tmp_main_output1.c_str());
				continue;
			}
			args.push_back(argv[i]);
		}

		// Adding -rdynamic to use executable global symbols
		// to resolve dependencies of subsequently loaded kernel objects.
		args.push_back("-rdynamic");
		args.push_back("/opt/kernelgen/lib/libkernelgen-rt.so");
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
			cerr << err;
			return status;
		}
	}
	else
	{
		// When no output, kernelgen-simple acts as and LTO backend.
		// Here we need to output the list of objects collect2 will
		// pass to linker.
		if (nloops % 2) tmp_main_object1.keep();
		else tmp_main_object2.keep();
		for (int i = 1; argv[i]; i++)
		{
			string name = tmp_main_output1;

			// To be compatible with LTO, we need to clone all
			// objects, except the one containing main entry,
			// which is already clonned.
			if (strcmp(argv[i], main_output))
			{
		                tmp_main_vector.clear();
        		        if (unique_file(tmp_mask, fd, tmp_main_vector))
                		{
                        		cout << "Cannot generate temporary main object file name" << endl;
	                        	return 1;
	        	        }
        	        	name = (StringRef)tmp_main_vector;
	        	        close(fd);
		        	tool_output_file tmp_object(name.c_str(), err, raw_fd_ostream::F_Binary);
				if (!err.empty())
				{
					cerr << "Cannot open output file" << name.c_str() << endl;
					return 1;
				}
				tmp_object.keep();
				{
					vector<const char*> args;
					args.push_back(cp);
					args.push_back(argv[i]);
					args.push_back(name.c_str());
					args.push_back(NULL);
					if (verbose)
					{
						cout << args[0];
						for (int i = 1; args[i]; i++)
							cout << " " << args[i];
						cout << endl;
					}
					int status = Program::ExecuteAndWait(
						Program::FindProgramByName(cp), &args[0],
						NULL, NULL, 0, 0, &err);
					if (status)
					{
						cerr << err;
						return status;
					}
				}
			}

			// Remove the .kernelgen section from the clonned object.
			{
				vector<const char*> args;
				args.push_back(objcopy);
				args.push_back("--remove-section=.kernelgen");
				args.push_back(name.c_str());
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
					return status;
				}
			}

			cout << name << endl;
		}
	}
	
	return 0;
}

int main(int argc, char* argv[])
{
	llvm::PrettyStackTraceProgram X(argc, argv);

	// Behave like compiler if no arguments.
	if (argc == 1)
	{
		cout << "kernelgen: note \"simple\" is a development frontend not intended for regular use!" << endl;
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
	if (input && !output)
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
	if (input || output)
		cout << "kernelgen: note \"simple\" is a development frontend not intended for regular use!" << endl;

	fallback_args_t* fallback_args = new fallback_args_t();
	fallback_args->argc = argc;
	fallback_args->argv = argv;
	fallback_args->input = input;
	fallback_args->output = output;
	tracker = new PassTracker(input, &fallback, fallback_args);

	// Execute either compiler or linker.
	int result;
	if (input)
	{
		PluginLoader loader;
		loader.operator =("libkernelgen-opt.so");
		result = compile(argc, argv, input, output);
	}
	else
		result = link(argc, argv, input, output);
	delete tracker;
	return result;
}
