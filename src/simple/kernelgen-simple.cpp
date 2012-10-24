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
#include <sstream>
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
#include "llvm/Object/Archive.h"
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
#include "llvm/Transforms/Utils/ValueMapper.h"
#include "llvm/Support/MDBuilder.h"

#include "BranchedLoopExtractor.h"
#include "TrackedPassManager.h"

using namespace llvm;
using namespace llvm::object;
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
extern cl::opt<bool> EnablePRE;
extern cl::opt<bool> EnableLoadPRE;
extern cl::opt<bool> DisableLoadsDeletion;
extern cl::opt<bool> DisablePromotion;

namespace kernelgen
{
    void getAllDependencesForValue(llvm::GlobalValue * value, DepsByType & dependencesByType);
}


static void addKernelgenPasses(const PassManagerBuilder &Builder, PassManagerBase &PM)
{
	PM.add(createFixPointersPass());
	PM.add(createInstructionCombiningPass());
	PM.add(createMoveUpCastsPass());
	PM.add(createInstructionCombiningPass());
	PM.add(createBasicAliasAnalysisPass());
	PM.add(createGVNPass()); 
	//PM.add(createEarlyCSEPass());
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
		args.push_back("-fkeep-inline-functions");
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
        // 3) Extract loops into new functions. Apply some optimization
        // passes to the resulting module.
        //
	{
		PassManager manager;
		manager.add(new TargetData(m.get()));
		manager.add(createFixPointersPass());
		manager.add(createInstructionCombiningPass());
		manager.add(createMoveUpCastsPass());
		manager.add(createInstructionCombiningPass());
		manager.add(createEarlyCSEPass());
		manager.add(createCFGSimplificationPass());
		manager.run(*m);
	}
	{
		EnableLoadPRE.setValue(false);
		DisableLoadsDeletion.setValue(true);
		DisablePromotion.setValue(true);
		PassManager manager;
		manager.add(new TargetData(m.get()));
		manager.add(createBasicAliasAnalysisPass());
		manager.add(createLICMPass());
		manager.add(createGVNPass());
		manager.run(*m);
	}
	{
		PassManager manager;
		manager.add(createBranchedLoopExtractorPass());
		manager.add(createCFGSimplificationPass());
		manager.run(*m);
	}

	verifyModule(*m);
	if (verbose) m->dump();

	//
	// 4) Emit the resulting LLVM IR module into temporary
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

// Get the object data from the specified file.
// If the filename has offset suffix, it is removed on output.
int getArchiveObjData(string& filename, vector<char>& container, long offset = 0)
{
	// Get the object data in static library (archive) located
	// at the specified offset.
	if (offset)
	{
		// Open the archive and determine the size of member having the
		// specified offset.
		OwningPtr<MemoryBuffer> buffer;
		error_code ec = MemoryBuffer::getFile(filename, buffer);
		Archive archive(buffer.get(), ec);
		Archive::child_iterator* found_child = NULL;
		for (Archive::child_iterator AI = archive.begin_children(),
			AE = archive.end_children(); AI != AE; ++AI)
		{
			long current_offset = (ptrdiff_t)AI->getBuffer()->getBufferStart() -
				(ptrdiff_t)buffer->getBufferStart();
			if (current_offset == offset)
			{
				found_child = &AI;
				break;
			}
		}
		if (!found_child)
		{
			cerr << "Cannot find " << filename << " @" << offset << endl;
			return -1;
		}

		size_t size = (size_t)((*found_child)->getSize());
		container.resize(size + 1);
		memcpy((char*)&container[0],
			(*found_child)->getBuffer()->getBufferStart(), size);
		container[size] = '\0';

		buffer.take();

		return 0;
	}

	// For LTO static archive, support handling of input file specifications
	// that are composed of a filename and an offset like FNAME@OFFSET.
	int consumed;
	const char *p = strrchr(filename.c_str(), '@');
	if (p && (p != filename.c_str()) &&
		(sscanf(p, "@%li%n", &offset, &consumed) >= 1) &&
		(strlen (p) == (unsigned int)consumed))
	{
		filename = string(filename.c_str(), p - filename.c_str());

		// Only accept non-stdin and existing FNAME parts, otherwise
		// try with the full name.
		if (filename == "-") return -1;
	}
	
	// Check the file exists.
	if (access(filename.c_str(), F_OK) < 0)
		return -1;

	// Read the object into memory.
	if (p) return getArchiveObjData(filename, container, offset);

	// Create data container and open the object file.
	stringstream stream(stringstream::in | stringstream::out |
		stringstream::binary);
	ifstream f(filename.c_str(), ios::in | ios::binary);
	filebuf* buffer = f.rdbuf();

	// Get file size and load its data.
	size_t size = buffer->pubseekoff(0, ios::end,ios::in);
	buffer->pubseekpos(0, ios::in);
	container.resize(size + 1);
	buffer->sgetn((char*)&container[0], size);
	container[size] = '\0';
	
	f.close();

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
	string main_output = "";
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
	for (int i = 1; argv[i]; i++)
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

		if (verbose)
			cout << "Linking " << arg << " ..." << endl;

		// Load the object data into memory.
		vector<char> container;
		string filename = arg;
		if (getArchiveObjData(filename, container)) return -1;
		char* image = (char*)&container[0];

		// Check the ELF magic.
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
	if (main_output == "")
	{
		// In general case this is not an error.
		// Missing main entry only means we are linking
		// a library.
		cerr << "Cannot find object containing main entry" << endl;
		cerr << "Note kernelgen-simple only searches in objects!" << endl;
		return 1;
	}

	// Run -instcombine pass.
	{
		PassManager manager;
		manager.add(new TargetData(&composite));
		manager.add(createInstructionCombiningPass());
		manager.run(composite);
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


		// Replace all allocas by one big global variable
		Function* f = composite.getFunction("kernelgen_launch");
		if (f)
		{
			Module *module = &composite;
			
			//maximum size of aggregated structure with parameters
			unsigned long long maximumSizeOfData = 0;
			
			//list of allocas for aggregated structures with parameters
			list<AllocaInst *> allocasForArgs;
			//set<AllocaInst *> allocasForArgs;
			
			allocasForArgs.clear();
			Value * tmpArg = NULL;
			
			// Walk on all kernelgen_launch's users
			for (Value::use_iterator UI = f->use_begin(), UE = f->use_end(); UI != UE; UI++)
			{
				CallInst* call = dyn_cast<CallInst>(*UI);
				if (!call) continue;		
				
				// Retrive size of data
				tmpArg = call->getArgOperand(1);
				assert( isa<ConstantInt>(*tmpArg) && "by this time, after optimization,"
					"second parameter of kernelgen_launch "
					"must be ConstantInt");
				
				// Get maximum size of data
				uint64_t sizeOfData = ((ConstantInt*)tmpArg)->getZExtValue();
				if(maximumSizeOfData < sizeOfData)
				    maximumSizeOfData=sizeOfData;
				
				// Retrive allocas from kernelgen_launches
				tmpArg = call -> getArgOperand(3);
				assert(isa<BitCastInst>(*tmpArg) &&  "4th parameter of kernelgen_launch "
					"must be BitCast for int32 *");
				BitCastInst *castStructToPtr = (BitCastInst *)tmpArg;
				
				tmpArg = castStructToPtr->getOperand(0);
				assert(isa<AllocaInst>(*tmpArg) && "must be cast of AllocaInst's result");
				AllocaInst *allocaForArgs = (AllocaInst*)tmpArg;

				assert(allocaForArgs->getAllocatedType()->isStructTy() &&
					"must be allocation of structure for args");
				
				//store alloca
				allocasForArgs.push_back(allocaForArgs);
				//allocasForArgs.insert(allocaForArgs);
			}
			// allocate maximumSizeOfData of i8
			//AllocaInst *collectiveAlloca = new AllocaInst(Type::getInt8Ty(module->getContext()), 
			//                           ConstantInt::get(Type::getInt64Ty(module->getContext()),maximumSizeOfData),
			//						   8, "collectiveAllocaForArgs",
			//						   kernelgen_main_->begin()->getFirstNonPHI());
									   
			Type * allocatedType=ArrayType::get(Type::getInt8Ty(module->getContext()),maximumSizeOfData);
			// allocate array [i8 x maximumSizeOfData]
		   /*	AllocaInst *collectiveAlloca = new AllocaInst(
				allocatedType, maximumSizeOfData),
				"collectiveAllocaForArgs", kernelgen_main_->begin()->begin());*/
				
			GlobalVariable *collectiveAlloca = new GlobalVariable(
						*module, allocatedType,
                        false, GlobalValue::PrivateLinkage,
						Constant::getNullValue(allocatedType), "memoryForKernelArgs");
			collectiveAlloca->setAlignment(4096);
			
			// Walk on all stored allocas
			for(list<AllocaInst *>::iterator iter=allocasForArgs.begin(), iter_end=allocasForArgs.end();
			//for(set<AllocaInst *>::iterator iter=allocasForArgs.begin(), iter_end=allocasForArgs.end();
				iter!=iter_end;iter++ )
			{
				AllocaInst* allocaInst = *iter;

				// FIXME: This is a temporary workaround for an issue
				// spotted during COSMO linking: by some reason the same
				// allocaInst is accounted in list multiple times. This
				// check should bypass possible duplicates.
				if (!allocaInst->getParent()) continue;
				
				// Get type of old alloca
				Type* structPtrType = allocaInst -> getType();
				
				// Create bit cast of created alloca for specified type
				BitCastInst* bitcast = new BitCastInst(
					collectiveAlloca, structPtrType, "ptrToArgsStructure");
				// Insert after old alloca
				bitcast->insertAfter(allocaInst);
				// Replace uses of old alloca with create bit cast
				allocaInst->replaceAllUsesWith(bitcast);
				// Erase old alloca from parent basic block
				allocaInst->eraseFromParent();
			}
		}
		
		// Store addreses of all globals
		{
			Value *Idx3[1];
			Idx3[0] = ConstantInt::get(Type::getInt64Ty(context), 0);
			GetElementPtrInst *GEP3 = GetElementPtrInst::Create(arg, Idx3, "", root);
			Value* memory2 = new BitCastInst(GEP3,
				Type::getInt64PtrTy(context)->getPointerTo(0), "", root);
			LoadInst* memory3 = new LoadInst(memory2, "MemoryForGlobals", root);
			memory3->setAlignment(8);
			Value *Idx4[1];

			Value * MemoryForGlobals = memory3;
			Type * Int64Ty = Type::getInt64Ty(context);

			int i = 0;
			MDBuilder mdBuilder(context);
			NamedMDNode *namedMdNode = composite.getOrInsertNamedMetadata("OrderOfGlobals");
			assert(namedMdNode);

			for (Module::global_iterator iter=composite.global_begin(),
				iter_end=composite.global_end(); iter!=iter_end; iter++)
			{
				GlobalVariable *globalVar = iter;
				Idx4[0] = ConstantInt::get(Type::getInt64Ty(context), i);
                
				Value *vals[2];
				vals[0] = mdBuilder.createString(iter->getName());
				vals[1] = ConstantInt::get(Int64Ty, i);

				MDNode *mdNode = MDNode::get(context, vals);
				namedMdNode->addOperand(mdNode);

				GetElementPtrInst *placeOfGlobal = GetElementPtrInst::Create(
					memory3, Idx4, (string)"placeOf." + iter->getName(), root);
				Constant *bitCastOfGlobal = ConstantExpr::getPtrToInt(globalVar,Int64Ty);
				StoreInst *storeOfGlobal = new StoreInst(bitCastOfGlobal, placeOfGlobal, root);
				i++;
			}
		}
		
		// Create global variable with pointer to callback structure.
		GlobalVariable* callback1 = new GlobalVariable(
			composite, Type::getInt32PtrTy(context), false,
			GlobalValue::ExternalLinkage,
			Constant::getNullValue(Type::getInt32PtrTy(context)),
			"__kernelgen_callback", 0, false, 1);

		// Assign callback structure pointer with value received
		// from the arguments structure.
		// %struct.callback_t = type { i32, i32, i8*, i32, i8* }
		// %0 = getelementptr inbounds i32* %args, i64 2
		// %1 = bitcast i32* %0 to %struct.callback_t**
		// %2 = load %struct.callback_t** %1, align 8
		// %3 = getelementptr inbounds %struct.callback_t* %2, i64 0, i32 0
		// store i32* %3, i32** @__kernelgen_callback, align 8
		{
			Value *Idx3[1];
			Idx3[0] = ConstantInt::get(Type::getInt64Ty(context), 2);
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
			GlobalValue::ExternalLinkage,
			Constant::getNullValue(Type::getInt32PtrTy(context)),
			"__kernelgen_memory", 0, false, 1);

		// Assign memory structure pointer with value received
		// from the arguments structure.
		// %struct.memory_t = type { i8*, i64, i64, i64 }
		// %4 = getelementptr inbounds i32* %args, i64 4
		// %5 = bitcast i32* %4 to %struct.memory_t**
		// %6 = load %struct.memory_t** %5, align 8
		// %7 = bitcast %struct.memory_t* %6 to i32*
		// store i32* %7, i32** @__kernelgen_memory, align 8
		{
			Value *Idx3[1];
			Idx3[0] = ConstantInt::get(Type::getInt64Ty(context), 4);
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
			// Create and insert GEP to (int64*)args + 3.
			Value* Idx[] = { ConstantInt::get(Type::getInt64Ty(context), 6) };
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
			// Create and insert GEP to (int64*)args + 4.
			Value* Idx[] = { ConstantInt::get(Type::getInt64Ty(context), 8) };
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
		// Create and insert GEP to (int64*)(args) + 5.
		if (main_->getFunctionType()->getNumParams() == 3)
		{
			Value* Idx[] = { ConstantInt::get(Type::getInt64Ty(context), 10) };
			GetElementPtrInst* GEP = GetElementPtrInst::CreateInBounds(
					arg, Idx, "", root);

			// Bitcast (int8***)((int*)(args) + 4)).
			Value* envp1 = new BitCastInst(GEP, Type::getInt8Ty(context)->
					getPointerTo(0)->getPointerTo(0)->getPointerTo(0), "", root);

			// Load envp from int8***.
			LoadInst* envp2 = new LoadInst(envp1, "", root);
			envp2->setAlignment(1);

			call_args.push_back(envp2);
		}

		// Create a call to kernelgen_start() to begin execution.
		Function* start = Function::Create(TypeBuilder<void(), true>::get(context),
			GlobalValue::ExternalLinkage, "kernelgen_start", &composite);
		SmallVector<Value*, 16> start_args;
		CallInst* start_call = CallInst::Create(start, start_args, "", root);

		// Create a call to main_(int argc, char* argv[], char* envp[]).
		CallInst* call = CallInst::Create(main_, call_args, "", root);
		call->setTailCall();
		call->setDoesNotThrow();

		// Set return value, if present.
		if (!main_->getReturnType()->isVoidTy())
		{
			// Create and insert GEP to (int64*)args + 6.
			Value* Idx[] = { ConstantInt::get(Type::getInt64Ty(context), 12) };
			GetElementPtrInst* GEP = GetElementPtrInst::CreateInBounds(
				arg, Idx, "", root);

			// Store the call return value to ret.
			StoreInst* ret = new StoreInst(call, GEP, false, root);
			ret->setAlignment(1);
		}
		
		// Create a call to kernelgen_finish() to finalize execution.
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
				
				
				// Rename "loop" function to "__kernelgen_loop".
				func->setName("__kernelgen_" + func->getName());
				
				// Create new module and populate it with entire loop function.
				Module loop(func->getName(), context);
				loop.setTargetTriple(composite.getTargetTriple());
				loop.setDataLayout(composite.getDataLayout());
				loop.setModuleInlineAsm(composite.getModuleInlineAsm());

				// List of required functions
				DepsByType dependences;
				getAllDependencesForValue(func, dependences);
				
				// Map values from composite to new module
				ValueToValueMapTy VMap;
				
				// Copy all of the dependent libraries over.
				for (Module::lib_iterator I = composite.lib_begin(),
					E = composite.lib_end(); I != E; ++I)
					loop.addLibrary(*I);
				
				// Loop over all of the global variables, making corresponding
				// globals in the new module.  Here we add them to the VMap and
				// to the new Module.  We don't worry about attributes or initializers,
				// they will come later.
				for (variable_iter iter = dependences.variables.begin(),
					iter_end = dependences.variables.end(); iter != iter_end; ++iter)
				{
					GlobalVariable *I = *iter;

					GlobalVariable *GV = new GlobalVariable(loop,
						I->getType()->getElementType(),
						I->isConstant(), GlobalValue::ExternalLinkage,//I->getLinkage(),
						(Constant*) 0, I->getName(),
						(GlobalVariable*) 0,
						I->isThreadLocal(),
						I->getType()->getAddressSpace());
					
					GV->copyAttributesFrom(I);
					VMap[I] = GV;
				}

				// Loop over the functions in the module, making external functions as before.
				for (function_iter iter = dependences.functions.begin(),
					iter_end = dependences.functions.end(); iter != iter_end; ++iter)
				{
					Function *I = *iter;
					
					Function *NF = Function::Create(
						cast<FunctionType>(I->getType()->getElementType()),
						I->getLinkage(), I->getName(), &loop);
					NF->copyAttributesFrom(I);
					VMap[I] = NF;
				}

				// Loop over the aliases in the module.
				for (alias_iter iter = dependences.aliases.begin(),
					iter_end = dependences.aliases.end(); iter != iter_end; ++iter)
				{
					GlobalAlias *I = *iter;
					
					GlobalAlias *GA = new GlobalAlias(I->getType(), I->getLinkage(),
						I->getName(), NULL, &loop);
					GA->copyAttributesFrom(I);
					VMap[I] = GA;
				}

				// Now that all of the things that global variable initializer can refer to
				// have been created, loop through and copy the global variable referrers
				// over...  We also set the attributes on the global now.
				/*for (variable_iter iter = dependences.variables.begin(),
					iter_end = dependences.variables.end(); iter != iter_end; ++iter)
				{
					GlobalVariable *I = *iter;
					
					GlobalVariable *GV = cast<GlobalVariable>(VMap[I]);
					if (I->hasInitializer())
						GV->setInitializer(MapValue(I->getInitializer(), VMap));
				}*/

				// Similarly, copy over required function bodies now...
				for (function_iter iter = dependences.functions.begin(),
					iter_end = dependences.functions.end(); iter != iter_end; ++iter)
				{
					Function *I = *iter;
					Function *F = cast<Function>(VMap[I]);
					
					if (!I->isDeclaration())
					{
						Function::arg_iterator DestI = F->arg_begin();
						for (Function::const_arg_iterator J = I->arg_begin();
							J != I->arg_end(); ++J)
						{
							DestI->setName(J->getName());
							VMap[J] = DestI++;
						}
						
						SmallVector<ReturnInst*, 8> Returns;  // Ignore returns cloned.
						CloneFunctionInto(F, I, VMap, /*ModuleLevelChanges=*/true, Returns);
						
						for (Function::arg_iterator argI = I->arg_begin(),
							argE = I->arg_end(); argI != argE; ++argI)
							VMap.erase(argI);
					}
				}

				// And aliases.
				for (alias_iter iter = dependences.aliases.begin(),
					iter_end = dependences.aliases.end(); iter != iter_end; ++iter)
				{
					GlobalAlias *I = *iter;
					
					GlobalAlias *GA = cast<GlobalAlias>(VMap[I]);
					if (const Constant *C = I->getAliasee())
						GA->setAliasee(MapValue(C, VMap));
				}

				// And named metadata....
				// !!!!!!!!!!!!!!!
				// copy metadata??
				// !!!!!!!!!!!!!!!
				
				Function *newFunc = cast<Function>(VMap[func]);
				newFunc ->setName(newFunc->getName());
				
				// Defined functions will be deleted after inlining
				// if linkage type is LinkerPrivateLinkage.
				for (Module::iterator iter = loop.begin(), iter_end = loop.end();
					iter != iter_end; iter++)
						if(cast<Function>(iter) != newFunc)
						{
							if(!iter->isDeclaration())
								iter->setLinkage(GlobalValue::LinkerPrivateLinkage);
							else if(!iter->isIntrinsic())
								iter->setLinkage(GlobalValue::ExternalLinkage);
						}

				// Append "always inline" attribute to all existing functions
				// in loop module.
				for (Module::iterator f = loop.begin(), fe = loop.end(); f != fe; f++)
				{
					Function* func = f;
					if (func->isDeclaration()) continue;

					f->removeFnAttr(Attribute::NoInline);
					f->addFnAttr(Attribute::AlwaysInline);
				}
				
				verifyModule(loop);
				
				/* // Delete unused globals.
				for (Module::global_iterator iter = loop.global_begin(),
					iter_end = loop.global_end(); iter != iter_end; iter++)
					if(!iter->isDeclaration())
						iter->setLinkage(GlobalValue::LinkerPrivateLinkage);*/
					
				// Perform inlining (required by Polly).
				{
					PassManager manager;
					manager.add(createFunctionInliningPass());
					manager.add(createCFGSimplificationPass());
					manager.run(loop);
				}

				// Delete unnecessary globals and function declarations
				//{
				//	PassManager manager;
				//	manager.add(createGlobalOptimizerPass());     // Optimize out global vars
				//	manager.add(createStripDeadPrototypesPass()); // Get rid of dead prototypes
				//	manager.run(loop);
				//}
		       
				// Embed "loop" module into object or just issue
				// the temporary object in case of LTO.
				do
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

					tmp_main_vector.clear();
					if (unique_file(tmp_mask, fd, tmp_main_vector))
					{
						cout << "Cannot generate temporary main object file name" << endl;
						return 1;
					}
					string tmp_loop_output = (StringRef)tmp_main_vector;
					close(fd);
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

					// Just issue the temporary object in case of LTO.
					if (!output)
					{
						cout << tmp_loop_output.c_str() << endl;
						tmp_loop_object.keep();
						break;
					}

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
				while (0);

				func->eraseFromParent();
				nloops++;
			}
		}

	 	
		//TrackedPassManager manager(tracker);
		PassManager manager;
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
	// 6) Add wrapper around main to make it compatible with kernelgen_launch.
	//
	if (verbose)
		cout << "Extracting kernel main ..." << endl;
	{
		//TrackedPassManager manager(tracker);
		PassManager manager;
		manager.add(new TargetData(&composite));
		
		// Rename "main" to "__kernelgen_main".
		Function* kernelgen_main_ = composite.getFunction("main");
		
		list<Function *> functionsToDelete;
		for (Module::iterator iter = composite.begin(), iter_end = composite.end();
			iter != iter_end; iter++)
			if(cast<Function>(iter) != kernelgen_main_)
			{
				/*if(!iter->isDeclaration())
					iter->setLinkage(GlobalValue::LinkerPrivateLinkage);
				else if(!iter->isIntrinsic())
					iter->setLinkage(GlobalValue::ExternalLinkage);*/
				if(iter->getNumUses() == 0)
					functionsToDelete.push_back(iter);
			}
		
		for (list<Function *>::iterator iter = functionsToDelete.begin(),
			iter_end = functionsToDelete.end(); iter!=iter_end; iter++)
			(*iter)->eraseFromParent();
		
		verifyModule(composite);

		/*// Optimize only composite module with main function.
		{
			//TrackedPassManager manager(tracker);
			PassManager manager;
			manager.add(new TargetData(&composite));
			PassManagerBuilder builder;
			builder.Inliner = NULL;
			builder.OptLevel = 3;
			builder.SizeLevel = 3;
			builder.DisableSimplifyLibCalls = true;
			builder.populateModulePassManager(manager);
			manager.run(composite);
		}*/

		kernelgen_main_->setName("__kernelgen_main");

		// Add __kernelgen_regular_main reference.
		Function::Create(mainTy, GlobalValue::ExternalLinkage,
			"__kernelgen_regular_main", &composite);

		// Embed "main" module into main_output or just issue
		// the temporary object in case of LTO.
		do
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

			// Just issue the temporary object in case of LTO.
			if (!output)
			{
				cout << tmp_main_output.c_str() << endl;
				tmp_main_object.keep();
				break;
			}
			
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
		while (0);
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
			if (!strcmp(argv[i], main_output.c_str())) {
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
		// When no output, kernelgen-simple acts as an LTO backend.
		// Here we need to output objects collect2 will pass to linker.
		tmp_main_object1.keep();
		for (int i = 1; argv[i]; i++)
		{
			string filename = tmp_main_output1;

			// Copy existing objects to temporary files.
			if (strcmp(argv[i], main_output.c_str()))
			{
				// Get object data.
				filename = argv[i];
				vector<char> container;
				if (getArchiveObjData(filename, container)) return -1;
			
				tmp_main_vector.clear();
				if (unique_file(tmp_mask, fd, tmp_main_vector))
				{
					cout << "Cannot generate temporary main object file name" << endl;
					return 1;
				}
				filename = (StringRef)tmp_main_vector;
				tool_output_file tmp_object(filename.c_str(), err, raw_fd_ostream::F_Binary);
				if (!err.empty())
				{
					cerr << "Cannot open output file" << filename.c_str() << endl;
					return 1;
				}
				tmp_object.keep();

				// Copy the object data to the temporary file.
				write(fd, (char*)&container[0], container.size() - 1);
				close(fd);
			}

			// Remove .kernelgen section from each clonned object.
			{
				vector<const char*> args;
				args.push_back(objcopy);
				args.push_back("--remove-section=.kernelgen");
				args.push_back(filename.c_str());
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
			
			cout << filename << endl;
		}
	}
	
	return 0;
}

extern "C" void expandargv(int* argcp, char*** argvp);

int main(int argc, char* argv[])
{
	llvm::PrettyStackTraceProgram X(argc, argv);

	// Behave like compiler if no arguments.
	if (argc == 1)
	{
		cout << "kernelgen: note \"simple\" is a development " <<
			"frontend not intended for regular use!" << endl;
		cout << "kernelgen: no input files" << endl;
		return 0;
	}

	// Enable or disable verbose output.
	char* cverbose = getenv("kernelgen_verbose");
	if (cverbose) verbose = atoi(cverbose);

	// We may be called with all the arguments stored in some file and
	// passed with @file. Expand them into argv before processing.
	expandargv(&argc, &argv);

	if (argc == 1)
		return 0;
/*for (int i = 0; argv[i]; i++)
	fprintf(stderr, "%s ", argv[i]);
fprintf(stderr, "\n");*/

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
