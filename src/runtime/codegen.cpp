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

#include "io.h"
#include "util.h"
#include "runtime.h"

#include "cuda_dyloader.h"
#include "KernelGen.h"

#include "llvm/Module.h"
#include "llvm/PassManager.h"
#include "llvm/Target/TargetData.h"
#include "llvm/Target/TargetOptions.h"
#include "llvm/Support/FormattedStream.h"
#include "llvm/Support/Host.h"
#include "llvm/Support/Program.h"
#include "llvm/Support/raw_os_ostream.h"
#include "llvm/Support/TargetRegistry.h"
#include "llvm/Support/TargetSelect.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <elf.h>
#include <fcntl.h>
#include <fstream>
#include <gelf.h>
#include <link.h>
#include <map>
#include <sstream>
#include <vector>

using namespace kernelgen;
using namespace kernelgen::bind::cuda;
using namespace kernelgen::runtime;
using namespace kernelgen::utils;
using namespace llvm;
using namespace llvm::sys;
using namespace util::io;
using namespace std;

static bool debug = false;

// Target machines for runmodes.
auto_ptr<TargetMachine> kernelgen::targets[KERNELGEN_RUNMODE_COUNT];

// Function-address mapping of main kernel.
static map<string, size_t> funcmap;

// Export the function-address mapping from the given cubin image.
static void cubin_export_funcmap(const char* cubin,
		map<string, size_t>& funcmap) {
	int fd = -1;
	Elf* e = NULL;
	try {
		//
		// 1) First, load the ELF file.
		//
		if ((fd = open(cubin, O_RDWR)) < 0) {
			fprintf(stderr, "Cannot open file %s\n", cubin);
			throw;
		}

		if ((e = elf_begin(fd, ELF_C_RDWR, e)) == 0) {
			fprintf(stderr, "Cannot read ELF image from %s\n", cubin);
			throw;
		}

		//
		// 2) Find ELF section containing the symbol table and
		// load its data.
		//
		size_t shstrndx;
		if (elf_getshdrstrndx(e, &shstrndx)) {
			fprintf(stderr, "elf_getshdrstrndx() failed for %s: %s\n", cubin,
					elf_errmsg(-1));
			throw;
		}
		GElf_Shdr shdr;
		Elf_Data* symbols = NULL;
		int nsymbols = 0, nsections = 0;
		Elf_Scn* scn = elf_nextscn(e, NULL);
		for (int i = 1; scn != NULL;
				scn = elf_nextscn(e, scn), i++, nsections++) {
			if (!gelf_getshdr(scn, &shdr)) {
				fprintf(stderr, "gelf_getshdr() failed for %s: %s\n", cubin,
						elf_errmsg(-1));
				throw;
			}

			if (shdr.sh_type == SHT_SYMTAB) {
				symbols = elf_getdata(scn, NULL);
				if (!symbols) {
					fprintf(stderr, "elf_getdata() failed for %s: %s\n", cubin,
							elf_errmsg(-1));
					throw;
				}
				if (shdr.sh_entsize)
					nsymbols = shdr.sh_size / shdr.sh_entsize;
				break;
			}
		}
		if (!symbols) {
			fprintf(stderr, "Cannot find symbols table in %s\n", cubin);
			throw;
		}

		//
		// 3) Find function symbols and record them into map.
		//
		for (int isymbol = 0; isymbol < nsymbols; isymbol++) {
			GElf_Sym symbol;
			gelf_getsym(symbols, isymbol, &symbol);

			if (ELF32_ST_TYPE(symbol.st_info) != STT_FUNC)
				continue;

			char* name = elf_strptr(e, shdr.sh_link, symbol.st_name);
			funcmap.insert(pair<string, size_t>(name, symbol.st_value));

			//cout << name << " -> " << symbol.st_value << endl;
		}

		elf_end(e);
		close(fd);
		e = NULL;
	} catch (...) {
		if (e)
			elf_end(e);
		if (fd >= 0)
			close(fd);
		throw;
	}
}

// Import the function-address mapping to the given cubin image.
static void cubin_import_funcmap(const char* cubin,
		const map<string, size_t> funcmap) {
	// Retrieve the current cubin function-address mapping.
	map<string, size_t> funcmap_old;
	cubin_export_funcmap(cubin, funcmap_old);

#define MAX(a,b) ((a) > (b) ? (a) : (b))

	// Merge mappings.
	vector<pair<size_t, size_t> > addrmap;
	addrmap.resize(MAX(funcmap.size(), funcmap_old.size()));
	for (map<string, size_t>::iterator I1 = funcmap_old.begin(), IE1 =
			funcmap_old.end(); I1 != IE1; I1++) {
		pair<string, size_t> item_old = *I1;
		map<string, size_t>::const_iterator I2 = funcmap.find(item_old.first);
		if (I2 == funcmap.end())
			continue;
		pair<string, size_t> item = *I2;
		addrmap.push_back(pair<size_t, size_t>(item_old.second, item.second));

		//cout << item_old.second << " -> " << item.second << endl;
	}

	// TODO: call libasfermi to replace JCALs according to address-address mapping.
}

// Compile C source to x86 binary or PTX assembly,
// using the corresponding LLVM backends.
KernelFunc kernelgen::runtime::Codegen(int runmode, Kernel* kernel,
		Module* m) {
	// Codegen LLVM IR into PTX or host, depending on the runmode.
	string name = kernel->name;
	switch (runmode) {

	case KERNELGEN_RUNMODE_NATIVE: {

		// Create target machine for NATIVE target and get its target data.
		if (!targets[KERNELGEN_RUNMODE_NATIVE].get()) {
			InitializeAllTargets();
			InitializeAllTargetMCs();
			InitializeAllAsmPrinters();
			InitializeAllAsmParsers();

			Triple triple;
			triple.setTriple(sys::getDefaultTargetTriple());
			string err;
			TargetOptions options;
			const Target* target = TargetRegistry::lookupTarget(
					triple.getTriple(), err);
			if (!target)
				THROW(
						"Error auto-selecting target for module '" << err << "'." << endl << "Please use the -march option to explicitly pick a target.");
			targets[KERNELGEN_RUNMODE_NATIVE].reset(
					target->createTargetMachine(triple.getTriple(), "", "",
							options, Reloc::PIC_, CodeModel::Default));
			if (!targets[KERNELGEN_RUNMODE_NATIVE].get())
				THROW("Could not allocate target machine");

			// Override default to generate verbose assembly.
			targets[KERNELGEN_RUNMODE_NATIVE].get()->setAsmVerbosityDefault(
					true);
		}

		// Setup output stream.
		string bin_string;
		raw_string_ostream bin_stream(bin_string);
		formatted_raw_ostream bin_raw_stream(bin_stream);

		// Ask the target to add backend passes as necessary.
		PassManager manager;
		const TargetData* tdata =
				targets[KERNELGEN_RUNMODE_NATIVE].get()->getTargetData();
		manager.add(new TargetData(*tdata));
		if (targets[KERNELGEN_RUNMODE_NATIVE].get()->addPassesToEmitFile(
				manager, bin_raw_stream, TargetMachine::CGFT_ObjectFile,
				CodeGenOpt::Aggressive))
			THROW("Target does not support generation of this file type");
		manager.run(*m);

		// Flush the resulting object binary to the
		// underlying string.
		bin_raw_stream.flush();

		// Dump generated kernel object to first temporary file.
		TempFile tmp1 = Temp::getFile("%%%%%%%%.o");
		if (settings.getVerboseMode() != Verbose::Disable) tmp1.keep();
		tmp1.download(bin_string.c_str(), bin_string.size());

		// Link first and second objects together into third one.
		TempFile tmp2 = Temp::getFile("%%%%%%%%.so");
		if (settings.getVerboseMode() != Verbose::Disable) tmp2.keep();
		{
			int i = 0;
			vector<const char*> args;
			args.resize(6);
			args[i++] = "ld";
			args[i++] = "-shared";
			args[i++] = "-o";
			args[i++] = tmp2.getName().c_str();
			args[i++] = tmp1.getName().c_str();
			args[i++] = NULL;
			string err;
			VERBOSE(args);
			int status = Program::ExecuteAndWait(Program::FindProgramByName(args[0]),
					&args[0], NULL, NULL, 0, 0, &err);
			if (status) {
				cerr << err;
				exit(1);
			}
		}

		// Load linked image and extract kernel entry point.
		void* handle = dlopen(tmp2.getName().c_str(),
				RTLD_NOW | RTLD_GLOBAL | RTLD_DEEPBIND);

		if (!handle)
			THROW("Cannot dlopen " << dlerror());

		KernelFunc kernel_func = (KernelFunc) dlsym(handle, name.c_str());
		if (!kernel_func)
			THROW("Cannot dlsym " << dlerror());

		VERBOSE("Loaded '" << name << "' at: " << (void*)kernel_func << "\n");

		return kernel_func;
	}

	case KERNELGEN_RUNMODE_CUDA: {

		int device;
		CUresult err = cuDeviceGet(&device, 0);
		if (err)
			THROW("Error in cuDeviceGet " << err);

		int major = 2, minor = 0;
		err = cuDeviceComputeCapability(&major, &minor, device);
		if (err)
			THROW("Cannot get the CUDA device compute capability" << err);

		// Create target machine for CUDA target and get its target data.
		if (!targets[KERNELGEN_RUNMODE_CUDA].get()) {
			InitializeAllTargets();
			InitializeAllTargetMCs();
			InitializeAllAsmPrinters();
			InitializeAllAsmParsers();

			const Target* target = NULL;
			Triple triple(m->getTargetTriple());
			if (triple.getTriple().empty())
				triple.setTriple(sys::getDefaultTargetTriple());
			for (TargetRegistry::iterator it = TargetRegistry::begin(), ie =
					TargetRegistry::end(); it != ie; ++it) {
				if (!strcmp(it->getName(), "nvptx64")) {
					target = &*it;
					break;
				}
			}

			if (!target)
				THROW("LLVM is built without NVPTX Backend support");

			stringstream sarch;
			sarch << "sm_" << (major * 10 + minor);
			targets[KERNELGEN_RUNMODE_CUDA].reset(
					target->createTargetMachine(triple.getTriple(), sarch.str(),
							"", TargetOptions(), Reloc::PIC_,
							CodeModel::Default, CodeGenOpt::Aggressive));
			if (!targets[KERNELGEN_RUNMODE_CUDA].get())
				THROW("Could not allocate target machine");

			// Override default to generate verbose assembly.
			targets[KERNELGEN_RUNMODE_CUDA].get()->setAsmVerbosityDefault(true);
		}

		// Setup output stream.
		string ptx_string;
		raw_string_ostream ptx_stream(ptx_string);
		formatted_raw_ostream ptx_raw_stream(ptx_stream);

		// Ask the target to add backend passes as necessary.
		PassManager manager;
		const TargetData* tdata =
				targets[KERNELGEN_RUNMODE_CUDA].get()->getTargetData();
		manager.add(new TargetData(*tdata));
		if (targets[KERNELGEN_RUNMODE_CUDA].get()->addPassesToEmitFile(manager,
				ptx_raw_stream, TargetMachine::CGFT_AssemblyFile,
				CodeGenOpt::Aggressive))
			THROW("Target does not support generation of this file type");
		manager.run(*m);

		// Flush the resulting object binary to the
		// underlying string.
		ptx_raw_stream.flush();

		VERBOSE(Verbose::Sources << ptx_string << "\n" << Verbose::Default);

		// Dump generated kernel object to first temporary file.
		TempFile tmp2 = Temp::getFile("%%%%%%%%.ptx");
		if (settings.getVerboseMode() != Verbose::Disable) tmp2.keep();
		tmp2.download(ptx_string.c_str(), ptx_string.size());

		// Compile PTX code in temporary file to CUBIN.
		TempFile tmp3 = Temp::getFile("%%%%%%%%.cubin");
		if (settings.getVerboseMode() != Verbose::Disable) tmp3.keep();
		{
			int i = 0;
			vector<const char*> args;
			args.resize(14);
			args[i++] = "ptxas";
			if (settings.getVerboseMode() != Verbose::Disable)
				args[i++] = "-v";
			stringstream sarch;
			sarch << "-arch=sm_" << (major * 10 + minor);
			string arch = sarch.str();
			args[i++] = arch.c_str();
			args[i++] = "-m64";
			args[i++] = tmp2.getName().c_str();
			args[i++] = "-o";
			args[i++] = tmp3.getName().c_str();
			args[i++] = "--cloning=no";

			const char* __maxrregcount = "--maxrregcount";
			string maxrregcount;
			if (name == "__kernelgen_main") {
				// Create a relocatable cubin, to be later linked
				// with dyloader cubin.
				// ptxas_args.push_back("--compile-only");
			} else {
				// Calculate and apply the maximum register count
				// constraint, depending on used compute grid dimensions.
				// TODO This constraint is here due to chicken&egg problem:
				// grid dimensions are chosen before the register count
				// becomes known. This thing should go away, once we get
				// some time to work on it.
				typedef struct {
					int maxThreadsPerBlock;
					int maxThreadsDim[3];
					int maxGridSize[3];
					int sharedMemPerBlock;
					int totalConstantMemory;
					int SIMDWidth;
					int memPitch;
					int regsPerBlock;
					int clockRate;
					int textureAlign;
				} CUdevprop;

				CUdevprop props;
				err = cuDeviceGetProperties((void*) &props, device);
				if (err)
					THROW("Error in cuDeviceGetProperties " << err);

				dim3 blockDim = kernel->target[runmode].blockDim;
				int maxregcount = props.regsPerBlock
						/ (blockDim.x * blockDim.y * blockDim.z) - 4;
				if ((major == 3) && (minor >= 5)) {
					if (maxregcount > 128)
						maxregcount = 128;
				} else {
					if (maxregcount > 63)
						maxregcount = 63;
				}

				args[i++] = __maxrregcount;
				stringstream smaxrregcount;
				smaxrregcount << maxregcount;
				maxrregcount = smaxrregcount.str();
				args[i++] = maxrregcount.c_str();
			}

			const char* _g = "-g";
			const char* __return_at_end = "--return-at-end";
			const char* __dont_merge_basicblocks = "--dont-merge-basicblocks";
			if (::debug) {
				args[i++] = _g;
				args[i++] = __return_at_end;
				args[i++] = __dont_merge_basicblocks;
			}
			args[i++] = NULL;

			string err;
			VERBOSE(args);
			int status = Program::ExecuteAndWait(Program::FindProgramByName(args[0]),
					&args[0], NULL, NULL, 0, 0, &err);
			if (status) {
				cerr << err;
				exit(1);
			}
		}

		if (name == "__kernelgen_main") {
			// Initialize the dynamic kernels loader.
			int err = cudyInit(&cuda_context->loader, cuda_context->capacity, tmp3.getName());
			if (err)
				THROW("Cannot initialize the dynamic loader " << err);

			// Align main kernel cubin global data to the virtual memory
			// page boundary.
			CUBIN::AlignData(tmp3.getName().c_str(), 4096);

			// Export main kernel cubin function-address map.
			cubin_export_funcmap(tmp3.getName().c_str(), funcmap);
		} else {
			// Import main kernel cubin function-address map.
			cubin_import_funcmap(tmp3.getName().c_str(), funcmap);
		}

		// Dump Fermi assembly from CUBIN.
		if (settings.getVerboseMode() & Verbose::ISA) {
			int i = 0;
			vector<const char*> args;
			args.resize(4);
			args[i++] = "cuobjdump";
			args[i++] = "-sass";
			args[i++] = tmp3.getName().c_str();
			args[i++] = NULL;

			string err;
			VERBOSE(args);
			int status = Program::ExecuteAndWait(Program::FindProgramByName(args[0]),
					&args[0], NULL, NULL, 0, 0, &err);
			if (status) {
				cerr << err;
				exit(1);
			}
		}

		// Load CUBIN into string.
		string cubin;
		{
			std::ifstream tmp_stream(tmp3.getName().c_str());
			tmp_stream.seekg(0, std::ios::end);
			cubin.reserve(tmp_stream.tellg());
			tmp_stream.seekg(0, std::ios::beg);

			cubin.assign((std::istreambuf_iterator<char>(tmp_stream)),
					std::istreambuf_iterator<char>());
			tmp_stream.close();
		}

		CUfunction kernel_func = NULL;
		if (name == "__kernelgen_main") {
			// Load CUBIN from string into module.
			CUmodule module;
			int err = cuModuleLoad(&module, tmp3.getName().c_str());
			if (err)
				THROW("Error in cuModuleLoadData " << err);

			err = cuModuleGetFunction(&kernel_func, module, name.c_str());
			if (err)
				THROW("Error in cuModuleGetFunction " << err);
		} else {
			// FIXME: The following ugly fix mimics the backend asm printer
			// mangler behavior. We should instead get names from the real
			// mangler, but currently it is unclear how to instantiate it,
			// since it needs MCContext, which is not available here.
			string dot = "2E_";
			for (size_t index = name.find(".", 0);
					index = name.find(".", index); index++) {
				if (index == string::npos)
					break;
				name.replace(index, 1, "_");
				name.insert(index + 1, dot);
			}

			// Load kernel function from the binary opcodes.
			CUstream stream = kernel->target[runmode].MonitorStream;
			CUresult err = cudyLoadCubin((CUDYfunction*) &kernel_func,
					cuda_context->loader, (char*) tmp3.getName().c_str(),
					name.c_str(), stream);
			if (err)
				THROW("Error in cudyLoadCubin " << err);
		}

		VERBOSE("Loaded '" << name << "' at: " << kernel_func << "\n");

		return (KernelFunc) kernel_func;
	}
	}
}

