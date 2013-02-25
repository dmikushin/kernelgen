//===- CodeGen.cpp - Kernels target code generation API -------------------===//
//
//     KernelGen -- A prototype of LLVM-based auto-parallelizing Fortran/C
//        compiler for NVIDIA GPUs, targeting numerical modeling code.
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements target code generation for supported architectures.
//
//===----------------------------------------------------------------------===//

#include "Runtime.h"
#include "cuda_dyloader.h"
#include "KernelGen.h"
#include "Platform.h"

#include "llvm/Module.h"
#include "llvm/PassManager.h"
#include "llvm/MC/MCSymbol.h"
#include "llvm/Target/TargetData.h"
#include "llvm/Support/FormattedStream.h"
#include "llvm/Support/Program.h"
#include "llvm/Support/raw_os_ostream.h"

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
using namespace std;

static bool debug = false;

// The filename of the temp file containing main kernel.
static string kernelgen_main_filename;

// The position of the LEPC instruction to account when
// computing the starting address of __kernelgen_main code region.
static unsigned int kernelgen_main_lepc_offset;

// Compile C source to x86 binary or PTX assembly,
// using the corresponding LLVM backends.
KernelFunc kernelgen::runtime::Codegen(int runmode, Kernel* kernel,
		Module* m) {
	// Codegen LLVM IR into PTX or host, depending on the runmode.
	string name = kernel->name;
	switch (runmode) {

	case KERNELGEN_RUNMODE_NATIVE: {
		// Setup output stream.
		string bin_string;
		raw_string_ostream bin_stream(bin_string);
		formatted_raw_ostream bin_raw_stream(bin_stream);

		// Ask the target to add backend passes as necessary.
		PassManager manager;
		const TargetData* tdata =
				platforms[KERNELGEN_RUNMODE_NATIVE]->machine->getTargetData();
		manager.add(new TargetData(*tdata));
		if (platforms[KERNELGEN_RUNMODE_NATIVE]->machine->addPassesToEmitFile(
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

		// Translate name into mangled name.
		StringRef mangledName =
				platforms[KERNELGEN_RUNMODE_NATIVE]->mangler.get()->getSymbol(
						m->getFunction(name))->getName();
		name = string(mangledName.data(), mangledName.size());

		KernelFunc kernel_func = (KernelFunc) dlsym(handle, name.c_str());
		if (!kernel_func)
			THROW("Cannot dlsym " << dlerror());

		VERBOSE("Loaded '" << name << "' at: " << (void*)kernel_func << "\n");

		return kernel_func;
	}

	case KERNELGEN_RUNMODE_CUDA: {
		// Setup output stream.
		string ptx_string;
		raw_string_ostream ptx_stream(ptx_string);
		formatted_raw_ostream ptx_raw_stream(ptx_stream);

		// Ask the target to add backend passes as necessary.
		PassManager manager;
		const TargetData* tdata =
				platforms[KERNELGEN_RUNMODE_CUDA]->machine->getTargetData();
		manager.add(new TargetData(*tdata));
		if (platforms[KERNELGEN_RUNMODE_CUDA]->machine->addPassesToEmitFile(manager,
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
			args[i++] = cuda_context->getPtxAS().c_str();
			if (settings.getVerboseMode() != Verbose::Disable)
				args[i++] = "-v";
			stringstream sarch;
			sarch << "-arch=" << cuda_context->getSubarch();
			string arch = sarch.str();
			args[i++] = arch.c_str();
			args[i++] = "-m64";
			args[i++] = tmp2.getName().c_str();
			args[i++] = "-o";
			args[i++] = tmp3.getName().c_str();
			args[i++] = "--cloning=no";

			const char* __maxrregcount = "--maxrregcount";
			string maxrregcount;
			if (name != "__kernelgen_main") {
				// Calculate and apply the maximum register count
				// constraint, depending on used compute grid dimensions.
				// TODO This constraint is here due to chicken&egg problem:
				// grid dimensions are chosen before the register count
				// becomes known. This thing should go away, once we get
				// some time to work on it.
				dim3 blockDim = kernel->target[runmode].blockDim;
				int maxregcount = cuda_context->getRegsPerBlock() /
						(blockDim.x * blockDim.y * blockDim.z) - 4;
				if ((cuda_context->getSubarchMajor() == 3) &&
						(cuda_context->getSubarchMinor() >= 5)) {
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
			CU_SAFE_CALL(cudyInit(&cuda_context->loader, cuda_context->capacity, tmp3.getName()));

			// Align main kernel cubin global data to the virtual memory
			// page boundary.
			CUBIN::AlignData(tmp3.getName().c_str(), 4096);

			// Insert commands to perform LEPC reporting.
			kernelgen_main_lepc_offset = CUBIN::InsertLEPCReporter(
					tmp3.getName().c_str(), "__kernelgen_main");

			kernelgen_main_filename = tmp3.getName();
		} else {
			// Check if loop kernel contains unresolved calls and resolve them
			// using the load-effective layout obtained from the main kernel.
			CUBIN::ResolveExternalCalls(
					tmp3.getName().c_str(), kernel->name.c_str(),
					kernelgen_main_filename.c_str(), "__kernelgen_main",
					kernelgen_main_lepc_offset);
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

		// Translate name into mangled name.
		StringRef mangledName =
				platforms[KERNELGEN_RUNMODE_CUDA]->mangler.get()->getSymbol(
						m->getFunction(name))->getName();
		name = string(mangledName.data(), mangledName.size());

		CUfunction kernel_func = NULL;
		if (name == "__kernelgen_main") {
			// Load CUBIN from string into module.
			CUmodule module;
			CU_SAFE_CALL(cuModuleLoad(&module, tmp3.getName().c_str()));

			// Load function responsible for GPU-side memcpy.
			CU_SAFE_CALL(cuModuleGetFunction((CUfunction*)&cuda_context->kernelgen_memcpy,
					module, "kernelgen_memcpy"));

			CU_SAFE_CALL(cuModuleGetFunction(&kernel_func, module, name.c_str()));
		} else {
			// Load kernel function from the binary opcodes.
			CU_SAFE_CALL(cudyLoadCubin((CUDYfunction*) &kernel_func,
					cuda_context->loader, name.c_str(),
					(char*)tmp3.getName().c_str(), cuda_context->getSecondaryStream()));
		}

		VERBOSE("Loaded '" << name << "' at: " << kernel_func << "\n");

		return (KernelFunc) kernel_func;
	}
	}
}
