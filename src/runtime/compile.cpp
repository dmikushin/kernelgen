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
#include "llvm/Support/PassManagerBuilder.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/ToolOutputFile.h"
#include "llvm/Support/TypeBuilder.h"
#include "llvm/Target/TargetData.h"
#include "llvm/Target/TargetOptions.h"
#include "llvm/Target/TargetRegistry.h"
#include "llvm/Target/TargetSelect.h"
#include "llvm/Transforms/Scalar.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include "llvm/Transforms/Utils/Cloning.h"
#include "llvm/Transforms/Utils/FunctionUtils.h"

#include "polly/LinkAllPasses.h"
#include "CodeGeneration.h"

#include "io.h"
#include "util.h"
#include "runtime.h"

#include <cstdlib>
#include <dlfcn.h>
#include <fstream>
#include <list>
#include <set>

using namespace kernelgen;
using namespace util::io;
using namespace llvm;
using namespace polly;
using namespace std;
namespace polly
{
extern cl::opt<bool> CUDA;
};
// Arrays and sets of KernelGen and CUDA intrinsics.
static string cuda_intrinsics[] = {
#include "cuda_intrinsics.h"
};
static string kernelgen_intrinsics[] = {
#include "kernelgen_intrinsics.h"
};
static set<string> kernelgen_intrinsics_set, cuda_intrinsics_set;

// Target machines for runmodes.
auto_ptr<TargetMachine> kernelgen::targets[KERNELGEN_RUNMODE_COUNT];

static PassManager getPollyPassManager(Module* m)
{
	PassManager polly;
	PassRegistry &Registry = *PassRegistry::getPassRegistry();
	initializeCore(Registry);
	initializeScalarOpts(Registry);
	initializeIPO(Registry);
	initializeAnalysis(Registry);
	initializeIPA(Registry);
	initializeTransformUtils(Registry);
	initializeInstCombine(Registry);
	initializeInstrumentation(Registry);
	initializeTarget(Registry);

	polly.add(new TargetData(m));
	polly.add(createBasicAliasAnalysisPass());	// -basicaa
	polly.add(createPromoteMemoryToRegisterPass());	// -mem2reg
	polly.add(createCFGSimplificationPass());	// -simplifycfg
	polly.add(createInstructionCombiningPass());	// -instcombine
	polly.add(createTailCallEliminationPass());	// -tailcallelim
	polly.add(createLoopSimplifyPass());		// -loop-simplify
	polly.add(createLCSSAPass());			// -lcssa
	polly.add(createLoopRotatePass());		// -loop-rotate
	polly.add(createLCSSAPass());			// -lcssa
	polly.add(createLoopUnswitchPass());		// -loop-unswitch
	polly.add(createInstructionCombiningPass());	// -instcombine
	polly.add(createLoopSimplifyPass());		// -loop-simplify
	polly.add(createLCSSAPass());			// -lcssa
	polly.add(createIndVarSimplifyPass());		// -indvars
	polly.add(createLoopDeletionPass());		// -loop-deletion
	polly.add(createInstructionCombiningPass());	// -instcombine
	polly.add(createCodePreperationPass());		// -polly-prepare
	polly.add(createRegionSimplifyPass());		// -polly-region-simplify
	polly.add(createIndVarSimplifyPass());		// -indvars
	polly.add(createBasicAliasAnalysisPass());	// -basicaa
	polly.add(createScheduleOptimizerPass());	// -polly-optimize-isl

	return polly;
}

void ConstantSubstitution( Function * func, void * args);

Pass* createSizeOfLoopsPass(vector<Size3> *memForSize3 = NULL);
Size3 convertLoopSizesToLaunchParameters(Size3 LoopSizes)
{
	int64_t sizes[3];
	int64_t launchParameters[3];
	sizes[0] = LoopSizes.x;
	sizes[1] = LoopSizes.y;
	sizes[2] = LoopSizes.z;
	int numberOfLoops=0;
	for(int i = 0; i < 3; i++)
		if(sizes[i] == -1) launchParameters[i] = 1;
		else numberOfLoops++;
	for(int i = 0; i < numberOfLoops; i++) {
		launchParameters[i] = sizes[numberOfLoops-i-1];
		if(launchParameters[i] == 0) launchParameters[i] = 1;
	}
	return Size3(launchParameters);
}
kernel_func_t kernelgen::runtime::compile(
    int runmode, kernel_t* kernel, Module* module, void * data, int szdatai)
{
	// Do not compile, if no source.
	if (kernel->source == "")
		return kernel->target[runmode].binary;

	Module* m = module;
	LLVMContext &context = getGlobalContext();
	if (!m) {
		// Load LLVM IR source into module.
		SMDiagnostic diag;
		MemoryBuffer* buffer = MemoryBuffer::getMemBuffer(kernel->source);
		m = ParseIR(buffer, diag, context);
		if (!m)
			THROW(kernel->name << ":" << diag.getLineNo() << ": " <<
			      diag.getLineContents() << ": " << diag.getMessage());
		m->setModuleIdentifier(kernel->name + "_module");
		kernel->module = m;

		//raw_os_ostream OS1(cout);
		//m->print(OS1, NULL);
		//OS1.flush()
		if(szdatai != 0)
			ConstantSubstitution( m -> begin(), data);
	}

	//m->dump();

	// Dump result of polly passes without codegen, if requested
	// for testing purposes.
	char* dump_polly = getenv("kernelgen_dump_polly");
	if (dump_polly) {
		std::auto_ptr<Module> m_clone;
		m_clone.reset(CloneModule(m));
		PassManager polly = getPollyPassManager(m_clone.get());
		polly.run(*m_clone.get());
		m_clone.get()->dump();
	}

	PassManager polly = getPollyPassManager(m);

	// Emit target assembly and binary image, depending
	// on runmode.
	switch (runmode) {
	case KERNELGEN_RUNMODE_NATIVE : {
		// Create target machine for NATIVE target and get its target data.
		if (!targets[KERNELGEN_RUNMODE_NATIVE].get()) {
			InitializeAllTargets();
			InitializeAllTargetMCs();
			InitializeAllAsmPrinters();
			InitializeAllAsmParsers();

			Triple triple(m->getTargetTriple());
			if (triple.getTriple().empty())
				triple.setTriple(sys::getHostTriple());
			string err;
			const Target* target = TargetRegistry::lookupTarget(triple.getTriple(), err);
			if (!target)
				THROW("Error auto-selecting target for module '" << err << "'." << endl <<
				      "Please use the -march option to explicitly pick a target.");
			targets[KERNELGEN_RUNMODE_NATIVE].reset(target->createTargetMachine(
			        triple.getTriple(), "", "", Reloc::PIC_, CodeModel::Default));
			if (!targets[KERNELGEN_RUNMODE_NATIVE].get())
				THROW("Could not allocate target machine");

			// Override default to generate verbose assembly.
			targets[KERNELGEN_RUNMODE_NATIVE].get()->setAsmVerbosityDefault(true);
		}

		// Apply the Polly codegen for native target.
		polly::CUDA.setValue(false);
		polly.add(polly::createCodeGenerationPass()); // -polly-codegen
		vector<Size3> sizes;
		if(kernel->name != "__kernelgen_main")
			polly.add(createSizeOfLoopsPass(&sizes));
		polly.run(*m);
		if(kernel->name != "__kernelgen_main") {
			if(sizes.size() == 0)
				cout << "    No Scops detected in kernel" << endl;
			else {
				Size3 SizeOfLoops = sizes[0]; // 3-dimensional
				// non-negative define sizes
				// if parallelized less than 3 loops then remaining will be -1
				// example:
				//for (c2=0;c2<=122;c2++) {
				//   for (c4=0;c4<=c2+13578;c4++) {
				//      Stmt_polly_stmt_4_cloned(c2,c4);
				//	}
				// }
				// SizeOfLoops : 123 13640 -1
				Size3 launchParameters = convertLoopSizesToLaunchParameters(SizeOfLoops);
				kernel->target[runmode].launchParameters = launchParameters;
			}

		}

		// Dump result of polly passes with codegen, if requested
		// for testing purposes.
		char* dump_pollygen = getenv("kernelgen_dump_pollygen");
		if (dump_pollygen) m->dump();

		const TargetData* tdata =
		    targets[KERNELGEN_RUNMODE_NATIVE].get()->getTargetData();
		PassManager manager;
		manager.add(new TargetData(*tdata));

		// Setup output stream.
		string bin_string;
		raw_string_ostream bin_stream(bin_string);
		formatted_raw_ostream stream(bin_stream);

		// Ask the target to add backend passes as necessary.
		if (targets[KERNELGEN_RUNMODE_NATIVE].get()->addPassesToEmitFile(manager, stream,
		        TargetMachine::CGFT_ObjectFile, CodeGenOpt::Aggressive))
			THROW("Target does not support generation of this file type");

		manager.run(*m);

		// Flush the resulting object binary to the
		// underlying string.
		stream.flush();

		//cout << bin_string;

		// Dump generated kernel object to first temporary file.
		cfiledesc tmp1 = cfiledesc::mktemp("/tmp/");
		{
			fstream tmp_stream;
			tmp_stream.open(tmp1.getFilename().c_str(),
			                fstream::binary | fstream::out | fstream::trunc);
			tmp_stream << bin_string;
			tmp_stream.close();
		}

		// Link first and second objects together into third one.
		cfiledesc tmp2 = cfiledesc::mktemp("/tmp/");
		{
			string linker = "ld";
			std::list<string> linker_args;
			linker_args.push_back("-shared");
			linker_args.push_back("-o");
			linker_args.push_back(tmp2.getFilename());
			linker_args.push_back(tmp1.getFilename());
			if (verbose) {
				cout << linker;
				for (std::list<string>::iterator it = linker_args.begin();
				     it != linker_args.end(); it++)
					cout << " " << *it;
				cout << endl;
			}
			execute(linker, linker_args, "", NULL, NULL);
		}

		// Do not return anything if module is explicitly specified.
		if (module) return NULL;

		// Load linked image and extract kernel entry point.
		void* handle = dlopen(tmp2.getFilename().c_str(),
		                      RTLD_NOW | RTLD_GLOBAL | RTLD_DEEPBIND);
		if (!handle)
			THROW("Cannot dlopen " << dlerror());

		kernel_func_t kernel_func = (kernel_func_t)dlsym(handle, kernel->name.c_str());
		if (!kernel_func)
			THROW("Cannot dlsym " << dlerror());

		if (verbose)
			cout << "Loaded '" << kernel->name << "' at: " << (void*)kernel_func << endl;

		return kernel_func;
	}
	case KERNELGEN_RUNMODE_CUDA : {
		// Apply the Polly codegen for native target.
		polly::CUDA.setValue(true);
		polly.add(polly::createCodeGenerationPass()); // -polly-codegenn
		vector<Size3> sizes;
		if(kernel->name != "__kernelgen_main")
			polly.add(createSizeOfLoopsPass(&sizes));
		polly.run(*m);
		if(kernel->name != "__kernelgen_main") {
			if(sizes.size() == 0)
				cout << "    No Scops detected in kernel" << endl;
			else {
				Size3 SizeOfLoops = sizes[0]; // 3-dimensional
				// non-negative define sizes
				// if parallelized less than 3 loops then remaining will be -1
				// example:
				//for (c2=0;c2<=122;c2++) {
				//   for (c4=0;c4<=c2+13578;c4++) {
				//      Stmt_polly_stmt_4_cloned(c2,c4);
				//	}
				// }
				// SizeOfLoops : 123 13640 -1
				Size3 launchParameters = convertLoopSizesToLaunchParameters(SizeOfLoops);
				kernel->target[runmode].launchParameters = launchParameters;
			}
		}

		// Initially, fill intrinsics tables.
		if (kernelgen_intrinsics_set.empty()) {
			kernelgen_intrinsics_set.insert(
			    kernelgen_intrinsics, kernelgen_intrinsics +
			    sizeof(kernelgen_intrinsics) / sizeof(string));
			cuda_intrinsics_set.insert(
			    cuda_intrinsics, cuda_intrinsics +
			    sizeof(cuda_intrinsics) / sizeof(string));
		}

		// Convert external functions CallInst-s into
		// host callback form. Do not convert CallInst-s
		// to device-resolvable intrinsics (syscalls and math).
		Function* f = m->getFunction(kernel->name);
		if (kernel->name == "__kernelgen_main") {
			vector<CallInst*> erase_calls;
			for (Function::iterator bb = f->begin(); bb != f->end(); bb++)
				for (BasicBlock::iterator ii = bb->begin(), ie = bb->end(); ii != ie; ii++) {
					// Check if instruction in focus is a call.
					CallInst* call = dyn_cast<CallInst>(cast<Value>(ii));
					if (!call) continue;

					// Check if function is called (needs -instcombine pass).
					Function* callee = call->getCalledFunction();
					if (!callee) continue;
					if (!callee->isDeclaration()) continue;
					if (callee->isIntrinsic()) continue;

					string name = callee->getName();

					// Check function is natively supported.
					bool native = false;
					set<string>::iterator i1 =
					    kernelgen_intrinsics_set.find(callee->getName());
					if (i1 != kernelgen_intrinsics_set.end()) {
						if (verbose)
							cout << "KernelGen native: " << name << endl;
						continue;
					}
					set<string>::iterator i2 =
					    cuda_intrinsics_set.find(callee->getName());
					if (i2 != cuda_intrinsics_set.end()) {
						if (verbose)
							cout << "CUDA native: " << name << endl;
						continue;
					}

					// Check if function is malloc or free.
					// In case it is, replace it with kernelgen_* variant.
					if ((name == "malloc") || (name == "free")) {
						string rename = "kernelgen_";
						rename += callee->getName();
						Function* replacement = m->getFunction(rename);
						if (!replacement) {
							replacement = Function::Create(callee->getFunctionType(),
							                               GlobalValue::ExternalLinkage, rename, m);
						}
						call->setCalledFunction(replacement);
						if (verbose)
							cout << "replacement: " << name << " -> " << rename << endl;
						continue;
					}

					// Also record hostcall to the kernels map.
					kernel_t* hostcall = kernels[name];
					if (!hostcall) {
						hostcall = new kernel_t();
						hostcall->name = name;
						hostcall->source = "";

						// No targets supported, except NATIVE.
						for (int i = 0; i < KERNELGEN_RUNMODE_COUNT; i++) {
							hostcall->target[i].supported = false;
							hostcall->target[i].binary = NULL;
						}
						hostcall->target[KERNELGEN_RUNMODE_NATIVE].supported = true;

						hostcall->target[runmode].monitor_kernel_stream =
						    kernel->target[runmode].monitor_kernel_stream;
						hostcall->module = kernel->module;

						kernels[name] = hostcall;
					}

					// Replace entire call with hostcall and set old
					// call for erasing.
					wrapCallIntoHostcall(call, hostcall);
					erase_calls.push_back(call);
				}

			for (vector<CallInst*>::iterator i = erase_calls.begin(),
			     ie = erase_calls.end(); i != ie; i++)
				(*i)->eraseFromParent();
		} else {
			// If the target kernel is loop, do not allow host calls in it.
			// Also do not allow malloc/free, probably support them later.
			// TODO: kernel *may* have kernelgen_launch called, but it must
			// always evaluate to -1.
			for (Function::iterator bb = f->begin(); bb != f->end(); bb++)
				for (BasicBlock::iterator ii = bb->begin(), ie = bb->end(); ii != ie; ii++) {
					// Check if instruction in focus is a call.
					CallInst* call = dyn_cast<CallInst>(cast<Value>(ii));
					if (!call) continue;

					// Check if function is called (needs -instcombine pass).
					Function* callee = call->getCalledFunction();
					if (!callee) continue;
					if (!callee->isDeclaration()) continue;
					if (callee->isIntrinsic()) continue;

					// Check function is natively supported.
					set<string>::iterator i1 =
					    kernelgen_intrinsics_set.find(callee->getName());
					if (i1 != kernelgen_intrinsics_set.end()) {
						if (verbose)
							cout << "KernelGen native: " << callee->getName().data() << endl;
						continue;
					}
					set<string>::iterator i2 =
					    cuda_intrinsics_set.find(callee->getName());
					if (i2 != cuda_intrinsics_set.end()) {
						if (verbose)
							cout << "CUDA native: " << callee->getName().data() << endl;
						continue;
					}

					// Loop contains non-native calls, and therefore
					// cannot be executed on GPU.
					kernel->target[runmode].supported = false;
					return NULL;
				}
		}

		// Optimize module.
		{
			PassManager manager;
			manager.add(createLowerSetJmpPass());
			PassManagerBuilder builder;
			builder.Inliner = createFunctionInliningPass();
			builder.OptLevel = 3;
			builder.DisableSimplifyLibCalls = true;
			builder.populateModulePassManager(manager);
			manager.run(*m);
		}

		if (verbose) m->dump();

		// Create target machine for CUDA target and get its target data.
		if (!targets[KERNELGEN_RUNMODE_CUDA].get()) {
			InitializeAllTargets();
			InitializeAllTargetMCs();
			InitializeAllAsmPrinters();
			InitializeAllAsmParsers();

			const Target* target = NULL;
			Triple triple(m->getTargetTriple());
			if (triple.getTriple().empty())
				triple.setTriple(sys::getHostTriple());
			for (TargetRegistry::iterator it = TargetRegistry::begin(),
			     ie = TargetRegistry::end(); it != ie; ++it) {
				if (!strcmp(it->getName(), "c")) {
					target = &*it;
					break;
				}
			}

			if (!target)
				THROW("LLVM is built without C Backend support");

			targets[KERNELGEN_RUNMODE_CUDA].reset(target->createTargetMachine(
			        triple.getTriple(), "", "", Reloc::PIC_, CodeModel::Default));
			if (!targets[KERNELGEN_RUNMODE_CUDA].get())
				THROW("Could not allocate target machine");

			// Override default to generate verbose assembly.
			targets[KERNELGEN_RUNMODE_CUDA].get()->setAsmVerbosityDefault(true);
		}

		PassManager manager;

		// Setup output stream.
		string bin_string;
		raw_string_ostream bin_stream(bin_string);
		formatted_raw_ostream stream(bin_stream);

		// Ask the target to add backend passes as necessary.
		if (targets[KERNELGEN_RUNMODE_CUDA].get()->addPassesToEmitFile(manager, stream,
		        TargetMachine::CGFT_AssemblyFile, CodeGenOpt::Aggressive))
			THROW("Target does not support generation of this file type");

		manager.run(*m);

		// Flush the resulting object binary to the
		// underlying string.
		stream.flush();

		if (verbose) cout << bin_string;

		return nvopencc(bin_string, kernel->name);

		break;
	}
	case KERNELGEN_RUNMODE_OPENCL : {
		THROW("Unsupported runmode" << runmode);
		break;
	}
	default :
		THROW("Unknown runmode " << runmode);
	}

	return NULL;
}
