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

#include "polly/LinkAllPasses.h"
#include "polly/RegisterPasses.h"

#include "io.h"
#include "util.h"
#include "runtime.h"

#include <math.h>
#include <cstdlib>
#include <dlfcn.h>
#include <fstream>
#include <list>
#include <set>
#include <stdio.h>

using namespace kernelgen;
using namespace util::io;
using namespace llvm;
using namespace polly;
using namespace std;
namespace polly
{
	extern cl::opt<bool> CUDA;
};
namespace llvm
{
	void RemoveStatistics();
}
extern cl::opt<bool>  IgnoreAliasing;
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
	polly.add(new TargetData(m));
	registerPollyPreoptPasses(polly);
	polly.add(polly::createIslScheduleOptimizerPass());
	return polly;
}

void ConstantSubstitution( Function * func, void * args);
Pass* createSizeOfLoopsPass(vector<Size3> *memForSize3 = NULL);
Pass* createRuntimeAliasAnalysisPass();
Size3 convertLoopSizesToLaunchParameters(Size3 LoopSizes)
{
	int64_t sizes[3];
	int64_t launchParameters[3];
	LoopSizes.writeToArray(sizes);
	int numberOfLoops = LoopSizes.getNumOfDimensions();
	for(int i = 0; i < numberOfLoops; i++) {
		launchParameters[i] = sizes[numberOfLoops-i-1];
		if(launchParameters[i] == 0) launchParameters[i] = 1;
	}
	for(int i = numberOfLoops; i < 3; i++) {
		launchParameters[i] = 1;
	}
	return Size3(launchParameters);
}
void printSpecifiedStatistics(vector<string> statisticsNames)
{
	string allStatistics;
	raw_string_ostream stringOS(allStatistics);
	llvm::PrintStatistics(stringOS);
	  outs().changeColor(raw_ostream::YELLOW);
	for(int i = 0; i < statisticsNames.size(); i++) {
		string statisticName = statisticsNames[i];
		int start = 0;
		int end = 0;
		while( (start = allStatistics.find(statisticName,end)) != -1) {
			start = allStatistics.rfind('\n',start);
			if(start == -1) start == 0;
			end = allStatistics.find('\n',start+1);
			outs() << allStatistics.substr(start+1,end-start);
		}
	}
	outs().resetColor();
}
static void runPolly(kernel_t *kernel, Size3 *sizeOfLoops,bool mode)
{
	llvm::EnableStatistics();
	PassManager polly = getPollyPassManager(kernel->module);
	polly.run(*kernel->module);
	llvm::RemoveStatistics();
	
	IgnoreAliasing.setValue(true);
	polly::CUDA.setValue(mode);

	llvm::EnableStatistics();
	bool debug = ::llvm::DebugFlag;
	::llvm::DebugFlag = true;
	vector<Size3> sizes;
	{
		PassManager polly;
		polly.add(new TargetData(kernel->module));
		//registerPollyPreoptPasses(polly);
		if(kernel->name != "__kernelgen_main") {
			polly.add(createRuntimeAliasAnalysisPass());
			polly.add(createSizeOfLoopsPass(&sizes));
		}
		polly.add(polly::createCodeGenerationPass()); // -polly-codegenn
		polly.add(createCFGSimplificationPass());
		polly.run(*kernel->module);
	}
	if (kernel->name != "__kernelgen_main") {
		Size3 SizeOfLoops;
		if(sizes.size() == 0)
		{
	        outs().changeColor(raw_ostream::RED);
			outs() << "\n    FAIL: No Valid Scops detected in kernel!!!\n\n";
			outs().resetColor();
		}
		else {
			// non-negative define sizes
			// if parallelized less than 3 loops then remaining will be -1
			// example:
			//for (c2=0;c2<=122;c2++) {
			//   for (c4=0;c4<=c2+13578;c4++) {
			//      Stmt_polly_stmt_4_cloned(c2,c4);
			//	}
			// }
			// SizeOfLoops : 123 13640 -1
			SizeOfLoops  = sizes[0]; // 3-dimensional
			if(sizeOfLoops) *sizeOfLoops=SizeOfLoops;
		}

	}
	vector<string> statisticsNames;
	statisticsNames.push_back("polly-detect");
	statisticsNames.push_back("runtime-AA");
	printSpecifiedStatistics(statisticsNames);
	llvm::RemoveStatistics();
	::llvm::DebugFlag = debug;
	
        if (verbose) {
		outs().changeColor(raw_ostream::BLUE);
		outs() << "<------------------ "<< kernel->name << ": compile completed ------------------->\n\n";
		outs().resetColor();
	}
}
static void runPollyNATIVE(kernel_t *kernel, Size3 *sizeOfLoops)
{
	return runPolly(kernel, sizeOfLoops, false);
}
static void runPollyCUDA(kernel_t *kernel, Size3 *sizeOfLoops)
{
	return runPolly(kernel, sizeOfLoops, true);
}
void substituteGridParams( kernel_t* kernel,dim3 & gridDim, dim3 & blockDim)
{
	Module * m = kernel-> module;
	vector<string> dimensions, parameters;

	dimensions.push_back("x");
	dimensions.push_back("y");
	dimensions.push_back("z");

	parameters.push_back("gridDim");
	parameters.push_back("blockDim");

	string prefix1("kernelgen_");
	string prefix2("_");
	string prefix3(".");

	unsigned int * gridParams[2];
	gridParams[0] = (unsigned int *)&gridDim;
	gridParams[1] = (unsigned int *)&blockDim;

	for(int parameter =0; parameter < parameters.size(); parameter++)
		for(int dimension=0; dimension < dimensions.size(); dimension++) {
			string functionName = prefix1 + parameters[parameter] + prefix2 + dimensions[dimension];
			Function* function = m->getFunction(functionName);
			if(function && function->getNumUses() > 0) {
				assert(function->getNumUses() == 1);
				CallInst *functionCall;
				assert(functionCall = dyn_cast<CallInst>(*(function -> use_begin())));
				functionCall->replaceAllUsesWith(
				    ConstantInt::get(cast<IntegerType>(function->getReturnType()),
				                     (uint64_t)gridParams[parameter][dimension]));
				functionCall->eraseFromParent();
			}
		}
}

// Loop unrolling with runtime part enabled
// (currently not used, final unrollment gives better
// result by some reason).
static void addRuntimeLoopUnrollPass(const PassManagerBuilder &Builder, PassManagerBase &PM)
{
	PM.add(createLoopUnrollPass(128, -1, 1));
}

kernel_func_t kernelgen::runtime::compile(
    int runmode, kernel_t* kernel, Module* module, void * data, int szdatai)
{
	// Do not compile, if no source.
	if (kernel->source == "")
		return kernel->target[runmode].binary;

        if(verbose)
		{
			outs().changeColor(raw_ostream::BLUE);
		    outs() << "\n<------------------ "<< kernel->name << ": compile started --------------------->\n";
			outs().resetColor();
		}
		
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

		if (szdatai != 0) {
			Function* f = m->getFunction(kernel->name);
			if (kernel->name != "__kernelgen_main")
			{
				ConstantSubstitution(f, data);
			}
		}
	}

   	if (verbose & KERNELGEN_VERBOSE_SOURCES) m->dump();
    
	PassManager polly = getPollyPassManager(m);

	// Emit target assembly and binary image, depending
	// on runmode.
	switch (runmode) {
	case KERNELGEN_RUNMODE_NATIVE : {
		if (kernel->name != "__kernelgen_main") {
			// Apply the Polly codegen for native target.
			// To perform this analysis, require the native runmode to be
			// used globally for entire app, i.e. not a fallback from
			// non-portable GPU kernel or hostcall.
			if (runmode == kernelgen::runmode)
			{
				Size3 sizeOfLoops;
				runPollyNATIVE(kernel, &sizeOfLoops);

				// Do not compile the loop kernel if no grid detected.
				// Important to place this condition *after* hostcalls check
				// above to generate full host kernel in case hostcalls are around
				// (.supported flag), rather than running kernel on device and invoke
				// hostcalls on each individual iteration, which will be terribly slow.
				if (sizeOfLoops.x == -1) return NULL;
			}

			// Optimize module.
			{
				PassManager manager;
				PassManagerBuilder builder;
				builder.Inliner = createFunctionInliningPass();
				builder.OptLevel = 3;
				builder.DisableSimplifyLibCalls = true;
				builder.populateModulePassManager(manager);
				manager.run(*m);
			}
		}

		if (verbose & KERNELGEN_VERBOSE_SOURCES) m->dump();

		// Create target machine for NATIVE target and get its target data.
		if (!targets[KERNELGEN_RUNMODE_NATIVE].get()) {
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
				THROW("Error auto-selecting target for module '" << err << "'." << endl <<
				      "Please use the -march option to explicitly pick a target.");
			targets[KERNELGEN_RUNMODE_NATIVE].reset(target->createTargetMachine(
			        triple.getTriple(), "", "", options, Reloc::PIC_, CodeModel::Default));
			if (!targets[KERNELGEN_RUNMODE_NATIVE].get())
				THROW("Could not allocate target machine");

			// Override default to generate verbose assembly.
			targets[KERNELGEN_RUNMODE_NATIVE].get()->setAsmVerbosityDefault(true);
		}

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
		dim3 blockDim(1,1,1);
		dim3 gridDim(1,1,1);
#define BLOCK_SIZE 1024
#define BLOCK_DIM_X 32
		// Apply the Polly codegen for native target.
		Size3 sizeOfLoops;
		if (kernel->name != "__kernelgen_main")  {
			runPollyCUDA(kernel, &sizeOfLoops);

			//   x   y     z       x     y  z
			//  123 13640   -1  ->  13640 123 1     two loops
			//  123 13640 2134  ->  2134 13640 123   three loops
			//  123   -1    -1  ->  123 1 1        one loop
			Size3 launchParameters = convertLoopSizesToLaunchParameters(sizeOfLoops);
			
			int numberOfLoops = sizeOfLoops.getNumOfDimensions();
			if (launchParameters.x * launchParameters.y * launchParameters.z > BLOCK_SIZE)
			switch(numberOfLoops)
			{
				    case 0: blockDim = dim3(1,1,1);
					        assert(false);
						break;
				    case 1: blockDim = dim3(BLOCK_SIZE,1,1);
				        break;
				    case 2: blockDim = dim3(BLOCK_DIM_X,BLOCK_SIZE / BLOCK_DIM_X, 1);
				        break;
				    case 3: 
				    {
					    double remainder = BLOCK_SIZE / BLOCK_DIM_X;
					    double coefficient = (double)launchParameters.z / (double)launchParameters.y;
					    double yPow2 = remainder / coefficient;
					    double y = sqrt(yPow2); 	
					    blockDim =  dim3(BLOCK_DIM_X, y , coefficient*y);
					    assert(blockDim.x * blockDim.y * blockDim.z <= BLOCK_SIZE);
				    }
				    break;
			}
			else
			{ 
				//?????????????
				// Number of all iterations lower that number of threads in block
				blockDim = dim3(launchParameters.x,launchParameters.y,launchParameters.z);
			}
			
			dim3 iterationsPerThread(1,1,1);
			// Compute grid parameters from specified blockDim and desired
			//iterationsPerThread.
			gridDim.x = ((unsigned int)launchParameters.x - 1) / (blockDim.x * iterationsPerThread.x) + 1;
			gridDim.y = ((unsigned int)launchParameters.y - 1) / (blockDim.y * iterationsPerThread.y) + 1;
			gridDim.z = ((unsigned int)launchParameters.z - 1) / (blockDim.z * iterationsPerThread.z) + 1;
            
			// Substitute grid parameters to reduce amount of instructions
			// and used registers.
			substituteGridParams(kernel, gridDim, blockDim);
		}

		kernel->target[KERNELGEN_RUNMODE_CUDA].gridDim = gridDim;
		kernel->target[KERNELGEN_RUNMODE_CUDA].blockDim = blockDim;

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
					if ((name == "malloc") || (name == "posix_memalign") || (name == "free")) {
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

			// Do not compile the loop kernel if no grid detected.
			// Important to place this condition *after* hostcalls check
			// above to generate full host kernel in case hostcalls are around
			// (.supported flag), rather than running kernel on device and invoke
			// hostcalls on each individual iteration, which will be terribly slow.
			if (sizeOfLoops.x == -1) return NULL;
		}

		// Optimize module.
		{
			PassManager manager;
			PassManagerBuilder builder;
			builder.Inliner = createFunctionInliningPass();
			builder.OptLevel = 3;
			builder.DisableSimplifyLibCalls = true;
			builder.Vectorize = false;
			if (sizeOfLoops.x == -1) {
				// Single-threaded kernels performance could be
				// significantly improved by unrolling.
				manager.add(createLoopUnrollPass(512, -1, 1));
			}
			

			/*if (sizeOfLoops.x == -1)
			{
				// Use runtime unrolling (disabpled by default).
				builder.DisableUnrollLoops = true;
				builder.addExtension(PassManagerBuilder::EP_LoopOptimizerEnd,
					addRuntimeLoopUnrollPass);
			}*/
			builder.populateModulePassManager(manager);
			/*if (sizeOfLoops.x == -1)
			{
				manager.add(createLoopUnrollPass(2048, -1, 1));
			}*/
			manager.run(*m);
		}

		if (verbose & KERNELGEN_VERBOSE_SOURCES) m->dump();

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
			        triple.getTriple(), "", "", TargetOptions(), Reloc::PIC_, CodeModel::Default));
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

		if (verbose & KERNELGEN_VERBOSE_SOURCES) cout << bin_string;

		return nvopencc(bin_string, kernel->name,
			kernel->target[runmode].monitor_kernel_stream);

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
