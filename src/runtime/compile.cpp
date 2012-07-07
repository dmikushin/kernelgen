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

#include "polly/LinkAllPasses.h" // must include before "llvm/Transforms/Scalar.h"
#include "polly/RegisterPasses.h"

#include "llvm/Analysis/Verifier.h"
#include "llvm/Constants.h"
#include "llvm/LLVMContext.h"
#include "llvm/Instructions.h"
#include "llvm/Linker.h"
#include "llvm/Module.h"
#include "llvm/PassManager.h"
#include "llvm/Analysis/Passes.h"
#include "llvm/Support/FormattedStream.h"
#include "llvm/Support/IRReader.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Target/TargetData.h"
#include "llvm/Transforms/IPO/PassManagerBuilder.h"
#include "llvm/Support/raw_os_ostream.h"
#include "llvm/Support/ToolOutputFile.h"
#include "llvm/Support/TypeBuilder.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Transforms/Scalar.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include "llvm/Transforms/IPO.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/FormattedStream.h"

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
	IgnoreAliasing.setValue(true);
	polly::CUDA.setValue(mode);

	if (verbose & KERNELGEN_VERBOSE_POLLYGEN)
		llvm::EnableStatistics();
	//bool debug = ::llvm::DebugFlag;
	//if (verbose)
	//	::llvm::DebugFlag = true;
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
			if (verbose & KERNELGEN_VERBOSE_POLLYGEN)
			{
				outs().changeColor(raw_ostream::RED);
				outs() << "\n    FAIL: No Valid Scops detected in kernel!!!\n\n";
				outs().resetColor();
			}
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
	if (verbose & KERNELGEN_VERBOSE_POLLYGEN)
	{
		vector<string> statisticsNames;
		statisticsNames.push_back("polly-detect");
		statisticsNames.push_back("runtime-AA");
		printSpecifiedStatistics(statisticsNames);
		llvm::RemoveStatistics();
	}
	//::llvm::DebugFlag = debug;
	
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

        if (verbose)
	{
		outs().changeColor(raw_ostream::BLUE);
		outs() << "\n<------------------ "<< kernel->name << ": compile started --------------------->\n";
		outs().resetColor();
		
		outs() << kernel->source;
		outs() << "\n";
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

		return codegen(runmode, m, kernel->name, 0);
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

		if (verbose & KERNELGEN_VERBOSE_SOURCES) m->dump();

		// Change the target triple for entire module to be the same
		// as for KernelGen device runtime module.
		m->setTargetTriple(runtime_module->getTargetTriple());
		m->setDataLayout(runtime_module->getDataLayout());

		// Mark all module functions as device functions.
		for (Module::iterator F = m->begin(), FE = m->end(); F != FE; F++)
			F->setCallingConv(CallingConv::PTX_Device);

		// Mark kernel as GPU global function.
		f->setCallingConv(CallingConv::PTX_Kernel);

		// Mark all calls as calls to device functions.
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

				call->setCallingConv(CallingConv::PTX_Device);
		}

		// Link entire module with the KernelGen device runtime module.
		string err;
		if (Linker::LinkModules(m, runtime_module, Linker::DestroySource, &err))
		{
			THROW("Error linking runtime with kernel " << kernel->name << " : " << err);
		}

		if (verbose & KERNELGEN_VERBOSE_SOURCES) m->dump();

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

		/*// Replace alloca-s with global variables.
		vector<AllocaInst*> erase_allocas;
		for (Function::iterator bb = f->begin(); bb != f->end(); bb++)
			for (BasicBlock::iterator ii = bb->begin(), ie = bb->end(); ii != ie; ii++) {
				// Check if instruction in focus is an alloca.
				AllocaInst* alloca = dyn_cast<AllocaInst>(cast<Value>(ii));
				if (!alloca) continue;
				
				// Replace alloca with global variable.
				GlobalVariable* GV = new GlobalVariable(
					*m, alloca->getAllocatedType(), false, GlobalValue::PrivateLinkage,
					Constant::getNullValue(alloca->getAllocatedType()), "");
				GV->setAlignment(alloca->getAlignment());
				alloca->replaceAllUsesWith(GV);
				erase_allocas.push_back(alloca);
			}
		for (vector<AllocaInst*>::iterator i = erase_allocas.begin(),
			ie = erase_allocas.end(); i != ie; i++)
			(*i)->eraseFromParent();*/

		// Mark all pointer types referring to the GPU global address space.
		// Note this is an illegal hack, because it changes *types*, that are
		// unique instanses in LLVM. We used just for simplicity, to avoid
		// creating new types and instructions replacement.
		// For LLVM state consistency, after codegen the modified types
		// are reset back.
		/*for (LLVMContext::PointerTypesMap::iterator i = context.beginPointerTypes(),
			ie = context.endPointerTypes(); i != ie; i++)
		{
			std::pair<Type*, PointerType*>& type = *i;
			type.second->setAddressSpace(1);
		}*/
		
		/*// Change llvm.memset.p0i8.i64 into llvm.memset.p1i8.i64.
		Function* llvm_memset = m->getFunction("llvm.memset.p0i8.i64");
		llvm_memset->setName("llvm.memset.p1i8.i64");
		for (Function::iterator bb = f->begin(); bb != f->end(); bb++)
			for (BasicBlock::iterator ii = bb->begin(), ie = bb->end(); ii != ie; ii++) {	
				// Check if instruction in focus is a call.
				CallInst* call = dyn_cast<CallInst>(cast<Value>(ii));
				if (!call) continue;

				// Check if function is called (needs -instcombine pass).
				Function* callee = call->getCalledFunction();
				if (!callee) continue;
				if (!callee->isDeclaration()) continue;
				if (callee->isIntrinsic())
				{
					if (callee->getName() == "llvm.memset.p0i8.i64")
						callee->setName("llvm.memset.p1i8.i64");
				}
			}*/
		
		if (verbose & KERNELGEN_VERBOSE_SOURCES) m->dump();

		verifyModule(*m);

		kernel_func_t result = codegen(
			runmode, m, kernel->name, kernel->target[runmode].monitor_kernel_stream);

		/*// Reset to default address space.
		for (LLVMContext::PointerTypesMap::iterator i = context.beginPointerTypes(),
			ie = context.endPointerTypes(); i != ie; i++)
		{
			std::pair<Type*, PointerType*>& type = *i;
			type.second->setAddressSpace(0);
		}*/


		return result;

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
