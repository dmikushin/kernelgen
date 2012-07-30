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
#include "llvm/Target/TargetOptions.h"
#include "llvm/Transforms/IPO/PassManagerBuilder.h"
#include "llvm/Support/raw_os_ostream.h"
#include "llvm/Support/ToolOutputFile.h"
#include "llvm/Support/TypeBuilder.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Transforms/Scalar.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include "llvm/Transforms/Utils/Cloning.h"
#include "llvm/Transforms/IPO.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/FormattedStream.h"
#include "llvm/Support/Host.h"
#include "llvm/Support/TargetRegistry.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/IRBuilder.h"
#include "polly/ScopInfo.h"

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

#include "LinkFunctionBody.h"

using namespace kernelgen;
using namespace kernelgen::bind::cuda;
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
extern cl::opt<bool> IgnoreAliasing;

void ConstantSubstitution( Function * func, void * args);
Pass* createSizeOfLoopsPass(vector<Size3> *memForSize3 = NULL);
Pass* createRuntimeAliasAnalysisPass();

Pass* createTransformAccessesPass();
Pass* createInspectDependencesPass();
Pass* createScopDescriptionPass();
Pass* createSetRelationTypePass(MemoryAccess::RelationType relationType = MemoryAccess::RelationType_polly);

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

static void registerPollyPreoptPasses(llvm::PassManagerBase &PM) {
  // A standard set of optimization passes partially taken/copied from the
  // set of default optimization passes. It is used to bring the code into
  // a canonical form that can than be analyzed by Polly. This set of passes is
  // most probably not yet optimal. TODO: Investigate optimal set of passes.
  PM.add(llvm::createPromoteMemoryToRegisterPass());
  PM.add(llvm::createInstructionCombiningPass());  // Clean up after IPCP & DAE
  PM.add(llvm::createCFGSimplificationPass());     // Clean up after IPCP & DAE
  PM.add(llvm::createTailCallEliminationPass());   // Eliminate tail calls
  PM.add(llvm::createCFGSimplificationPass());     // Merge & remove BBs
  PM.add(llvm::createReassociatePass());           // Reassociate expressions
  PM.add(llvm::createLoopRotatePass());            // Rotate Loop
  PM.add(llvm::createInstructionCombiningPass());
  PM.add(polly::createIndVarSimplifyPass());        // Canonicalize indvars

  PM.add(polly::createCodePreparationPass());
  PM.add(polly::createRegionSimplifyPass());
  // FIXME: The next two passes should not be necessary here. They are currently
  //        because of two problems:
  //
  //        1. The RegionSimplifyPass destroys the canonical form of induction
  //           variables,as it produces PHI nodes with incorrectly ordered
  //           operands. To fix this we run IndVarSimplify.
  //
  //        2. IndVarSimplify does not preserve the region information and
  //           the regioninfo pass does currently not recover simple regions.
  //           As a result we need to run the RegionSimplify pass again to
  //           recover them
  PM.add(polly::createIndVarSimplifyPass());
  PM.add(polly::createRegionSimplifyPass());
}

void getAllocasAndMaximumSize(Function *f,set<AllocaInst *> *allocasForArgs, unsigned long long * maximumSizeOfData )
{
	Value *tmpArg=NULL;
	if (f && allocasForArgs && maximumSizeOfData) {
		for (Value::use_iterator UI = f->use_begin(), UE = f->use_end(); UI != UE; UI++) {
			CallInst* call = dyn_cast<CallInst>(*UI);
			if(!call) continue;

			//retrive size of data
			tmpArg = call -> getArgOperand(1);
			assert( isa<ConstantInt>(*tmpArg) && "by this time, after targetData,"
			        "second parameter of kernelgen functions "
			        "must be ConstantInt");

			//get maximum size of data
			uint64_t sizeOfData = ((ConstantInt*)tmpArg)->getZExtValue();
			if(*maximumSizeOfData < sizeOfData)
				*maximumSizeOfData=sizeOfData;

			tmpArg = call -> getArgOperand(3);
			while(!isa<AllocaInst>(*tmpArg)) {
				assert(isa<BitCastInst>(*tmpArg));
				tmpArg=cast<BitCastInst>(tmpArg)->getOperand(0);
			}
			allocasForArgs->insert(cast<AllocaInst>(tmpArg));
		}
	}
}

static void runPolly(kernel_t *kernel, Size3 *sizeOfLoops,bool mode)
{
	{
		PassManager polly;
		polly.add(new TargetData(kernel->module));
		registerPollyPreoptPasses(polly);
		//polly.add(polly::createIslScheduleOptimizerPass());
		polly.run(*kernel->module);
	}

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
		//polly.add(polly::createIslScheduleOptimizerPass());
		if (kernel->name != "__kernelgen_main") {
			//polly.add(createRuntimeAliasAnalysisPass());
		    polly.add(createScopDescriptionPass());   // print scop description
		    polly.add(createTransformAccessesPass()); // create General Form for each scop's memory Access
			                                          // set their current relation types to RelationType_general
		    //polly.add(createScopDescriptionPass());
		    polly.add(createInspectDependencesPass()); // Dependences run and compute dependences 
			                                           // before InspectDependences, but after TransformAccesses
                                                       // and use general form of memory accesses
			polly.add(createSizeOfLoopsPass(&sizes));  // compute size of loops
		    polly.add(createSetRelationTypePass()); // set current relation types in scop's memory Accesses back to 
			                                        // RelationType_polly
		    //polly.add(createScopDescriptionPass());
		}
		polly.add(polly::createCodeGenerationPass()); // -polly-codegenn
													  // is use polly's representation of Memory Accesses
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
void substituteGridParams(kernel_t* kernel,dim3 & gridDim, dim3 & blockDim)
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

kernel_func_t kernelgen::runtime::compile(
    int runmode, kernel_t* kernel, Module* module, void * data, int szdata, int szdatai)
{
	// Do not compile, if no source.
	if (kernel->source == "")
		return kernel->target[runmode].binary;

        if (verbose)
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
	}

	// Add signature record.
	Constant* CSig = ConstantDataArray::getString(context, "0.2/" KERNELGEN_VERSION, true);
	GlobalVariable* GVSig = new GlobalVariable(*m, CSig->getType(),
		true, GlobalValue::ExternalLinkage, CSig, "__kernelgen_version", 0, false);

	// Emit target assembly and binary image, depending
	// on runmode.
	Function* f = m->getFunction(kernel->name);
	switch (runmode) {
	case KERNELGEN_RUNMODE_NATIVE : {
		if (kernel->name != "__kernelgen_main") {
			// Apply the Polly codegen for native target.
			// To perform this analysis, require the native runmode to be
			// used globally for entire app, i.e. not a fallback from
			// non-portable GPU kernel or hostcall.
			if (runmode == kernelgen::runmode)
			{
				// Substitute integer and pointer arguments.
				if (szdatai != 0) ConstantSubstitution(f, data);

				Size3 sizeOfLoops;
				runPollyNATIVE(kernel, &sizeOfLoops);

				// Do not compile the loop kernel if no grid detected.
				if (sizeOfLoops.x == -1)
				{
					// XXX Turn off future kernel analysis, if it has been detected as
					// non-parallel at least once. This behavior is subject for change in future.
					kernel->target[runmode].supported = false;
					return NULL;
				}
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

		verifyModule(*m);
		if (verbose & KERNELGEN_VERBOSE_SOURCES) m->dump();

		return codegen(runmode, kernel, m);
	}
	case KERNELGEN_RUNMODE_CUDA : {

		// Change the target triple for entire module to be the same
		// as for KernelGen device runtime module.
		m->setTargetTriple(runtime_module->getTargetTriple());
		m->setDataLayout(runtime_module->getDataLayout());

		// Mark all module functions as device functions.
		for (Module::iterator F = m->begin(), FE = m->end(); F != FE; F++)
			F->setCallingConv(CallingConv::PTX_Device);

		// Mark kernel as GPU global function.
		f->setCallingConv(CallingConv::PTX_Kernel);

		dim3 blockDim(1, 1, 1);
		dim3 gridDim(1, 1, 1);
		if (kernel->name != "__kernelgen_main")  {	
			// If the target kernel is loop, do not allow host calls in it.
			// Also do not allow malloc/free, probably support them later.
			// TODO: kernel *may* have kernelgen_launch called, but it must
			// always evaluate to -1.
			for (Function::iterator bb = f->begin(); bb != f->end(); bb++)
				for (BasicBlock::iterator ii = bb->begin(), ie = bb->end(); ii != ie; ii++) {
					
										//Check if instruction in focus is a AllocaInst
					AllocaInst *alloca = dyn_cast<AllocaInst>(ii);
					if(alloca)
						if(alloca->isArrayAllocation())
							if(!isa<Constant>(*alloca->getArraySize())) {
								if (verbose) {
									outs().changeColor(raw_ostream::RED);
									outs() << "Not allowed dynamic alloca: " << *alloca << "\n";
									outs().resetColor();
								}
								kernel->target[runmode].supported = false;
								return NULL;
							}
						
					// Check if instruction in focus is a call.
					CallInst* call = dyn_cast<CallInst>(cast<Value>(ii));
					if (!call) continue;

					// Check if function is called (needs -instcombine pass).
					Function* callee = call->getCalledFunction();
					if (!callee) continue;
					if (!callee->isDeclaration()) continue;
				
					// If function is only declared, then try to find and
					// insert its implementation from the CUDA runtime module.
					Function* Dst = callee;
					Function* Src = cuda_module->getFunction(callee->getName());
					if (Src)
					{
						if (verbose)
							cout << "Device call: " << callee->getName().data()<< endl;
						call->setCallingConv(CallingConv::PTX_Device);					
						LinkFunctionBody(Dst, Src);
						continue;
					}

					if (callee->isIntrinsic()) continue;

					// Loop kernel contains non-native calls, and therefore
					// cannot be executed on GPU.
					if (verbose)
						cout << "Not allowed host call: " << callee->getName().data()<< endl;
					kernel->target[runmode].supported = false;
					return NULL;
				}

			// Substitute integer and pointer arguments.
			if (szdatai != 0) ConstantSubstitution(f, data);

			// Apply the Polly codegen for CUDA target.
			Size3 sizeOfLoops;
			runPollyCUDA(kernel, &sizeOfLoops);

			// Do not compile the loop kernel if no grid detected.
			// Important to place this condition *after* hostcalls check
			// above to generate full host kernel in case hostcalls are around
			// (.supported flag), rather than running kernel on device and invoke
			// hostcalls on each individual iteration, which will be terribly slow.
			if (sizeOfLoops.x == -1)
			{
				// Dump the LLVM IR Polly has failed to create
				// gird in for futher analysis with kernelgen-polly.
				/*if (verbose & KERNELGEN_VERBOSE_POLLYGEN)
				{
					// Put the resulting module into LLVM output file
					// as object binary. Method: create another module
					// with a global variable incorporating the contents
					// of entire module and emit it for X86_64 target.
					string ir_string;
					raw_string_ostream ir(ir_string);
					ir << *m;
					Module obj_m("module" + kernel->name, context);
					Constant* C1 = ConstantDataArray::getString(context, ir_string, true);
					GlobalVariable* GV1 = new GlobalVariable(obj_m, C1->getType(),
						true, GlobalValue::LinkerPrivateLinkage, C1,
						kernel->name, 0, false);
					SmallVector<uint8_t, 16> adata((char*)data, (char*)data + szdata);
					Constant* C2 = ConstantDataArray::get(context, adata);
					GlobalVariable* GV2 = new GlobalVariable(obj_m, C2->getType(),
						true, GlobalValue::LinkerPrivateLinkage, C2,
						"args" + kernel->name, 0, false);
					APInt aszdata(8 * sizeof(int), szdata);
					Constant* C3 = Constant::getIntegerValue(Type::getInt32Ty(context), aszdata);
					GlobalVariable* GV3 = new GlobalVariable(obj_m, C3->getType(),
						true, GlobalValue::LinkerPrivateLinkage, C3,
						"szargs" + kernel->name, 0, false);
					APInt aszdatai(8 * sizeof(int), szdatai);
					Constant* C4 = Constant::getIntegerValue(Type::getInt32Ty(context), aszdatai);
					GlobalVariable* GV4 = new GlobalVariable(obj_m, C4->getType(),
						true, GlobalValue::LinkerPrivateLinkage, C4,
						"szargsi" + kernel->name, 0, false);
					
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

					// Setup output stream.
					string bin_string;
					raw_string_ostream bin_stream(bin_string);
					formatted_raw_ostream bin_raw_stream(bin_stream);

					// Ask the target to add backend passes as necessary.
					PassManager manager;
					const TargetData* tdata =
						targets[KERNELGEN_RUNMODE_NATIVE].get()->getTargetData();
					manager.add(new TargetData(*tdata));
					if (targets[KERNELGEN_RUNMODE_NATIVE].get()->addPassesToEmitFile(manager, bin_raw_stream,
						TargetMachine::CGFT_ObjectFile, CodeGenOpt::Aggressive))
						THROW("Target does not support generation of this file type");
					manager.run(obj_m);

					// Flush the resulting object binary to the underlying string.
					bin_raw_stream.flush();

					// Dump the generated kernel object to file.
					fstream stream;
					string filename = kernel->name + ".kernelgen.o";
					stream.open(filename.c_str(),
						fstream::binary | fstream::out | fstream::trunc);
					stream << bin_string;
					stream.close();
				}*/
			
				// XXX Turn off future kernel analysis, if it has been detected as
				// non-parallel at least once. This behavior is subject for change in future.
				kernel->target[runmode].supported = false;
				return NULL;
			}

			int device;
			CUresult err = cuDeviceGet(&device, 0);
			if (err)
				THROW("Error in cuDeviceGet " << err);

			typedef struct
			{
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
			err = cuDeviceGetProperties((void*)&props, device);
			if (err)
				THROW("Error in cuDeviceGetProperties " << err);

			// x   y     z         x     y     z
			// 123 13640   -1  ->  13640 123   1     two loops
			// 123 13640 2134  ->  2134  13640 123   three loops
			// 123   -1    -1  ->  123   1     1     one loop
			Size3 launchParameters = convertLoopSizesToLaunchParameters(sizeOfLoops);
#define BLOCK_DIM_X 32
			int numberOfLoops = sizeOfLoops.getNumOfDimensions();
			if (launchParameters.x * launchParameters.y * launchParameters.z > props.maxThreadsPerBlock)
			switch (numberOfLoops)
			{
			case 0:	blockDim = dim3(1, 1, 1);
				assert(false);
				break;
			case 1: blockDim = dim3(props.maxThreadsPerBlock, 1, 1);
				break;
			case 2: blockDim = dim3(BLOCK_DIM_X, props.maxThreadsPerBlock / BLOCK_DIM_X, 1);
				break;
			case 3:
				{
					double remainder = props.maxThreadsPerBlock / BLOCK_DIM_X;
					double coefficient = (double)launchParameters.z / (double)launchParameters.y;
					double yPow2 = remainder / coefficient;
					double y = sqrt(yPow2);
					blockDim = dim3(BLOCK_DIM_X, y , coefficient * y);
					assert(blockDim.x * blockDim.y * blockDim.z <= props.maxThreadsPerBlock);
				}
				break;
			}
			else
			{ 
				// Number of all iterations lower that number of threads in block
				blockDim = dim3(launchParameters.x, launchParameters.y, launchParameters.z);
			}
			
			// Compute grid parameters from specified blockDim and desired iterationsPerThread.
			dim3 iterationsPerThread(1,1,1);
			gridDim.x = ((unsigned int)launchParameters.x - 1) / (blockDim.x * iterationsPerThread.x) + 1;
			gridDim.y = ((unsigned int)launchParameters.y - 1) / (blockDim.y * iterationsPerThread.y) + 1;
			gridDim.z = ((unsigned int)launchParameters.z - 1) / (blockDim.z * iterationsPerThread.z) + 1;
            
			// Substitute grid parameters to reduce amount of instructions
			// and used registers.
			substituteGridParams(kernel, gridDim, blockDim);
		}

		kernel->target[KERNELGEN_RUNMODE_CUDA].gridDim = gridDim;
		kernel->target[KERNELGEN_RUNMODE_CUDA].blockDim = blockDim;

		// Convert external functions CallInst-s into
		// host callback form. Do not convert CallInst-s
		// to intrinsics and calls linked from the CUDA device runtime module.
		if (kernel->name == "__kernelgen_main") {
			// Mark all calls as calls to device functions.
			for (Function::iterator bb = f->begin(); bb != f->end(); bb++)
				for (BasicBlock::iterator ii = bb->begin(), ie = bb->end(); ii != ie; ii++) {
					// Check if instruction in focus is a call.
					CallInst* call = dyn_cast<CallInst>(cast<Value>(ii));
					if (!call) continue;

					// Check if function is called (needs -instcombine pass).
					Function* callee = call->getCalledFunction();
					if (!callee) continue;

					call->setCallingConv(CallingConv::PTX_Device);
				}

			// Link entire module with the KernelGen device runtime module.
			string linker_err;
			if (Linker::LinkModules(m, runtime_module, Linker::PreserveSource, &linker_err))
			{
				THROW("Error linking runtime with kernel " << kernel->name << " : " << linker_err);
			}

			vector<CallInst*> erase_calls;
			for (Function::iterator bb = f->begin(); bb != f->end(); bb++)
				for (BasicBlock::iterator ii = bb->begin(), ie = bb->end(); ii != ie; ii++) {
					// Check if instruction in focus is a call.
					CallInst* call = dyn_cast<CallInst>(cast<Value>(ii));
					if (!call) continue;

					// Check if function is called (needs -instcombine pass).
					// Could be also a function called inside a bitcast.
					// So try to extract function from the underlying constant expression.
					// Required to workaround GCC/DragonEGG issue:
					// http://lists.cs.uiuc.edu/pipermail/llvmdev/2012-July/051786.html
					Function* callee = dyn_cast<Function>(
						call->getCalledValue()->stripPointerCasts());
					if (!callee) continue;
					if (!callee->isDeclaration()) continue;

					string name = callee->getName();

					// If function is only declared, then try to find and
					// insert its implementation from the CUDA runtime module.
					Function* Dst = callee;
					Function* Src = cuda_module->getFunction(callee->getName());
					if (Src)
					{
						if (verbose)
							cout << "Device call: " << callee->getName().data()<< endl;
						call->setCallingConv(CallingConv::PTX_Device);					
						LinkFunctionBody(Dst, Src);
						continue;
					}

					if (callee->isIntrinsic()) continue;

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

			// Evaluate ConstantExpr::SizeOf to integer number ConstantInt
			PassManager manager;
			manager.add(new TargetData(m));
			manager.add(createInstructionCombiningPass());
			manager.run(*m);
			
			{
				Function* kernelgenFunction = NULL;
				Value * tmpArg = NULL;
				AllocaInst* oldCollectiveAlloca = NULL;
				unsigned long long maximumSizeOfData=0;
				
				//set of allocas we want to collect together
				set<AllocaInst *> allocasForArgs;
				allocasForArgs.clear();

				// collect alloca-s for kernelgen_launch-s
				kernelgenFunction = m->getFunction("kernelgen_launch");
				getAllocasAndMaximumSize(kernelgenFunction, &allocasForArgs, &maximumSizeOfData);
				
				// there must be one collective alloca for all kernelgen_launch-s, it was made in link-step
				// if it is
				assert(allocasForArgs.size() <= 1);
				if(allocasForArgs.size() == 1) {
					oldCollectiveAlloca=*allocasForArgs.begin();
					assert((oldCollectiveAlloca->getAllocatedType()->isStructTy() || oldCollectiveAlloca->getAllocatedType()->isArrayTy())
					       && "must be allocation of array or struct");
					oldCollectiveAlloca->setName("oldCollectiveAllocaForArgs");
				}

				// Collect allocas for kernelgen_hostcall-s
				kernelgenFunction = m->getFunction("kernelgen_hostcall");
				getAllocasAndMaximumSize(kernelgenFunction, &allocasForArgs, &maximumSizeOfData);

				// Replace all allocas by one collective alloca
				// allocate array [i8 x maximumSizeOfData]
				AllocaInst *collectiveAlloca = new AllocaInst(ArrayType::get(Type::getInt8Ty(m->getContext()),maximumSizeOfData),
				        "collectiveAllocaForArgs",
				        f->begin()->begin());

				for(set<AllocaInst *>::iterator iter=allocasForArgs.begin(), iter_end=allocasForArgs.end();
				    iter!=iter_end; iter++ ) {
					AllocaInst * allocaInst=*iter;

					//get type of old alloca
					Type * structPtrType = allocaInst -> getType();

					//create bit cast of created alloca for specified type
					BitCastInst * bitcast=new BitCastInst(collectiveAlloca,structPtrType,"ptrToArgsStructure");
					//insert after old alloca
					bitcast->insertAfter(allocaInst);
					//replace uses of old alloca with created bit cast
					allocaInst -> replaceAllUsesWith(bitcast);
					//erase old alloca from parent basic block
					allocaInst -> eraseFromParent();
				}
			}

		// Replace static alloca-s with global variables.
		// Replace dynamic alloca-s with kernelgen_malloc
		vector<AllocaInst*> allocas;
		for (Function::iterator bb = f->begin(); bb != f->end(); bb++)
			for (BasicBlock::iterator ii = bb->begin(), ie = bb->end(); ii != ie; ii++) {
				// Check if instruction in focus is an alloca.
				AllocaInst* alloca = dyn_cast<AllocaInst>(cast<Value>(ii));
				if (!alloca) continue;
				else allocas.push_back(alloca);
			}
		for (vector<AllocaInst*>::iterator i = allocas.begin(),
		ie = allocas.end(); i != ie; i++) {
			AllocaInst *alloca = *i;

			if(!alloca->isArrayAllocation()) {
				//allocation of single element
				// Replace "alloca type" with "@a=type".
				GlobalVariable* GV = new GlobalVariable(
				    *m, alloca->getAllocatedType(), false, GlobalValue::PrivateLinkage,
				    Constant::getNullValue(alloca->getAllocatedType()), "replacementOfAlloca");
				GV->setAlignment(alloca->getAlignment());
				alloca->replaceAllUsesWith(GV);
				if (verbose) {
					outs().changeColor(raw_ostream::RED);
					outs() << "Replace \"" << *alloca << "\" with \n  " << *GV;
					outs().resetColor();
				}
			} else //allocation of array of elements
				if(isa<ConstantInt>(*alloca->getArraySize())) {
					// simple case: array size is the ConstantInt and we can retrive actual size as uint64_t
					// Replace "alloca type, 10" with "@a=[type x 10]; bitcast [type x 10]* @a to type*"
					uint64_t numElements = cast<ConstantInt>(alloca->getArraySize())->getZExtValue();
					ArrayType *arrayType = ArrayType::get(alloca->getAllocatedType(),numElements);
					GlobalVariable* GV = new GlobalVariable(
					    *m, arrayType, false, GlobalValue::PrivateLinkage,
					    Constant::getNullValue(arrayType), "memoryForAlloca");
					GV->setAlignment(alloca->getAlignment());
					BitCastInst * bitcast=new BitCastInst(GV,alloca->getType(),"replacementOfAlloca",alloca);
					alloca->replaceAllUsesWith(bitcast);

					if (verbose) {
						outs().changeColor(raw_ostream::GREEN);
						outs() << "Replace \"" << *alloca << "\" with \n  " << *GV << *bitcast << "\n";
						outs().resetColor();
					}
				} else {
					// more complex case: array size is a common value and unknown at compile-time
					// Replace "alloca type, count" with "%0=call i8* kernelgen_malloc(sizeof(type)*count)
					//%1=bitcast i8* %0 to type*"

					IRBuilder<> Builder(alloca);
					Constant *sizeOfElement = ConstantExpr::getSizeOf(alloca->getAllocatedType());
					Value *sizeOfAllocation = Builder.CreateMul(alloca->getArraySize(),sizeOfElement,"allocationSize");
					Function *kernelgenMalloc =  m->getFunction("kernelgen_malloc");
					assert(kernelgenMalloc && sizeOfAllocation->getType()->isIntegerTy());
					CallInst *callOfMalloc = (CallInst *)Builder.CreateCall(kernelgenMalloc,sizeOfAllocation,"memoryForAlloca");
					callOfMalloc->setCallingConv(llvm::CallingConv::PTX_Device);


					if(callOfMalloc->getType() == alloca->getType()) {
						callOfMalloc->setName((string)"replacementOfAlloca");
						alloca->replaceAllUsesWith(callOfMalloc);
						if (verbose) {
							outs().changeColor(raw_ostream::GREEN);
							outs() << "Replace \"" << *alloca << "\" with \n" << *callOfMalloc << "\n";
							outs().resetColor();
						}
					} else {
						Value * bitcast = Builder.CreateBitCast(callOfMalloc, alloca->getType(),"replacementOfAlloca");
						alloca->replaceAllUsesWith(bitcast);
						if (verbose) {
							outs().changeColor(raw_ostream::GREEN);
							outs() << "Replace \"" << *alloca << "\" with \n" << *callOfMalloc << "\n" << *bitcast << "\"\n";
							outs().resetColor();
						}
					}

				}

			alloca->eraseFromParent();
		}
		
			// Align all globals to 4096.
			for (Module::global_iterator GV = m->global_begin(), GVE = m->global_end(); GV != GVE; GV++)
				GV->setAlignment(4096);
		}

		// Optimize only loop kernels.
		if(strcmp(kernel->name.c_str(), "__kernelgen_main"))
		{
			PassManager manager;
			PassManagerBuilder builder;
			builder.Inliner = createFunctionInliningPass();
			
			builder.OptLevel = 3;
			builder.DisableSimplifyLibCalls = true;
			builder.DisableUnrollLoops = true;
			builder.Vectorize = false;
			builder.populateModulePassManager(manager);
	
			manager.run(*m);
		}
		else
        {
	        PassManager MPM;
            MPM.add(createGlobalOptimizerPass());     // Optimize out global vars
            MPM.add(createFunctionInliningPass());
            MPM.add(createStripDeadPrototypesPass()); // Get rid of dead prototypes
            MPM.run(*m);
        }
		
			// Erase all defined functions, except the kernel.
		vector<Function*> erase_funcs;
		for (Module::iterator F = m->begin(), FE = m->end(); F != FE; F++)
			if (!F->isDeclaration() && (F->getName() != kernel->name))
				erase_funcs.push_back(F);
		for (vector<Function*>::iterator i = erase_funcs.begin(),
                        ie = erase_funcs.end(); i != ie; i++)
                        (*i)->eraseFromParent();

		verifyModule(*m);
		if (verbose & KERNELGEN_VERBOSE_SOURCES) m->dump();

		return codegen(runmode, kernel, m);
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
