//===- Compile.cpp - Kernels compiler API ---------------------------------===//
//
//     KernelGen -- A prototype of LLVM-based auto-parallelizing Fortran/C
//        compiler for NVIDIA GPUs, targeting numerical modeling code.
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements compiling of parallel loops on supported architectures.
//
//===----------------------------------------------------------------------===//

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

#include "Runtime.h"

#include <math.h>
#include <cstdlib>
#include <dlfcn.h>
#include <fstream>
#include <list>
#include <set>
#include <stdio.h>

#include "GlobalDependences.h"

using namespace kernelgen;
using namespace kernelgen::bind::cuda;
using namespace kernelgen::runtime;
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
Pass* createSizeOfLoopsPass(vector<Size3> *memForSize3 = NULL, bool * isThereAtLeastOneParallelLoop = NULL);
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

static void registerPollyPreoptPasses(llvm::PassManagerBase &PM)
{
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

void getAllocasAndMaximumSize(Function *f,list<Value *> *allocasForArgs, unsigned long long * maximumSizeOfData )
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

			tmpArg = call -> getArgOperand(3)->stripPointerCasts();
			/*while(!isa<AllocaInst>(*tmpArg)) {
				assert(isa<BitCastInst>(*tmpArg));
				tmpArg=cast<BitCastInst>(tmpArg)->getOperand(0);
			}*/
			assert(isa<AllocaInst>(*tmpArg));
			allocasForArgs->push_back(tmpArg);
		}
	}
}

static void runPolly(Kernel *kernel, Size3 *sizeOfLoops,bool mode, bool *isThereAtLeastOneParallelLoop)
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

	if (settings.getVerboseMode() & Verbose::Polly)
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
			polly.add(createSizeOfLoopsPass(&sizes, isThereAtLeastOneParallelLoop));  // compute size of loops
		    polly.add(createSetRelationTypePass()); // set current relation types in scop's memory Accesses back to 
			                                        // RelationType_polly
		    //polly.add(createScopDescriptionPass());
		}
		polly.add(polly::createCodeGenerationPass()); // -polly-codegenn
													  // use polly's representation of Memory Accesses
		polly.add(createCFGSimplificationPass());
		polly.run(*kernel->module);
	}
	if (kernel->name != "__kernelgen_main") {
		Size3 SizeOfLoops;
		if(sizes.size() == 0)
		{
			VERBOSE(Verbose::Polly << Verbose::Red <<
					"\n    FAIL: No Valid Scops detected in kernel!!!\n\n" <<
					Verbose::Default << Verbose::Reset);
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
			SizeOfLoops = sizes[0]; // 3-dimensional
			if (sizeOfLoops) *sizeOfLoops = SizeOfLoops;
		}

	}
	if (settings.getVerboseMode() & Verbose::Polly)
	{
		vector<string> statisticsNames;
		statisticsNames.push_back("polly-detect");
		statisticsNames.push_back("runtime-AA");
		printSpecifiedStatistics(statisticsNames);
		llvm::RemoveStatistics();
	}
	//::llvm::DebugFlag = debug;
	VERBOSE(Verbose::Blue <<
			"<------------------ "<< kernel->name << ": compile completed ------------------->\n\n" <<
			Verbose::Reset);
}
static void runPollyNATIVE(Kernel *kernel, Size3 *sizeOfLoops)
{
	return runPolly(kernel, sizeOfLoops, false, NULL);
}
static void runPollyCUDA(Kernel *kernel, Size3 *sizeOfLoops, bool *isThereAtLeastOneParallelLoop)
{
	return runPolly(kernel, sizeOfLoops, true, isThereAtLeastOneParallelLoop);
}
void substituteGridParams(Kernel* kernel,dim3 & gridDim, dim3 & blockDim)
{
	Module * m = kernel->module;
	vector<string> dimensions, parameters;

	dimensions.push_back("x");
	dimensions.push_back("y");
	dimensions.push_back("z");


	parameters.push_back("nctaid");
	parameters.push_back("ntid");

	string prefix1("llvm.nvvm.read.ptx.sreg.");
	string dot(".");
	

	unsigned int * gridParams[2];
	gridParams[0] = (unsigned int *)&gridDim;
	gridParams[1] = (unsigned int *)&blockDim;

	for(int parameter =0; parameter < parameters.size(); parameter++)
		for(int dimension=0; dimension < dimensions.size(); dimension++) {
			string functionName = prefix1 + parameters[parameter] + dot + dimensions[dimension];
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

static void substituteGlobalsByTheirAddresses(Module *m)
{
	list<GlobalVariable *> globalVariables;
	LLVMContext & context = m->getContext(); 
	for(Module::global_iterator iter=m->global_begin(), iter_end=m->global_end();
	     iter!=iter_end; iter++)
	{
	
		 GlobalVariable *globalVar = iter;
		 string globalName = globalVar->getName();
		 
		 if(!globalName.compare("__kernelgen_callback") || !globalName.compare("__kernelgen_memory"))
		 {
			 assert(globalVar->getNumUses() == 0);
			 globalVariables.push_back(globalVar);
			 continue;
		 }
		 std::map<llvm::StringRef, uint64_t>::const_iterator tuple;
		 tuple = orderOfGlobals.find(globalVar->getName());
		 assert(tuple != orderOfGlobals.end());
		
		 int index = tuple->second;
		 /* int offset = strlen("global.");
		 assert(!globalName.substr(0,offset).compare("global."));
		 int index = atoi(globalName.substr(offset, globalName.length() - offset).c_str());*/
		 
		 uint64_t address = AddressesOfGVars[index];
		 Constant *replacement = ConstantExpr::getIntToPtr(
				  ConstantInt::get(Type::getInt64Ty(context),address),
				  globalVar->getType());
				  
		globalVar->replaceAllUsesWith(replacement);
		globalVariables.push_back(globalVar);
	}
	for(list<GlobalVariable *>::iterator iter=globalVariables.begin(), iter_end = globalVariables.end();
	     iter!=iter_end; iter++)
			 (*iter)->eraseFromParent();
}

static void processFunctionFromMain(Kernel* kernel, Module* m, Function* f)
{
	list<AllocaInst*> allocas;
	list<CallInst*> erase_calls;
	for (Function::iterator bb = f->begin(); bb != f->end(); bb++)
		for (BasicBlock::iterator ii = bb->begin(), ie = bb->end(); ii != ie; ii++)
		{
			AllocaInst* alloca = dyn_cast<AllocaInst>(cast<Value>(ii));
			if(alloca) {
				// handle alloca later
				allocas.push_back(alloca);
				continue;
			}
					
			// Check if instruction in focus is a call.
			CallInst* call = dyn_cast<CallInst>(cast<Value>(ii));
			if (!call) continue;
			
			// Mark all calls as calls to device functions. 
			call->setCallingConv(CallingConv::PTX_Device);

			// Check if function is called (needs -instcombine pass).
			// Could be also a function called inside a bitcast.
			// So try to extract function from the underlying constant expression.
			// Required to workaround GCC/DragonEGG issue:
			// http://lists.cs.uiuc.edu/pipermail/llvmdev/2012-July/051786.html
			Function* callee = dyn_cast<Function>(
				call->getCalledValue()->stripPointerCasts());
			if (!callee) continue;

			call->setAttributes(callee->getAttributes());
			// If function is defined (has body),
			// it will be handled in another call of processCallTreeMain
			if (!callee->isDeclaration())
				continue;

			string name = callee->getName();

			// If function is only declared, then try to find and
			// insert its implementation from the CUDA runtime module.
			Function* Dst = callee;
			Function* Src = cuda_module->getFunction(callee->getName());
			if (Src && (Src != Dst))
				if (!Src->isDeclaration())
			{
				VERBOSE("Device call: " << callee->getName().data() << "\n");
				call->setCallingConv(CallingConv::PTX_Device);					
				linkFunctionWithAllDependendes(Src,Dst);
				Dst->setName(Dst->getName());
				
				Dst->setAttributes(Src->getAttributes());
				for (Value::use_iterator use_iter = Dst->use_begin(),
					use_iter_end = Dst->use_end(); use_iter != use_iter_end; use_iter++)
				{
					CallInst * call = cast<CallInst>(*use_iter);
					call->setAttributes(Dst->getAttributes());
				}
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
				VERBOSE("replacement: " << name << " -> " << rename << "\n");
				continue;
			}

			// Also record hostcall to the kernels map.
			Kernel* hostcall = kernels[name];
			if (!hostcall) {
				hostcall = new Kernel();
				hostcall->name = name;
				hostcall->source = "";

				// No targets supported, except NATIVE.
				for (int i = 0; i < KERNELGEN_RUNMODE_COUNT; i++) {
					hostcall->target[i].supported = false;
					hostcall->target[i].binary = NULL;
				}
				hostcall->target[KERNELGEN_RUNMODE_NATIVE].supported = true;

				hostcall->target[RUNMODE].MonitorStream =
				    kernel->target[RUNMODE].MonitorStream;
				hostcall->module = kernel->module;

				kernels[name] = hostcall;
			}

			// Replace entire call with hostcall and set old
			// call for erasing.
			WrapCallIntoHostcall(call, hostcall);
			erase_calls.push_back(call);
		}
		
		for (list<CallInst*>::iterator i = erase_calls.begin(),
			ie = erase_calls.end(); i != ie; i++)
			   (*i)->eraseFromParent();

		// Replace static alloca-s with global variables.
		// Replace dynamic alloca-s with kernelgen_malloc.
		Type* i1Ty = Type::getInt1Ty(m->getContext());
		for (list<AllocaInst*>::iterator i = allocas.begin(),
			ie = allocas.end(); i != ie; i++) {
			AllocaInst *alloca = *i;
			Type* Ty = alloca->getAllocatedType();

			// Use of i1 is not supported by NVPTX.
			assert(Ty != i1Ty);

			if(!alloca->isArrayAllocation()) {
				// Allocation of single element:
				// Replace "alloca type" with "@a = type".
				GlobalVariable* GV = new GlobalVariable(
					*m, Ty, false, GlobalValue::PrivateLinkage,
					Constant::getNullValue(Ty), "replacementOfAlloca");
				GV->setAlignment(alloca->getAlignment());
				alloca->replaceAllUsesWith(GV);
				VERBOSE(Verbose::Alloca << Verbose::Red <<
						"Replace \"" << *alloca << "\" with \n  " << *GV <<
						Verbose::Reset << Verbose::Default);
			} 
			else if(isa<ConstantInt>(*alloca->getArraySize())) {
				// Allocation of array of elements:
				// simple case: array size is the ConstantInt and we can retrive actual size as uint64_t
				// Replace "alloca type, 10" with "@a=[type x 10]; bitcast [type x 10]* @a to type*"
				uint64_t numElements = cast<ConstantInt>(alloca->getArraySize())->getZExtValue();
				ArrayType *arrayType = ArrayType::get(alloca->getAllocatedType(),numElements);
				GlobalVariable* GV = new GlobalVariable(
					*m, arrayType, false, GlobalValue::PrivateLinkage,
					Constant::getNullValue(arrayType), "memoryForAlloca");
				GV->setAlignment(alloca->getAlignment());
				BitCastInst* bitcast = new BitCastInst(GV,alloca->getType(),
					"replacementOfAlloca", alloca);
				alloca->replaceAllUsesWith(bitcast);

				VERBOSE(Verbose::Alloca << Verbose::Green <<
						"Replace \"" << *alloca << "\" with \n  " << *GV << *bitcast << "\n" <<
						Verbose::Reset << Verbose::Default);
			} 
			else  {
				// More complex case: array size is a common value and unknown at compile-time
				// Replace "alloca type, count" with "%0=call i8* kernelgen_malloc(sizeof(type)*count)
				// %1 = bitcast i8* %0 to type*"

				IRBuilder<> Builder(alloca);
				Constant *sizeOfElement = ConstantExpr::getSizeOf(alloca->getAllocatedType());
				Value *sizeOfAllocation = Builder.CreateMul(alloca->getArraySize(),sizeOfElement, "allocationSize");
				Function *kernelgenMalloc =  m->getFunction("kernelgen_malloc");
				assert(kernelgenMalloc && sizeOfAllocation->getType()->isIntegerTy());
				CallInst *callOfMalloc = (CallInst *)Builder.CreateCall(
					kernelgenMalloc, sizeOfAllocation, "memoryForAlloca");
				callOfMalloc->setCallingConv(llvm::CallingConv::PTX_Device);
				
				if(callOfMalloc->getType() == alloca->getType()) {
     				callOfMalloc->setName("replacementOfAlloca");
					alloca->replaceAllUsesWith(callOfMalloc);
					VERBOSE(Verbose::Alloca << Verbose::Green <<
							"Replace \"" << *alloca << "\" with \n" << *callOfMalloc << "\n" <<
							Verbose::Reset << Verbose::Default);
				} 
				else {
					Value * bitcast = Builder.CreateBitCast(callOfMalloc, alloca->getType(), "replacementOfAlloca");
					alloca->replaceAllUsesWith(bitcast);
					VERBOSE(Verbose::Alloca << Verbose::Green <<
							"Replace \"" << *alloca << "\" with \n" << *callOfMalloc << "\n" <<
							*bitcast << "\"\n" << Verbose::Reset << Verbose::Default);
				}
			}
			alloca->eraseFromParent();
		}
}

static bool processCallTreeLoop(Kernel* kernel, Module* m, Function* f)
{
	for (Function::iterator bb = f->begin(); bb != f->end(); bb++)
		for (BasicBlock::iterator ii = bb->begin(), ie = bb->end(); ii != ie; ii++) {
			
			// Check if instruction in focus is a AllocaInst.
			AllocaInst *alloca = dyn_cast<AllocaInst>(ii);
			if(alloca)
				if(alloca->isArrayAllocation())
					if(!isa<Constant>(*alloca->getArraySize())) {
						VERBOSE(Verbose::Red <<
								"\n    FAIL: Not allowed dynamic alloca in loop: " <<
								*alloca << "\n" << Verbose::Reset);
						kernel->target[RUNMODE].supported = false;
						return false;
					}
				
			// Check if instruction in focus is a call.
			CallInst* call = dyn_cast<CallInst>(cast<Value>(ii));
			if (!call) continue;

			// Mark all calls as calls to device functions. 
			call->setCallingConv(CallingConv::PTX_Device);

			// Check if function is called (needs -instcombine pass).
			Function* callee = dyn_cast<Function>(
				call->getCalledValue()->stripPointerCasts());
			if (!callee) continue;

			// If function is defined (not declared), then recursively
			// process its body.
			if (!callee->isDeclaration())
			{
				if (!processCallTreeLoop(kernel, m, callee))
					return false;
				//call -> setAttributes(callee -> getAttributes());
				continue;
			} 
		
			// If function is only declared, then try to find and
			// insert its implementation from the CUDA runtime module.
			Function* Dst = callee;
			Function* Src = cuda_module->getFunction(callee->getName());
			if (Src)
			{
				VERBOSE("Device call: " << callee->getName().data() << "\n");
				call->setCallingConv(CallingConv::PTX_Device);					
				linkFunctionWithAllDependendes(Src, Dst);
				Dst->setName(Dst->getName());
				//Dst->setAttributes(Src -> getAttributes());
				continue;
			}

			if (callee->isIntrinsic()) continue;

			// Loop kernel contains non-native calls, and therefore
			// cannot be executed on GPU.
			VERBOSE(Verbose::Red <<
					"\n    FAIL: Not allowed host call in loop: " << callee->getName().data() <<
					"\n" << Verbose::Reset);
			kernel->target[RUNMODE].supported = false;
			return false;
		}
	return true;
}

KernelFunc kernelgen::runtime::Compile(
    int runmode, Kernel* kernel, Module* module, void * data, int szdata, int szdatai)
{
	// Do not compile, if no source.
	if (kernel->source == "")
		return kernel->target[runmode].binary;

	VERBOSE(Verbose::Blue <<
			"\n<------------------ "<< kernel->name << ": compile started --------------------->\n" <<
			Verbose::Reset);

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
    
	if (kernel->name != "__kernelgen_main")
		substituteGlobalsByTheirAddresses(m);
	
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
			if (runmode == RUNMODE)
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
		VERBOSE(Verbose::Sources << *m << Verbose::Default);
 
		return Codegen(runmode, kernel, m);
	}
	case KERNELGEN_RUNMODE_CUDA : {

		// as for KernelGen device runtime module.
		m->setTargetTriple(runtime_module->getTargetTriple());
		m->setDataLayout(runtime_module->getDataLayout());

		// Mark all module functions as device functions.
		for (Module::iterator F = m->begin(), FE = m->end(); F != FE; F++)
			F->setCallingConv(CallingConv::PTX_Device);

		// Mark kernel as GPU global function.
		f->setCallingConv(CallingConv::PTX_Kernel);

		// Copy attributes for declarations from cuda_module.
		for (Module::iterator func = m->begin(), funce = m->end(); func != funce; func++) {
			if(func->isDeclaration()) {
				Function *Src = cuda_module->getFunction(func->getName());
				if(!Src) continue;

				func->setAttributes(Src->getAttributes());
			}
		}

		dim3 blockDim(1, 1, 1);
		dim3 gridDim(1, 1, 1);
		if (kernel->name != "__kernelgen_main")  {	
		
			// Substitute integer and pointer arguments.
			if (szdatai != 0) ConstantSubstitution(f, data);
            
			// Add ReadNone attribute to calls (Polly workaround).
			for (Module::iterator func = m->begin(), funce = m->end(); func != funce; func++) {
				if(func->isDeclaration()) {
  					Function *Src = cuda_module->getFunction(func->getName());
					if(!Src) continue;

					for(Value::use_iterator use_iter = func->use_begin(), use_iter_end = func->use_end();
						use_iter != use_iter_end; use_iter++)
					{
						CallInst* call = cast<CallInst>(*use_iter);

						const AttrListPtr attr = func->getAttributes();
						const AttrListPtr attr_new = attr.addAttr(
							~0U /*attr.getNumSlots()*/, Attribute::ReadNone);
						call->setAttributes(attr_new);
					}
				}
			}

			//printModuleToFile(m, kernel->name + (string)"_before_polly.txt" );
			
			// Apply the Polly codegen for CUDA target.
			Size3 sizeOfLoops;
			bool isThereAtLeastOneParallelLoop = false;
			runPollyCUDA(kernel, &sizeOfLoops, &isThereAtLeastOneParallelLoop);

			// Remove ReadNone attribute from calls (Polly workaround).
			for (Module::iterator func = m->begin(), funce = m->end(); func != funce; func++) {
				for(Value::use_iterator use_iter = func->use_begin(), use_iter_end = func->use_end();
					use_iter != use_iter_end; use_iter++)
				{
					CallInst* call = cast<CallInst>(*use_iter);
					call->setAttributes(func->getAttributes());
				}
			}

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
				if(!isThereAtLeastOneParallelLoop)
				    kernel->target[runmode].supported = false;
				return NULL;
			}
			assert(isThereAtLeastOneParallelLoop);

			// If the target kernel is loop, do not allow host calls in it.
			// Also do not allow malloc/free, probably support them later.
			// TODO: kernel *may* have kernelgen_launch called, but it must
			// always evaluate to -1.
			if (!processCallTreeLoop(kernel, m, f))
				return NULL;
	
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

			//printModuleToFile(m, kernel->name + (string)"_after_polly.txt" );

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
			//printModuleToFile(m, kernel->name + (string)"_after_substitution.txt" );
		}

		kernel->target[KERNELGEN_RUNMODE_CUDA].gridDim = gridDim;
		kernel->target[KERNELGEN_RUNMODE_CUDA].blockDim = blockDim;

		// Convert external functions CallInst-s into
		// host callback form. Do not convert CallInst-s
		// to intrinsics and calls linked from the CUDA device runtime module.
		if (kernel->name == "__kernelgen_main") {

			// Link entire module with the KernelGen device runtime module.
			string linker_err;
			if (Linker::LinkModules(m, runtime_module, Linker::PreserveSource, &linker_err))
			{
				THROW("Error linking runtime with kernel " << kernel->name << " : " << linker_err);
			}

			// Process the function calls tree with main function in root.
			for (Module::iterator F = m->begin(), FE = m->end(); F != FE; F++)
			    processFunctionFromMain(kernel, m, F);

			// Evaluate ConstantExpr::SizeOf to integer number ConstantInt
			PassManager manager;
			manager.add(new TargetData(m));
			manager.add(createInstructionCombiningPass());
			manager.run(*m);
			
			// Replace all allocas for kernelgen_hostcalls by one big global variable
			Function* kernelgenFunction = NULL;
			kernelgenFunction = m->getFunction("kernelgen_hostcall");
			if(kernelgenFunction)
			{				
				Value * tmpArg = NULL;
				unsigned long long maximumSizeOfData = 0;
				
				// Set of allocas we want to collect together
				list<Value *> allocasForArgs;
				allocasForArgs.clear();

				// Collect allocas for kernelgen_hostcall-s
				getAllocasAndMaximumSize(kernelgenFunction, &allocasForArgs, &maximumSizeOfData);

				// Allocate array [i8 x maximumSizeOfData]
				Type* allocatedType = ArrayType::get(Type::getInt8Ty(context),maximumSizeOfData);
				GlobalVariable *collectiveAlloca = new GlobalVariable(
					*m, allocatedType, false, GlobalValue::PrivateLinkage,
					Constant::getNullValue(allocatedType), "memoryForHostcallArgs");
				collectiveAlloca->setAlignment(4096);
					
				for(list<Value *>::iterator iter = allocasForArgs.begin(),
					iter_end = allocasForArgs.end(); iter != iter_end; iter++ ) {
					AllocaInst* allocaInst = cast<AllocaInst>(*iter);

					// Get type of old alloca
					Type* structPtrType = allocaInst->getType();

					// Create bit cast of created alloca for specified type
					BitCastInst* bitcast = new BitCastInst(collectiveAlloca,
						structPtrType, "ptrToArgsStructure");

					// Insert after old alloca
					bitcast->insertAfter(allocaInst);

					// Replace uses of old alloca with created bit cast
					allocaInst->replaceAllUsesWith(bitcast);

					// Erase old alloca from parent basic block
					allocaInst->eraseFromParent();
				}
			}

			// Align all globals to 4096.
			for (Module::global_iterator GV = m->global_begin(), GVE = m->global_end(); GV != GVE; GV++)
				GV->setAlignment(4096);
		}

		// Optimize only loop kernels.
		if (kernel->name != "__kernelgen_main")
		{
			// Internalize globals in order to let them get removed from
			// the optimized module.
			for (Module::iterator iter = m->begin(), iter_end = m->end();
				iter != iter_end; iter++)
				if (!iter->isDeclaration() && cast<Function>(iter) != f)
					iter->setLinkage(GlobalValue::LinkerPrivateLinkage);
			for (Module::global_iterator iter = m->global_begin(),
				iter_end = m->global_end();  iter != iter_end; iter++)
				if (!iter->isDeclaration() && cast<GlobalVariable>(iter) != GVSig)
					iter->setLinkage(GlobalValue::LinkerPrivateLinkage);
						  
			PassManager manager;
			PassManagerBuilder builder;

			// Do not inline anything in loop kernels. All kernel function
			// dependencies will be taken from the main kernel module.
			builder.Inliner = 0;
			
			builder.OptLevel = 3;
			builder.SizeLevel = 3;
			builder.DisableSimplifyLibCalls = true;
			builder.Vectorize = false;
			builder.populateModulePassManager(manager);
	
			manager.run(*m);
		}
		else
		{
			PassManager MPM;
			MPM.add(createGlobalOptimizerPass());     // Optimize out global vars
			MPM.add(createStripDeadPrototypesPass()); // Get rid of dead prototypes
			MPM.run(*m);
		}

		verifyModule(*m);
		VERBOSE(Verbose::Sources << *m << Verbose::Default);

		return Codegen(runmode, kernel, m);
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
