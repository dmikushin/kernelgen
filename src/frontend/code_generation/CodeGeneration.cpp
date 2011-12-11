#include "CodeGeneration.h"
#include "llvm/Analysis/Verifier.h"
#include "BranchedLoopExtractor.h"
//using namespace polly;
using namespace llvm;
namespace kernelgen
{

Flags flags;
void set_flags(unsigned int _flags)
{
	flags.Vector = (_flags & VECTOR) != 0;
	flags.OpenMP = (_flags & OPEN_MP) != 0;
	flags.AtLeastOnce = (_flags & AT_LEAST_ONCE) != 0;
	flags.Aligned = (_flags & ALIGNED) != 0;
	flags.CUDA = (_flags & CUDA) != 0;
}
Flags & get_flags()
{
	return flags;
}
}

namespace kernelgen
{

std::vector<const char*> CudaFunctions;
DenseMap<const char*,const char *> CudaInricics;
vector<string> dimensions;
//Add definitions of functions, which returns CUDA intricics
//For each dimension defines four functions, which returns parameters
//  threadId
//	blockId
//	BlockDim
//	gridDim
// for a dimension
void CodeGeneration::addCUDADefinitions(IRBuilder<> &Builder)
{
	Module *M = Builder.GetInsertBlock()->getParent()->getParent();
	LLVMContext &Context = Builder.getContext();
	IntegerType *intType = Type::getInt64Ty(Context); //TD->getInt32Type(Context);



	if (!M->getFunction("_get_threadId_x")) {
		/////////////////////////////////////////////////////////////////////
		//  define all dimensions, that can be used while code generation  //
		/////////////////////////////////////////////////////////////////////
		dimensions.push_back("x");                                     //
		dimensions.push_back("y");                                     //
		dimensions.push_back("z");                                     //
		/////////////////////////////////////////////////////////////////////

		////////////////////////////////////////
		// define parameters of dimensions    //
		////////////////////////////////////////
		vector<string> parameters;        //
		parameters.push_back("threadId"); //
		parameters.push_back("blockId");  //
		parameters.push_back("blockDim"); //
		parameters.push_back("gridDim");  //
		////////////////////////////////////////

		string prefix1("_get_");
		string prefix2("_");
		string prefix3(".");

		for(int i = 0; i < dimensions.size(); i++)
			for(int j =0; j < parameters.size(); j++) {
				CudaFunctions.push_back((new string(prefix1 + parameters[j] + prefix2 + dimensions[i]))->c_str());
				CudaInricics[CudaFunctions.back()] = (new string(parameters[j] + prefix3 + dimensions[i]))->c_str();
			}
		for(int i = 0; i < CudaFunctions.size(); i++) {
			FunctionType *FT = FunctionType::get(intType, std::vector<Type*>(), false);
			Function::Create(FT, Function::ExternalLinkage,(CudaFunctions)[i], M);
		}
	}
}
template<class T>
bool findInVector(std::vector<T> & vec, T elem)
{

	typedef typename std::vector<T>::iterator iteratorType ;
	iteratorType iter = vec.begin(), end = vec.end();
	for(; iter != end; iter++)
		if(*iter == elem) break;
	return (iter==end);
}
void getRegionBlocks(Region *region, BasicBlock *BB, std::vector<BasicBlock*> *visited)
{
	BasicBlock *exit = region -> getExit();
	visited->push_back(BB);
	for (succ_iterator SI = succ_begin(BB), SE = succ_end(BB); SI != SE; ++SI)
		if (*SI != exit && findInVector(*visited,*SI))
			getRegionBlocks(region, *SI, visited);
}
void getRegionBlocks(Region *region, BasicBlock *BB, SetVector<BasicBlock*> *visited)
{
	BasicBlock *exit = region -> getExit();
	visited->insert(BB);
	for (succ_iterator SI = succ_begin(BB), SE = succ_end(BB); SI != SE; ++SI)
		if (*SI != exit && !visited->count(*SI))
			getRegionBlocks(region, *SI, visited);
}
BasicBlock* MakeNewEntryBlock(BasicBlock* oldEntry, BasicBlock* EnteringBlock)
{
/////////////////////////////////////////////////////////////////////////////////////////////////////////
// Add new Block which is Entry Block of new region. Redirect links                                    //
/////////////////////////////////////////////////////////////////////////////////////////////////////////
	BasicBlock * NewEntry =                                                                        //
	    BasicBlock::Create(EnteringBlock->getParent()->getContext(), "RegionEntryBlock", EnteringBlock->getParent()); //
	TerminatorInst *TI = EnteringBlock->getTerminator();                                           //
	TI->replaceUsesOfWith(oldEntry, NewEntry);                                                     //
	return NewEntry;
/////////////////////////////////////////////////////////////////////////////////////////////////////////
}
bool CodeGeneration::runOnScop(polly::Scop &scop)
{
	S = &scop;
	region = &S->getRegion();
	DT = &getAnalysis<DominatorTree>();
	polly::Dependences *DP = &getAnalysis<polly::Dependences>();
	SE = &getAnalysis<ScalarEvolution>();
	SD = &getAnalysis<polly::ScopDetection>();
	TD = &getAnalysis<TargetData>();
	RI = &getAnalysis<RegionInfo>();

	parallelLoops.clear();

	assert(region->isSimple() && "Only simple regions are supported");

	Function * region_func   = region->getEntry()->getParent();
	Module   * region_module = region_func->getParent();
	string ModuleName = region_module->getModuleIdentifier();
	LLVMContext & context = region_module -> getContext();

	BasicBlock * EnteringBlock = region->getEnteringBlock();
	BasicBlock * ExitBlock = region->getExit();

	BasicBlock * oldEntry = region->getEntry();
	BasicBlock * oldExitingBlock = region->getExitingBlock();

	SetVector<BasicBlock *> region_blocks;
	SetVector<BasicBlock *> tail_blocks;
	{
		////////////////////////////////////////////////////////////////////////////////////////
		// fill in set of region blocks                                                       //
		////////////////////////////////////////////////////////////////////////////////////////
		//int i = 0;                                                                            //
		int firstRegionBlockInFunc = 0;
		getRegionBlocks(region,oldEntry,&region_blocks);//
		for(Function::iterator BB_Begin = region_func->begin(), BB_End = region_func->end();  //
		    BB_Begin != BB_End; BB_Begin++ ) {                                                //
			BasicBlock * bb = const_cast<BasicBlock*>(&(*BB_Begin));                          //
			firstRegionBlockInFunc++;                                                                              //
			if(region->contains(bb)) break;                                                                           //
		}                                                                                     //
		////////////////////////////////////////////////////////////////////////////////////////

		////////////////////////////////////////////////////////////////////////////
		// fill in "tail" - blocks of function that are not of region blocks and  //
		// and they place is after region entry                                   //
		////////////////////////////////////////////////////////////////////////////
		int i=0;                                                                      //
		Function::BasicBlockListType & BBList = region_func->getBasicBlockList(); //
		for(Function::iterator BB_Begin = BBList.begin(), BB_End = BBList.end();  //
		    BB_Begin != BB_End; BB_Begin++ ) {                                    //
			BasicBlock * bb = BB_Begin;                                           //
			i++;                                                                  //
			if(i >= firstRegionBlockInFunc && !region->contains(bb)) {            //
				tail_blocks.insert(bb);                                           //
			}                                                                     //
		}                                                                         //
		////////////////////////////////////////////////////////////////////////////
	}

	IRBuilder<> builder(EnteringBlock);
	if (flags.CUDA)
		addCUDADefinitions(builder);
	// The builder will be set to startBlock.

	BasicBlock *newEntry = MakeNewEntryBlock(oldEntry, EnteringBlock);
	DT->addNewBlock(newEntry, EnteringBlock);
	builder.SetInsertPoint(newEntry);
	////////////////////////////////////////////////////////////////////////////////////////////
	// generate code                                                                          //
	////////////////////////////////////////////////////////////////////////////////////////////
	int MaxDimensionsCount;
	if(flags.CUDA) MaxDimensionsCount = dimensions.size();                                    //
	else MaxDimensionsCount = 0;                                                              //
	kernelgen::ClastStmtCodeGen CodeGen(S, *SE, DT, SD, DP, TD, builder, MaxDimensionsCount); //
	polly::CloogInfo &C = getAnalysis<polly::CloogInfo>();                                    //
	CodeGen.codegen(C.getClast());                                                            //
	////////////////////////////////////////////////////////////////////////////////////////////

	parallelLoops.insert(parallelLoops.begin(),
	                     CodeGen.getParallelLoops().begin(),
	                     CodeGen.getParallelLoops().end());

	BasicBlock * newExitingBlock = builder.GetInsertBlock();
	builder.CreateBr(ExitBlock);

	{
		////////////////////////////////////////////////////////////////////////////////////////////////////////////
		// remove region blocks from function                                                                     //
		////////////////////////////////////////////////////////////////////////////////////////////////////////////
		for(SetVector<BasicBlock *>::iterator BB_Begin = region_blocks.begin(), BB_End = region_blocks.end(); //
		    BB_Begin != BB_End; BB_Begin++ ) {                                                                //
			BasicBlock * bb = *BB_Begin;                                                                      //
			bb->dropAllReferences();                                                                          //
		}                                                                                                     //
		for(SetVector<BasicBlock *>::iterator BB_Begin = region_blocks.begin(), BB_End = region_blocks.end(); //
		    BB_Begin != BB_End; BB_Begin++ ) {                                                                //
			BasicBlock * bb = *BB_Begin;                                                                      //
			bb->eraseFromParent();                                                                           //
		}                                                                                                     //
		////////////////////////////////////////////////////////////////////////////////////////////////////////////

		////////////////////////////////////////////////////////////////////////////////////////////////////////
		// move tail to back of the region function                                                           //
		////////////////////////////////////////////////////////////////////////////////////////////////////////
		for(SetVector<BasicBlock *>::iterator BB_Begin = tail_blocks.begin(), BB_End = tail_blocks.end(); //
		    BB_Begin != BB_End; BB_Begin++ ) {                                                            //
			BasicBlock * bb = *BB_Begin;                                                                  //
			bb->moveAfter(&region_func->back());                                                          //
		}                                                                                                 //
		////////////////////////////////////////////////////////////////////////////////////////////////////////
	}

	DT->DT->recalculate(*region_func); //TO DO - change DT manually, bacause recalculate all is very expensive
	assert(!verifyFunction(*region_func) && "error at function verifying");

	std::vector<Value *> LaunchParameters = CodeGen.LaunchParameters;
	if(LaunchParameters.size() > 0) {
/////////////////////////////////////////////////////////////////////////////
// extract function                                                        //
/////////////////////////////////////////////////////////////////////////////
		std::vector<BasicBlock*> BlocksToExtract;                          //
		region-> replaceEntry (newEntry);                                  //
		getRegionBlocks(region, *(succ_begin(newEntry)), &BlocksToExtract);//
		//
		CallInst *LoopFunctionCall =                                       //
		    BranchedExtractBlocks(*DT, BlocksToExtract,true);              //
		SD->markFunctionAsInvalid(LoopFunctionCall->getCalledFunction());  //
		Function * LoopFunction = LoopFunctionCall -> getCalledFunction(); //
/////////////////////////////////////////////////////////////////////////////

//////////////////////////////////////////////////////////////////////////////////////////////////////
// create struct with launch parameters                                                             //
//////////////////////////////////////////////////////////////////////////////////////////////////////
		std::vector<Type*> structFields;                                                            //
		std::vector<Value*> params, StructValues;                                                   //                         //
		//
		for(int i =0; i < dimensions.size(); i++) {                                                 //
			structFields.push_back(TD->getIntPtrType(context));                                     //
			if(i < LaunchParameters.size())                                                         //
				StructValues.push_back(LaunchParameters[i]);                                        //
			else StructValues.push_back(ConstantInt::get(TD->getIntPtrType(context), 1));           //
		}                                                                                           //
		//
		structFields.push_back(LoopFunctionCall -> getArgOperand(0)->getType());                    //
		StructValues.push_back(LoopFunctionCall -> getArgOperand(0));                               //
		//
		Type * StructTy = StructType::get(context,structFields,false);                              //
		Value * Struct = new AllocaInst(StructTy,"launch_parameters",LoopFunctionCall);             //
		//
		Value *Idx[2];                                                                              //
		Idx[0] = Constant::getNullValue(Type::getInt32Ty(context));                                 //
		//
		for(int i =0; i < StructValues.size(); i++) {                                               //
			Idx[1] = ConstantInt::get(Type::getInt32Ty(context), i);                                //
			GetElementPtrInst *GEP = GetElementPtrInst::Create(Struct, Idx, "", LoopFunctionCall);  //
			new StoreInst(StructValues[i], GEP, "",	LoopFunctionCall);                              //
		}                                                                                           //
//////////////////////////////////////////////////////////////////////////////////////////////////////

//////////////////////////////////////////////////////////////////////////////////////////////////////
// function name                                                                                     //
///////////////////////////////////////////////////////////////////////////////////////////////////////
		// Create a constant array holding original called function name.                            //
		Constant* name = ConstantArray::get(context, LoopFunction->getName(), true);                 //
		//
		// Create and initialize the memory buffer for name.                                         //
		ArrayType* nameTy = cast<ArrayType>(name->getType());                                        //
		AllocaInst* nameAlloc = new AllocaInst(nameTy, "", LoopFunctionCall);                        //
		StoreInst* nameInit = new StoreInst(name, nameAlloc, "", LoopFunctionCall);                  //
		Idx[0] = Constant::getNullValue(Type::getInt32Ty(context));                                  //
		Idx[1] = ConstantInt::get(Type::getInt32Ty(context), 0);                                     //
		GetElementPtrInst* namePtr = GetElementPtrInst::Create(nameAlloc, Idx, "", LoopFunctionCall);//
///////////////////////////////////////////////////////////////////////////////////////////////////////

///////////////////////////////////////////////////////////////////////////////////////////////////
// function kernelgen_launch
//////////////////////////////////////////////////////////////////////////////////////////////////
		Function * KernelgenLaunch;
		if (!(KernelgenLaunch = region_module->getFunction("kernelgen_launch"))) {                //
			// create kernelgen_launch function
			Type * retTy = Type::getInt32Ty(context);
			std::vector<Type*> paramTy; 
			paramTy.push_back(namePtr->getType());                                                 //
			paramTy.push_back(Struct->getType()); //paramTy.push_back(StructTy);                   //
			FunctionType * KernelgenLaunchType = FunctionType::get(retTy, paramTy, false);         //
			KernelgenLaunch = Function::Create(KernelgenLaunchType, GlobalValue::ExternalLinkage,  //
			                                   "kernelgen_launch", region_module);                 //
		}
/////////////////////////////////////////////////////////////////////////////////////////////////////

		SmallVector<Value*, 16> call_args;
		call_args.push_back(namePtr);
		call_args.push_back(Struct);

		// Create new function call with new call arguments
		// and copy old call properties.
		CallInst* newcall =
		    CallInst::Create(KernelgenLaunch, call_args, "kernelgen_launch", LoopFunctionCall);
		//newcall->takeName(call);
		newcall->setCallingConv(LoopFunctionCall->getCallingConv());
		newcall->setAttributes(LoopFunctionCall->getAttributes());
		newcall->setDebugLoc(LoopFunctionCall->getDebugLoc());

		// Replace old call with new one.
		LoopFunctionCall->replaceAllUsesWith(newcall);
		LoopFunctionCall->eraseFromParent();
	}


	//recalculate DT and RI or not? Is their information about current scop needed later?
	DT->DT->recalculate(*region_func); //TO DO - change DT manually, bacause recalculate all is very expensive
	//RI->runOnFunction(*region_func); //TO DO - change RI manually, bacause recalculate all is very expensive


	/////////////////////////////////////////////////////////////////////////////
	// decomment to obrain some dump information                               //
	/////////////////////////////////////////////////////////////////////////////
	// ofstream fout;                                                          //
	// fout.open(("../examples/for.dump"));                              //
	// raw_os_ostream OS(fout);                                                //
	// region_func -> getParent() ->print(OS, NULL);                           //
	// fout.close();                                                           //
	//                                                                         //
	// static int asdf = 0;                                                    //
	// ofstream cloog_print;                                                   //
	// if(asdf==0) cloog_print.open((ModuleName + ".cloog").c_str());          //
	// else cloog_print.open( (ModuleName + ".cloog").c_str(), ios_base::app); //
	// raw_os_ostream OS1(cloog_print);                                        //
	// C.pprint(OS1);                                                          //
	/////////////////////////////////////////////////////////////////////////////
	return true;
}

void CodeGeneration::getAnalysisUsage(AnalysisUsage &AU) const
{

	AU.addRequired<polly::CloogInfo>();
	AU.addRequired<polly::Dependences>();
	AU.addRequired<DominatorTree>();
	AU.addRequired<ScalarEvolution>();
	AU.addRequired<RegionInfo>();
	AU.addRequired<polly::ScopDetection>();
	AU.addRequired<polly::ScopInfo>();
	AU.addRequired<TargetData>();

	/*AU.addPreserved<polly::CloogInfo>();
	AU.addPreserved<polly::Dependences>();

	// FIXME: We do not create LoopInfo for the newly generated loops.
	AU.addPreserved<LoopInfo>();
	AU.addPreserved<DominatorTree>();
	AU.addPreserved<polly::ScopDetection>();
	AU.addPreserved<ScalarEvolution>();

	// FIXME: We do not yet add regions for the newly generated code to the
	//        region tree.
	AU.addPreserved<RegionInfo>();
	AU.addPreserved<polly::TempScopInfo>();
	AU.addPreserved<polly::ScopInfo>();
	AU.addPreservedID(polly::IndependentBlocksID);*/
}
char CodeGeneration::ID = 1;
}

static RegisterPass<kernelgen::CodeGeneration>
Z("kernelgen-codegen", "Polly - Create LLVM-IR from the polyhedral information");

Pass * kernelgen::createCodeGenerationPass()
{
	return new CodeGeneration();
}
