#include "CodeGeneration.h"
#include "llvm/Analysis/Verifier.h"
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

	Function * region_func = region->getEntry()->getParent();
	Module * region_module = region_func->getParent();
	string ModuleName = region_module->getModuleIdentifier();
	BasicBlock *EnteringBlock = region->getEnteringBlock();
	SetVector<BasicBlock *> region_blocks;
	SetVector<BasicBlock *> tail_blocks;
	SetVector<BasicBlock *> oldSuccessors;
	{
		////////////////////////////////////////////////////////////////////////////////////////
		// fill in set of region blocks                                                       //
		////////////////////////////////////////////////////////////////////////////////////////
		int i = 0;                                                                            //
		int firstRegionBlockInFunc = -1;                                                      //
		for(Function::iterator BB_Begin = region_func->begin(), BB_End = region_func->end();  //
		    BB_Begin != BB_End; BB_Begin++ ) {                                                //
			BasicBlock * bb = const_cast<BasicBlock*>(&(*BB_Begin));                          //
			i++;                                                                              //
			if(region->contains(bb)) {                                                        //
				region_blocks.insert(bb);                                                     //
				if(firstRegionBlockInFunc ==-1) firstRegionBlockInFunc=i;                     //
			}                                                                                 //
		}                                                                                     //
		////////////////////////////////////////////////////////////////////////////////////////
		
		////////////////////////////////////////////////////////////////////////////
		// fill in "tail" - blocks of function that are not of region blocks and  //
		// and they place is after region entry                                   //
		////////////////////////////////////////////////////////////////////////////
		i=0;                                                                      //
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
		
		for(succ_iterator Succ = succ_begin(EnteringBlock), SuccEnd = succ_end(EnteringBlock);
		    Succ != SuccEnd ; Succ ++)
		{
		    	oldSuccessors.insert(*Succ);
		}
	}
	
	// The builder will be set to startBlock.
	IRBuilder<> builder(region->getEnteringBlock());

    if (flags.CUDA)                                             
		addCUDADefinitions(builder);
	
	////////////////////////////////////////////////////////////////////////////////////////////
	// generate code                                                                          //
	////////////////////////////////////////////////////////////////////////////////////////////                
    int MaxDimensionsCount = 2;                                                               //
	kernelgen::ClastStmtCodeGen CodeGen(S, *SE, DT, SD, DP, TD, builder, MaxDimensionsCount); //
	polly::CloogInfo &C = getAnalysis<polly::CloogInfo>();                                    //
	CodeGen.codegen(C.getClast());                                                            //
	////////////////////////////////////////////////////////////////////////////////////////////
	
	parallelLoops.insert(parallelLoops.begin(),
	                     CodeGen.getParallelLoops().begin(),
	                     CodeGen.getParallelLoops().end());
						 BasicBlock * NewEnryBlock;//= *succ_begin(EnteringBlock);//wrong!
	int newSuccCount=0;
	int SuccCount = 0; 
	for(succ_iterator Succ = succ_begin(EnteringBlock), SuccEnd = succ_end(EnteringBlock);
		    Succ != SuccEnd ; Succ ++)
		{
			SuccCount++;
			if(!oldSuccessors.count(*Succ));
			{
			   newSuccCount++;
			   NewEnryBlock = *Succ;
			}
				
		}
	assert(newSuccCount==1 && "Generation added either more or less than \
	                           one new Successor Block of Entering Block");
	assert(SuccCount == oldSuccessors.size() && "Successor's count must not changed");
						
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
			bb->removeFromParent();                                                                           //
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
	BasicBlock * oldExitingBlock = region->getExitingBlock();
	BasicBlock * newExitingBlock = builder.GetInsertBlock();
	BasicBlock * ExitBlock = region->getExit();
	builder.CreateBr(ExitBlock);
	//ExitBlock->removePredecessor(oldExitingBlock);

    /////////////////////////////////////////////////////////////////////////////
	// decomment to obrain some dump information                               //
    /////////////////////////////////////////////////////////////////////////////
    // ofstream fout;                                                          //                                                        
	// fout.open((ModuleName + ".dump").c_str());                              //
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
	
	assert(!verifyFunction(*region_func) && "error at function verifying");
   
    DT->DT->recalculate(*region_func);
	Region * newRegion = new Region(NewEnryBlock, ExitBlock, RI, DT);
	//SD->markFunctionAsInvalid(region_func);

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
