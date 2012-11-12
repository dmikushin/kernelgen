
#ifndef KERNEL_VERIFICATION
#define KERNEL_VERIFICATION

#include "llvm/Analysis/ScalarEvolution.h"
#include "llvm/Analysis/ScalarEvolutionExpressions.h"

#include "llvm/Constants.h"
#include "llvm/DerivedTypes.h"
#include "llvm/GlobalVariable.h"
#include "llvm/GlobalAlias.h"
#include "llvm/Instructions.h"
#include "llvm/LLVMContext.h"
#include "llvm/Operator.h"
#include "llvm/Analysis/ConstantFolding.h"
#include "llvm/Analysis/Dominators.h"
#include "llvm/Analysis/InstructionSimplify.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/ValueTracking.h"
#include "llvm/Analysis/RegionInfo.h"
#include "llvm/Assembly/Writer.h"
#include "llvm/Target/TargetData.h"
#include "llvm/Target/TargetLibraryInfo.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/ConstantRange.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/GetElementPtrTypeIterator.h"
#include "llvm/Support/InstIterator.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/PassManager.h"

#include "SCEVValidator.h"
#include "ScopHelper.h"


#include "KernelPrepare.h"

#include <map>
#include <set>

namespace llvm
{
   class PassRegistry;
   void initializeKernelVerificationPass(llvm::PassRegistry&);
};

namespace kernelgen 
{
struct SCEVTreeNode;
class KernelVerification;

enum ParsingType {
    InterprocedureParsing,
    CodegenParsing
};
extern ParsingType parsingType;

struct SCEVTreeNode {

	ScalarEvolution * SE;
	CallInst *invokeCallInst;
	ReturnInst *returnInst;
	std::map<CallInst*, SCEVTreeNode *> childNodes;
	SCEVTreeNode *parentNode;

	Function *f;
	LoopInfo *LI;
	DominatorTree *DT;
	Loop *outerLoop;
	LoopInfo *rootLI;
	DominatorTree *rootDT;
	TargetData *TD;
	TargetLibraryInfo *TLI;

	void insertActualSCEVForArguments();
	void eraseActualSCEVForArguments();

	void insertSCEVForCallInstToParentSE();
	void eraseSCEVForCallInstFromParentSE();
	void updateAnalysis();
	void reestablishAnalysis();

	bool verifyReturnValue();


	void createAnalysisDataFromFunction(Function *f) {
		LI = new LoopInfo();
		DT = new DominatorTree();
		DT->runOnFunction(*f);
		LI->getBase().Calculate(DT->getBase());

	}

	explicit SCEVTreeNode(Function *F);
	explicit SCEVTreeNode(Function *F, ScalarEvolution *SE, DominatorTree *DT, LoopInfo *LI);
	explicit SCEVTreeNode(Function *F, CallInst * invokeCallInst, SCEVTreeNode *parentNode);
	SCEVTreeNode * addChild(SCEVTreeNode *child) {
		childNodes[child->invokeCallInst] = child;
		return child;
	}
	static void freeTree(SCEVTreeNode *&node);
	~SCEVTreeNode() {
		//eraseActualSCEVForArguments();// Удалить из SE SCEV для аргументов
		//eraseSCEVForCallInstFromParentSE();// Удалить из родительского SE SCEV для invokeCallInst
		//SE->releaseMemory();
		//delete SE;
		DEBUG(dbgs() << "release SCEVTree Node !!\n");
	}
	void updateRootDominatorTree(DomTreeNode *currentNode);
};

class KernelVerification : public FunctionPass
{

public:
	static char ID;

	enum VerificationResult {
	    stateValidKernel,
	    stateInvalidKernel,
	    stateUnknown
	};

	bool verificationResult;
	bool *memForAnswer;

	SCEVTreeNode *SCEVTreeLeaf;
	CallInst *invokeCallInst;
	Function *f;

    typedef std::map< llvm::Instruction *, const SCEV *> accessFunctionsMap;
	std::map<SCEVTreeNode *,  accessFunctionsMap > accessFunctions;

	RegionInfo *RI;

	KernelVerification(bool *memForAnswer = NULL)
		:FunctionPass(ID), verificationResult(false), memForAnswer(memForAnswer),
		 SCEVTreeLeaf(NULL), invokeCallInst(NULL),f(NULL)  {
		initializeKernelVerificationPass(*PassRegistry::getPassRegistry());
	}

	KernelVerification(SCEVTreeNode *SCEVTreeLeaf, CallInst *invokeCallInst, bool *memForAnswer = NULL)
		:FunctionPass(ID), verificationResult(false),memForAnswer(memForAnswer),
		 SCEVTreeLeaf(SCEVTreeLeaf),invokeCallInst(invokeCallInst),f(NULL)  {
		initializeKernelVerificationPass(*PassRegistry::getPassRegistry());
	}

	void releaseMemory() {
        // we in toppest function, all called function verified, free tree
		if(invokeCallInst == NULL && SCEVTreeLeaf)
		{
			assert(parsingType == InterprocedureParsing);
		    SCEVTreeNode::freeTree(SCEVTreeLeaf); // recursively free memory, obtained by SCEVTree
			
		}
	}
	~KernelVerification() {
		//DEBUG(dbgs() << "destructor!\n");
		 assert((invokeCallInst == NULL && SCEVTreeLeaf == NULL) || invokeCallInst);//releaseMemory();
	}

	void getAnalysisUsage(AnalysisUsage &AU) const {
		/// AU.addRequired<KernelPrepare>();
		AU.addRequired<RegionInfo>();
		AU.setPreservesAll();
	}

	// function must have only one exit block
	// find it and store return instruction in current leaf of ScalarEvolution-tree
	bool isOneReturnBlock();

	//check Call graph, because cycles(including recursion) in call-graph are not allowed
	bool isValidCallGraph(Function *f);

	// check treminator instrunction, which forms control flow
	bool isValidControlFlow(BasicBlock *basicBlock);

	// check Loop
	bool  isValidLoop(Loop *L);

	// check call instrunction
	bool isValidCallInst(CallInst *callInst);

	// check store/load instrunctions
	bool isValidMemoryAccess(Instruction *memAccess);

	// check instruction
	bool isValidInstruction(Instruction *inst);

	// check subloops of function
	bool isAllSubloopsValid(Loop *L);

	// verify function
	bool isValidFunction();

    bool hasScalarDependency(Instruction &Inst);
	
	// function, called by pass manager
	virtual bool runOnFunction(llvm::Function &f);

};



}
#endif
