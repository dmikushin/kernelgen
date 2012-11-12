#include "KernelVerification.h"

#define DEBUG_TYPE "kernel-verification"

STATISTIC(ValidFunction, "Number of valid functions");
STATISTIC(InvalidFunction, "Number of invalid functions");

#define BAD_STAT(NAME, DESC) STATISTIC(Bad##NAME##InFunction, \
                                       "function is not valid: "\
                                       DESC)

#define INVALID(NAME, MESSAGE) \
	do { \
		std::string Buf; \
		raw_string_ostream fmt(Buf); \
		fmt << MESSAGE; \
		fmt.flush(); \
		DEBUG(dbgs() << MESSAGE); \
		DEBUG(dbgs() << "\n"); \
		++Bad##NAME##InFunction; \
		return false; \
	} while (0);

BAD_STAT(CFG,             "CFG too complex");
BAD_STAT(IndVar,          "Non canonical induction variable in loop");
BAD_STAT(LoopBound,       "Loop bounds can not be computed");
BAD_STAT(FuncCall,        "Function call with side effects appeared");
BAD_STAT(AffAccess,       "Access function not affine");
BAD_STAT(AffCondition,    "Condition not affine");
BAD_STAT(Condition,       "Bad condition");
BAD_STAT(CastInst,        "Found cast instruction");
BAD_STAT(IntToPtr,        "Bad int-to-ptr instruction");
BAD_STAT(Alloca,          "Non-static alloca");
BAD_STAT(Exit,            "More than one exit block");
BAD_STAT(CallGraph,       "Cycle in call graph");
BAD_STAT(BadDecl,         "Call of not readnone function declaration");
BAD_STAT(AccessPointer,   "Problems with base pointer of memoty access");
BAD_STAT(PHI,             "PHINode whitch is not a induction variable");



namespace kernelgen
{

ParsingType parsingType;

SCEVTreeNode::SCEVTreeNode(Function *F)
	:invokeCallInst(NULL),returnInst(NULL), parentNode(NULL), f(F), outerLoop(NULL)
{
	createAnalysisDataFromFunction(F);
	TD = new TargetData(f->getParent());
	TLI = new TargetLibraryInfo();

	SE = new ScalarEvolution();
	SE->setAnalysisData(LI, TD, TLI,DT);
	SE->setFunction(f);

	rootLI = LI;
	rootDT = DT;
}
SCEVTreeNode::SCEVTreeNode(Function *F, ScalarEvolution *SE, DominatorTree *DT, LoopInfo *LI)
:f(F), SE(SE), DT(DT), LI(LI) {}

SCEVTreeNode::SCEVTreeNode(Function *F, CallInst * invokeCallInst, SCEVTreeNode *parentNode)
	:invokeCallInst(invokeCallInst),returnInst(NULL), parentNode(parentNode), f(F), rootLI(parentNode->rootLI), rootDT(parentNode->rootDT),
	 TD(parentNode->TD), TLI(parentNode->TLI)
{
	createAnalysisDataFromFunction(F);

	SE = new ScalarEvolution();
	SE->setAnalysisData(rootLI, TD, TLI, rootDT);
	SE->setFunction(F);

	Loop *tmp = parentNode->LI->getLoopFor(invokeCallInst -> getParent());
	outerLoop = (tmp != NULL)?tmp : parentNode->outerLoop;

}
void SCEVTreeNode::insertActualSCEVForArguments()
{
	assert(invokeCallInst && parentNode && f);

	for(Function::arg_iterator argumentIter = f->arg_begin(), argumentEnd = f->arg_end();
	    argumentIter != argumentEnd; argumentIter++) {
		Argument *argument = &(*argumentIter);
		if(SE->isSCEVable(argument->getType())) {
			Value *actualArgument = invokeCallInst -> getArgOperand(argument -> getArgNo());
			const SCEV * actualSCEV = parentNode->SE->getSCEV(actualArgument);
			SE->setSCEVForValue( &(*argument) , actualSCEV);
		}
	}
}
void SCEVTreeNode::updateRootDominatorTree(DomTreeNode *currentNode)
{
	typedef GraphTraits<DomTreeNode *> DominatorTraits;

	for(DominatorTraits::ChildIteratorType iterator = DominatorTraits::child_begin(currentNode),
	    iteratorEnd = DominatorTraits::child_end(currentNode); iterator != iteratorEnd; iterator++) {
		rootDT->addNewBlock((*iterator)->getBlock(), currentNode->getBlock());
		updateRootDominatorTree(*iterator);
	}
}
void SCEVTreeNode::updateAnalysis()
{
	assert(parentNode && LI && DT);

	if(outerLoop) {
		//add all top level loops as subloops of outer loop
		for(LoopInfo::iterator topLevelLoop = LI->begin(), topLevelLoopEnd = LI->end();
		    topLevelLoop != topLevelLoopEnd; topLevelLoop++) {
			outerLoop->addChildLoop(*topLevelLoop);
		}
		// addChildLoop does not update list of basic blocks of outer loop
		// do it manually
		for(Function::iterator basicBlock = f->begin(), basicBlockEnd = f->end();
		    basicBlock != basicBlockEnd; basicBlock++)
			outerLoop->addBasicBlockToLoop(basicBlock, rootLI->getBase());

	} else {
		for(LoopInfo::iterator topLevelLoop = LI->begin(), topLevelLoopEnd = LI->end();
		    topLevelLoop != topLevelLoopEnd; topLevelLoop++) {
			rootLI->addTopLevelLoop(*topLevelLoop);
		}

	}

	// now need to update block-to-loop map in parent LoopInfo
	for(Function::iterator basicBlock = f->begin(), basicBlockEnd = f->end();
	    basicBlock != basicBlockEnd; basicBlock++) {
		if(Loop *innerMostLoopForBB = LI->getLoopFor(basicBlock))
			rootLI->changeLoopFor(basicBlock,innerMostLoopForBB);
	}

	// update dominator tree

	DomTreeNode *callNode = rootDT->getNode(invokeCallInst->getParent());
	DomTreeNode *callIDom = callNode->getIDom();
	rootDT->addNewBlock(&f->getEntryBlock(), callIDom->getBlock());
	updateRootDominatorTree(DT->getRootNode());

	const vector<DomTreeNode *> childs(callNode->getChildren().begin(), callNode->getChildren().end());
	DomTreeNode *exitNode = rootDT->getNode(returnInst->getParent());
	assert(exitNode);
	for(unsigned i =0; i < childs.size(); i++)
		rootDT->changeImmediateDominator(childs[i], exitNode);
	rootDT->eraseNode(invokeCallInst -> getParent());
	assert(true);
}
void SCEVTreeNode::reestablishAnalysis()
{
	assert(parentNode && LI && DT);
	//если outerLoop не равен нулю, то циклы были добавлены в него, иначе - как topLevel
	if(outerLoop)
		for(LoopInfo::iterator topLoop = LI->begin(), topLoopEnd = LI->end();
		    topLoop != topLoopEnd; topLoop++) {
			for(Loop::iterator child = outerLoop->begin(),childEnd = outerLoop->end();
			    child!=childEnd; child++)
				if(*child == *topLoop)
					outerLoop->removeChildLoop(child);
		}
	else {
		bool removed = false;
		for(LoopInfo::iterator topLoop = LI->begin(), topLoopEnd = LI->end();
		    topLoop != topLoopEnd; topLoop++) {
			removed = true;
			for(LoopInfo::iterator parentTopLoop = rootLI->begin(), parentTopLoopEnd = rootLI->end();
			    parentTopLoop != parentTopLoopEnd; parentTopLoop++)
				if(*parentTopLoop == *topLoop) {
					rootLI->removeLoop(parentTopLoop);
					removed=true;
				}
			assert(removed);
		}
	}

	// now need to remove blocks of current function from outer loops
	// and remove block from map
	for(Function::iterator basicBlock = f->begin(), basicBlockEnd = f->end();
	    basicBlock != basicBlockEnd; basicBlock++) {
		if(outerLoop) {
			// outer loop exists
			// need to remove blocks from it and all it's parent loops
			rootLI->changeLoopFor(basicBlock,outerLoop);
			rootLI->removeBlock(basicBlock);
		} else
			// erase basic block from block-to-loop map
			rootLI->changeLoopFor(basicBlock,NULL);
	}

	//verify our work
	DenseSet<const Loop*> Loops;
	for(LoopInfo::iterator topLevelLoop = parentNode->LI->begin(), topLevelLoopEnd = parentNode->LI->end();
	    topLevelLoop != topLevelLoopEnd; topLevelLoop++) {
		Loops.clear();
		(*topLevelLoop)->verifyLoopNest(&Loops);
	}
	for(LoopInfo::iterator topLevelLoop = LI->begin(), topLevelLoopEnd = LI->end();
	    topLevelLoop != topLevelLoopEnd; topLevelLoop++) {
		Loops.clear();
		(*topLevelLoop)->verifyLoopNest(&Loops);
	}

	// update dominator tree

	DomTreeNode *entryNode = rootDT->getNode(&f->getEntryBlock());
	DomTreeNode *exitNode = rootDT->getNode(returnInst->getParent());
	assert(exitNode && entryNode);

	const vector<DomTreeNode *> childs(exitNode->getChildren().begin(), exitNode->getChildren().end());

	DomTreeNode *callIDom = entryNode->getIDom();
	DomTreeNode *callNode = rootDT -> addNewBlock(invokeCallInst -> getParent(), callIDom->getBlock());
	for(unsigned i =0; i < childs.size(); i++)
		rootDT->changeImmediateDominator(childs[i], callNode);
	assert(exitNode -> getNumChildren() == 0);

	entryNode->clearAllChildren();
	rootDT->eraseNode(entryNode->getBlock());

	DomTreeNode *currentNode = NULL;
	for(Function::iterator block = ++f->begin(), blockEnd = f->end();
	    block != blockEnd; block++) {
		currentNode = rootDT -> getNode(block);
		rootDT ->getBase().removeNode(block);
		delete currentNode;
	}
	assert(true);
}
void SCEVTreeNode::eraseActualSCEVForArguments()
{
	if(invokeCallInst && parentNode) {
		Function *f = invokeCallInst -> getCalledFunction();
		assert(f);
		for(Function::arg_iterator argumentIter = f->arg_begin(), argumentEnd = f->arg_end();
		    argumentIter != argumentEnd; argumentIter++) {
			Argument *argument = argumentIter;
			if(SE->isSCEVable(argument->getType()))
				SE->eraseSCEVForValue( &(*argument) );

		}
	}
	SE->eraseSCEVsNotInFunction(f);
	assert(true);
}
void SCEVTreeNode::freeTree(SCEVTreeNode *&node)
{
	//assert(root->invokeCallInst == NULL && root->parentNode == NULL);
	for(std::map<CallInst*,SCEVTreeNode *>::iterator child = node->childNodes.begin(), childEnd = node->childNodes.end();
	    child != childEnd; child++)
		freeTree(child->second);

    if(node -> parentNode)
	    node->eraseActualSCEVForArguments();
	node->SE->releaseMemory();
	delete node->SE;

	node->DT->releaseMemory();
	delete node->DT;
	node->LI->releaseMemory();
	delete node->LI;

	if( !node -> parentNode) {
		node->TD->releaseMemory();
		delete node->TD;
		node->TLI->releaseMemory();
		delete node->TLI;
	}


	delete node;

	node = NULL;
}

// function must have only one exit block
// find it and store return instruction in current leaf of ScalarEvolution-tree
bool KernelVerification::isOneReturnBlock()
{
	SCEVTreeLeaf->returnInst = NULL;
	for(llvm::Function::iterator basicBlock=f->begin(), basicBlockEnd=f->end();
	    basicBlock != basicBlockEnd; basicBlock ++) {

		if(isa<ReturnInst>(*basicBlock->getTerminator())) {
			if(SCEVTreeLeaf->returnInst != NULL)
				// function has more than one instrunction
				INVALID(Exit, "More than one exit block in function" <<  basicBlock ->getName() );
			//store return instruction in current leaf of ScalarEvolution-tree
			SCEVTreeLeaf->returnInst = cast<ReturnInst>(basicBlock->getTerminator());
		}
	}
	return(SCEVTreeLeaf->returnInst != NULL);
}

//check Call graph, because cycles(including recursion) in call-graph are not allowed
bool KernelVerification::isValidCallGraph(Function *calledFunction)
{
	SCEVTreeNode *node = SCEVTreeLeaf-> parentNode;
	while(node != NULL) {
		Function * previousCalledFunction = node->f;
		if(previousCalledFunction == calledFunction)
			INVALID(CallGraph, "Cycle in call graph" << f->getName());
		node = node->parentNode;
	}
	return true;
}

// check treminator instrunction, which forms control flow
bool KernelVerification::isValidControlFlow(BasicBlock *basicBlock)
{
	// get the terminator instrunction of the block
	TerminatorInst *TI = basicBlock->getTerminator();

	// return instrunction must be checked earlier
	if(isa<ReturnInst>(*TI))
		return true;

	BranchInst *Br = dyn_cast<BranchInst>(TI);

	// switch instrunctions are not allowed
	if (!Br)
		INVALID(CFG, "Non branch instruction terminates BB: " + basicBlock->getName());

	// unconditional brach instrunction does not influence the control flow
	if (Br->isUnconditional())
		return true;

	Value *Condition = Br->getCondition();

	// UndefValue is not allowed as a condition.
	if (isa<UndefValue>(Condition))
		INVALID(Condition, "Condition based on 'undef' value in BB: " + basicBlock->getName());

	// Only Constant and ICmpInst are allowed as condition.
	if (!(isa<Constant>(Condition) || isa<ICmpInst>(Condition)))
		INVALID(Condition, "Condition in BB '" + basicBlock->getName() + "' neither ""constant nor an icmp instruction");

	// Allow perfectly nested conditions
	assert(Br->getNumSuccessors() == 2 && "Unexpected number of successors");

	if (ICmpInst *ICmp = dyn_cast<ICmpInst>(Condition)) {
		// Unsigned comparisons are not allowed. They trigger overflow problems
		// in the code generation.
		//
		// TODO: This is not sufficient and just hides bugs. However it does pretty
		// well.
		if(ICmp->isUnsigned())
			assert(false);


		if (isa<UndefValue>(ICmp->getOperand(0))
		    || isa<UndefValue>(ICmp->getOperand(1)))
			INVALID(Condition, "undef operand in branch at BB: " + basicBlock->getName());

		const SCEV *LHS = this->SCEVTreeLeaf->SE->getSCEV(ICmp->getOperand(0));
		const SCEV *RHS = this->SCEVTreeLeaf->SE->getSCEV(ICmp->getOperand(1));

		// Are both operands of the ICmp affine?
		if (!isAffineExpr(NULL, LHS, *this->SCEVTreeLeaf->SE) ||
		    !isAffineExpr(NULL, RHS, *this->SCEVTreeLeaf->SE))
			INVALID(AffCondition, "Non affine branch in BB '" << basicBlock->getName() << "' with LHS: " << *LHS << " and RHS: " << *RHS);
	}

	// Allow loop exit conditions.
	Loop *L = SCEVTreeLeaf->LI->getLoopFor(basicBlock);
	if (L && L->getExitingBlock() == basicBlock) return true;

	// Allow perfectly nested conditions.
	Region *R = RI->getRegionFor(basicBlock);
	if (R->getEntry() != basicBlock )
		INVALID(CFG, "Not well structured condition at BB: " + basicBlock->getName());

	return true;
}

// check Loop
bool  KernelVerification::isValidLoop(Loop *L)
{
	//get induction variable of loop
	PHINode *IndVar = L->getCanonicalInductionVariable();
	// No canonical induction variable.
	if (!IndVar)
		INVALID(IndVar, "No canonical IV at loop header: "<< L->getHeader()->getName());

	// Is the loop count affine?
	const SCEV *LoopCount = SCEVTreeLeaf->SE->getBackedgeTakenCount(L);
	if (!isAffineExpr(NULL/*&Context.CurRegion*/, LoopCount, *SCEVTreeLeaf->SE))
		INVALID(LoopBound, "Non affine loop bound '" << *LoopCount << "' in loop: "<< L->getHeader()->getName());

	return true;
}

// check call instrunction
bool KernelVerification::isValidCallInst(CallInst *callInst)
{
	//get called function
	Function *calledFunction = callInst -> getCalledFunction();
	if(!calledFunction)
		assert(false);

	// cycles(including recursion) in call-graph are not allowed
	if(!isValidCallGraph(calledFunction))
		return false;

	// check function body, if we can
	if(! calledFunction->isDeclaration() ) {
		bool result = true;
		llvm::FunctionPassManager manager(f->getParent());
		manager.add(new TargetData(f->getParent()));
		manager.add(new KernelVerification(SCEVTreeLeaf,callInst,&result));
		manager.run(*calledFunction);
		return result;
	}
	// function is only declared and we can not verify it
	// do not allow calls to function with side-effects
	else {
		if (callInst->mayHaveSideEffects() || callInst->doesNotReturn())
			INVALID(FuncCall, "Call instruction with side effect: " << *callInst);

		if (callInst->doesNotAccessMemory())
			return true;

		// TODO: Intrinsics.
		// Да мне кажется, что интрисики можно грубо разрешить
		INVALID(BadDecl, "Call of not readnone function declaration: " << *callInst);
	}
}

// check store/load instrunctions
bool KernelVerification::isValidMemoryAccess(Instruction *memAccess)
{
	Value *Ptr = getPointerOperand(*memAccess);
	const SCEV *AccessFunction = SCEVTreeLeaf->SE->getSCEV(Ptr);
	const SCEVUnknown *BasePointer;
	Value *BaseValue;

	// get base pointer, which we access
	BasePointer = dyn_cast<SCEVUnknown>(SCEVTreeLeaf->SE->getPointerBase(AccessFunction));

	if (!BasePointer)
		INVALID(AccessPointer, "No base pointer" << *memAccess);

	BaseValue = BasePointer->getValue();
	if (isa<UndefValue>(BaseValue))
		INVALID(AccessPointer, "Undefined base pointer" << *memAccess);

	//!!!!!!!!!!!!!!!!!!!!!
	/* && !AllowNonAffine*/
	//!!!!!!!!!!!!!!!!!!!!!

	// get access function
	AccessFunction = SCEVTreeLeaf->SE->getMinusSCEV(AccessFunction, BasePointer);

	if (!isAffineExpr(NULL/*&Context.CurRegion*/, AccessFunction, *SCEVTreeLeaf->SE, BaseValue) /* && !AllowNonAffine*/)
		INVALID(AffAccess, "Non affine access function" << *AccessFunction);

	// FIXME: Alias Analysis thinks IntToPtrInst aliases with alloca instructions
	// created by IndependentBlocks Pass.

	//!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
	// Мне кажется, это не актуально !!
	//!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

	// FIXME: Alias Analysis thinks IntToPtrInst aliases with alloca instructions
	// created by IndependentBlocks Pass.
	if (isa<IntToPtrInst>(BaseValue))
		INVALID(IntToPtr, "Find bad intToptr prt: " << *BaseValue);

	return true;
}

// check instruction
bool KernelVerification::isValidInstruction(Instruction *inst)
{

	// every PHI-node must be canonical induction variable
	if (PHINode *PN = dyn_cast<PHINode>(inst))
		if (!isIndVar(PN, SCEVTreeLeaf->LI))
			INVALID(PHI, "Non canonical PHI node: " << *inst);

	//проверить,что все параметры скопа являются аргументами функции

	//if (hasScalarDependency(*inst)
	//    INVALID(Scalar, "Scalar dependency found: " << *inst);

	// check call inst
	if (CallInst *callInst = dyn_cast<CallInst>(inst))
		return isValidCallInst(callInst);

	//
	if (!inst->mayWriteToMemory() && !inst->mayReadFromMemory()) {
		// Handle cast instruction.
		// Надо проверить
		// Возможно, ScalarEvolution не переваривает биткасты
		// В таком случае, это ограничение можно снять (если от этого не испортится работа TempScopInfo)

		// ScalarEvolution вполне нормально переваривает биткасты ( SCEV для их аргументов)
		// IRAccess::Size вытаскивается из load/store инструкции

		if (isa<IntToPtrInst>(*inst) || isa<BitCastInst>(*inst))
			INVALID(CastInst, "Cast instruction: " << *inst);

		// only static alloca-s allowed
		if (isa<AllocaInst>(*inst)) {
			if(cast<AllocaInst>(inst)->isStaticAlloca())
				// can entry block of function be a part of loop?
				assert(SCEVTreeLeaf->LI->getLoopFor(inst->getParent()) == NULL);
			else
				INVALID(Alloca, "Non-static alloca instruction: " << *inst);
		}
		return true;
	}

	// Check the access function.
	if (isa<LoadInst>(*inst) || isa<StoreInst>(*inst))
		return isValidMemoryAccess(inst);

	// We do not know this instruction, therefore we assume it is invalid.
	return false;
}

// check subloops of function
bool KernelVerification::isAllSubloopsValid(Loop *L)
{
	typedef GraphTraits<Loop *> LoopTraits;
	for (typename LoopTraits::ChildIteratorType iter = LoopTraits::child_begin(L), iterEnd = LoopTraits::child_end(L);
	     iter != iterEnd; ++iter) {
		if(!isValidLoop(*iter) || !isAllSubloopsValid(*iter))
			return false;
	}
	return true;
}

// verify function
bool KernelVerification::isValidFunction()
{
	Loop *L = SCEVTreeLeaf->LI -> getLoopFor(SCEVTreeLeaf->returnInst -> getParent());
	assert(L == NULL);

	//!!!!!!!!!!!!!!!
	//-mergereturn !!
	//!!!!!!!!!!!!!!!

	//!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
	// что-то там с запретом на фи-узлы в exit-блоках !!
	// isValidExit                                    !!
	// hasScalarDependency                            !!
	// в isAffExpr наплевал на регион                 !!
	//!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

	// check basic blocks of function
	for(llvm::Function::iterator basicBlock=f->begin(), basicBlockEnd=f->end();
	    basicBlock != basicBlockEnd; basicBlock ++) {

		// check all instructions, except of terminator
		for (BasicBlock::iterator I = (*basicBlock).begin(), E = --(*basicBlock).end(); I != E; ++I)
			if(!isValidInstruction(I))
				return false;

		// check control flow, whitch formed by terminator instructions
		if(!isValidControlFlow(basicBlock))
			return false;
	}

	// check all loops of function
	for(LoopInfo::iterator iter = SCEVTreeLeaf->LI->begin(), iterEnd = SCEVTreeLeaf->LI->end();
	    iter != iterEnd; iter++) {
		Loop * L = *iter;
		//check top-level loop and check subloops
		if(!isValidLoop(L) || !isAllSubloopsValid(L))
			return false;
	}

	return true;
}
string stringFromBool(bool b)
{
	return b ? "true" : "false";
}
// function, called by pass manager

bool KernelVerification::runOnFunction(llvm::Function &F)
{
	RI = &getAnalysis<RegionInfo>();
	assert(parsingType == InterprocedureParsing);
	assert(!this->f && "we must run that pass only for one function, which is a kernel");
	this->f = &F;

	if(invokeCallInst == NULL) {
		// we in toppest function
		// create tree of ScalarEvolutions
		SCEVTreeLeaf = new SCEVTreeNode(this->f); // it is root if SCEVTree
		// function must have only one exit block
		// find it and store return instruction in current leaf of ScalarEvolution-tree
		isOneReturnBlock();
		assert(SCEVTreeLeaf->returnInst);

	} else {
		// we in called function
		// add leaf to tree of ScalarEvolutions
		SCEVTreeLeaf = SCEVTreeLeaf->addChild(new SCEVTreeNode(this->f, invokeCallInst, SCEVTreeLeaf));
		// function must have only one exit block
		// find it and store return instruction in current leaf of ScalarEvolution-tree
		isOneReturnBlock();
		assert(SCEVTreeLeaf->returnInst);
		// insert actual SCEV for arguments to current ScalarEvolution
		SCEVTreeLeaf->insertActualSCEVForArguments();
		SCEVTreeLeaf->updateAnalysis();
	}

	string offset = "";
	SCEVTreeNode * parent = SCEVTreeLeaf;
	while( (parent = parent->parentNode ) != NULL)
		offset += "  ";

	// verify function
	DEBUG(dbgs() << offset << "Checking function: " << f->getName() << "\n");
	verificationResult = isValidFunction();
	DEBUG(dbgs()  << offset << "result for " << this->f->getName() << ": " << stringFromBool(verificationResult) << "\n" );

	// if here specified memory for anwer, put it
	if(memForAnswer)
		*memForAnswer = verificationResult;

	if(invokeCallInst == NULL)
		DEBUG(dbgs()  << "Verification result: " << stringFromBool(verificationResult) << "\n");
	else
		SCEVTreeLeaf->reestablishAnalysis();

	if(verificationResult)
		ValidFunction++;
	else InvalidFunction++;
	return false;
}
char KernelVerification::ID = 0;

}
using namespace kernelgen;
INITIALIZE_PASS_BEGIN(KernelVerification, "kernel-verification",
                      "Kernelgen kernel verification", false, true)
INITIALIZE_PASS_DEPENDENCY(KernelPrepare)
INITIALIZE_PASS_DEPENDENCY(TargetData)
INITIALIZE_PASS_DEPENDENCY(TargetLibraryInfo)
INITIALIZE_PASS_DEPENDENCY(LoopInfo)
INITIALIZE_PASS_DEPENDENCY(RegionInfo)
INITIALIZE_PASS_DEPENDENCY(DominatorTree)
INITIALIZE_PASS_END(KernelVerification, "kernel-verification",
                    "Kernelgen kernel verification", false, true)
