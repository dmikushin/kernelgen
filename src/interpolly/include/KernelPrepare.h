
#ifndef KERNEL_PREPARE
#define KERNEL_PREPARE

#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include "llvm/PassManager.h"
#include "llvm/PassSupport.h"
#include "llvm/Instructions.h"
#include <list>
#include <set>
using namespace llvm;
using namespace std;
class KernelPrepare : public FunctionPass
{
public:
Function *f;
	static char ID;
	KernelPrepare()
		:FunctionPass(ID) {}
		
	BasicBlock* getReturnBlock() {
		Instruction *returnInst = NULL;
		for(llvm::Function::iterator basicBlock=f->begin(), basicBlockEnd=f->end();
		    basicBlock != basicBlockEnd; basicBlock ++) {

			if(isa<ReturnInst>(*basicBlock->getTerminator())) {
				if(returnInst != NULL)
					// function has more than one instrunction
					assert(false);
				//store return instruction in current leaf of ScalarEvolution-tree
				returnInst = cast<ReturnInst>(basicBlock->getTerminator());
			}
		}
		assert(returnInst != NULL);
		return returnInst->getParent();
	}


	list<CallInst *> callInsts;
	list<AllocaInst *> staticAllocas;
	template<typename ContainerType, typename ElemType> static
	typename ContainerType::iterator findInContainer(ContainerType &container, ElemType &elem) {
		typedef typename ContainerType::iterator iterator;
		iterator iter = container.begin();
		for(iterator iterEnd = container.end(); iter!=iterEnd && &(*iter) != &elem; iter++);
		return iter;
	}
	void splitBlockAtCall(CallInst *call) {
		BasicBlock *parentBlock = call->getParent();
		if( &(*parentBlock->begin()) != call)
			parentBlock = SplitBlock(parentBlock, call, this);

		if(  &(*(parentBlock->end()--)) != call ) {
			SplitBlock(parentBlock, ++findInContainer(parentBlock->getInstList(), *call), this);
		}
	}

	void splitBlockWithAllocas(BasicBlock *basicBlock) {

		bool isExistOtherIstructions = false;
		list<AllocaInst *> allocas;
		Instruction *firstNotAlloca = NULL;

		for(BasicBlock::iterator inst = basicBlock->begin(), instEnd = --basicBlock->end();
		    inst != instEnd; inst++) {

			if(AllocaInst *alloca = dyn_cast<AllocaInst>(inst)) {
				if(alloca->isStaticAlloca()) {
					allocas.push_back(alloca);
				}

			} else {
				if(!firstNotAlloca)
					firstNotAlloca = inst;
				isExistOtherIstructions=true;
			}

		}
		for(list<AllocaInst *>::reverse_iterator alloca = allocas.rbegin(),
		    allocaEnd = allocas.rend(); alloca != allocaEnd; alloca++) {
			(*alloca)->removeFromParent();
			(*alloca)->insertBefore(basicBlock->begin());
		}

		if(isExistOtherIstructions)
			SplitBlock(basicBlock, firstNotAlloca,this);
	}

	virtual bool runOnFunction(Function &F) {
		f = &F;
		for(Function::iterator block = f->begin(), blockEnd = f->end();
		    block!=blockEnd; block++)

			for(BasicBlock::iterator inst = block->begin(), instEnd = --block->end();
			    inst != instEnd; inst++) {
				if(CallInst *call = dyn_cast<CallInst>(inst))
					if(!call->getCalledFunction()->isDeclaration())
						callInsts.push_back(call);

				if(AllocaInst *alloca = dyn_cast<AllocaInst>(inst)) {
					if(alloca->isStaticAlloca())
						staticAllocas.push_back(alloca);
				}
			}

		for(list<CallInst *>::iterator call = callInsts.begin(),
		    callEnd = callInsts.end(); call != callEnd; call++)
			splitBlockAtCall(*call);

		 /*set<BasicBlock *> blocksWithStaticAllocas;
		for(list<AllocaInst *>::iterator alloca = staticAllocas.begin(),
		    allocaEnd = staticAllocas.end(); alloca != allocaEnd; alloca++)
			blocksWithStaticAllocas.insert( (*alloca)->getParent());

       assert(blocksWithStaticAllocas.size() <= 1);
		if(blocksWithStaticAllocas.size() == 1)
		   splitBlockWithAllocas(*blocksWithStaticAllocas.begin());
		else
			SplitBlock(&f->getEntryBlock(),f->getEntryBlock().begin(),this);*/


		/*BasicBlock *exitBB =  getReturnBlock();
		if(!exitBB->getSinglePredecessor())
			SplitBlock(exitBB, exitBB->begin(), this);*/
			
		return false;
	}
};


namespace llvm
{
class PassRegistry;
void initializeKernelPreparePass(llvm::PassRegistry&);
};

#endif