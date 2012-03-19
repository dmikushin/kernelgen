//===- BranchedLoopExtractor.cpp - Extract each loop into a new function ----------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// A pass wrapper around the ExtractLoop() scalar transformation.
// This pass creates equivalent function for all natural loops from the
// program if it can, insert call to this fuction before loop and add branch to switch between 
// original loop execution and fuction call. If the loop is the ONLY loop in a given function,
// it is not touched. This is a pass most useful for debugging via bugpoint.
//
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "loop-extract-with-branch"

#include "llvm/Transforms/IPO.h"
#include "llvm/Instructions.h"
#include "llvm/PassSupport.h"
#include "llvm/Module.h"
#include "llvm/Pass.h"
#include "llvm/Analysis/Dominators.h"
#include "llvm/Analysis/LoopPass.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Transforms/Scalar.h"
#include "llvm/Transforms/Utils/FunctionUtils.h"
#include "llvm/ADT/Statistic.h"

#include "BranchedLoopExtractor.h"

#include <fstream>
#include <set>
#include <vector>

using namespace llvm;

STATISTIC(NumBranchedExtracted, "Number of loops extracted");

namespace {
  struct BranchedLoopExtractor  : public LoopPass {
    static char ID; // Pass identification, replacement for typeid
    unsigned NumLoops;
    std::vector<CallInst *> * LoopFunctionCalls; 
   explicit BranchedLoopExtractor(unsigned numLoops = ~0) 
      : LoopPass(ID), NumLoops(numLoops)
	  {
          LoopFunctionCalls = NULL;
		  //initializeBranchedLoopExtractorPass(*PassRegistry::getPassRegistry());
      }
      
	  explicit BranchedLoopExtractor(std::vector<CallInst *> & LFC, unsigned numLoops = ~0) 
      : LoopPass(ID), NumLoops(numLoops), LoopFunctionCalls(&LFC)
	  {
        //initializeBranchedLoopExtractorPass(*PassRegistry::getPassRegistry());
      }
	  
    virtual bool runOnLoop(Loop *L, LPPassManager &LPM);

    virtual void getAnalysisUsage(AnalysisUsage &AU) const {
      AU.addRequiredID(BreakCriticalEdgesID);
      AU.addRequiredID(LoopSimplifyID);
      AU.addRequired<DominatorTree>();
    }
  };
}
char BranchedLoopExtractor::ID = 0;
Pass *llvm::createBranchedLoopExtractorPass() { return new BranchedLoopExtractor(); }
Pass *llvm::createBranchedLoopExtractorPass(std::vector<CallInst *> & LFC) { return new BranchedLoopExtractor(LFC); }
/*INITIALIZE_PASS_BEGIN(BranchedLoopExtractor, "loop-extract-with-branch",
                "Extract loops into new functions and add branches", false, false)
INITIALIZE_PASS_DEPENDENCY(BreakCriticalEdges)
INITIALIZE_PASS_DEPENDENCY(LoopSimplify)
INITIALIZE_PASS_DEPENDENCY(DominatorTree)
INITIALIZE_PASS_END(BranchedLoopExtractor, "loop-extract-with-branch",
                "Extract loops into new functions and add branches", false, false)
*/



static void* initializeBranchedLoopExtractorPassOnce (PassRegistry &Registry) {
    initializeBreakCriticalEdgesPass(Registry);
    initializeLoopSimplifyPass(Registry);
    initializeDominatorTreePass(Registry);
    PassInfo *PI = new PassInfo("Extract loops into new functions and add branches", 
                            "loop-extract-with-branch", 
							&BranchedLoopExtractor ::ID, 
							PassInfo::NormalCtor_t(callDefaultCtor< BranchedLoopExtractor >), false, false); 
    Registry.registerPass(*PI, true); return PI; 
} 
void llvm::initializeBranchedLoopExtractorPass(PassRegistry &Registry) { 
	static volatile sys::cas_flag initialized = 0; 
	sys::cas_flag old_val = sys::CompareAndSwap(&initialized, 1, 0); 
	if (old_val == 0) { 
		initializeBranchedLoopExtractorPassOnce(Registry); 
		sys::MemoryFence(); initialized = 2; 
	} else { 
		sys::cas_flag tmp = initialized; sys::MemoryFence(); 
		while (tmp != 2) { tmp = initialized; sys::MemoryFence(); } 
	} 
}

struct staticInit
{
	staticInit()
	{
		llvm::initializeBranchedLoopExtractorPass(*PassRegistry::getPassRegistry());
	}
};

static staticInit in;

// createBranchedLoopExtractorPass - This pass extracts all natural loops from the
// program into a function if it can.
//

bool BranchedLoopExtractor::runOnLoop(Loop *L, LPPassManager &LPM) {
  // Only visit top-level loops.
  if (L->getParentLoop())
    return false;

  // If LoopSimplify form is not available, stay out of trouble.
  if (!L->isLoopSimplifyForm())
    return false;

  DominatorTree &DT = getAnalysis<DominatorTree>();
  bool Changed = false;

  // If there is more than one top-level loop in this function, extract all of
  // the loops. Otherwise there is exactly one top-level loop; in this case if
  // this function is more than a minimal wrapper around the loop, extract
  // the loop.
  bool ShouldExtractLoop = false;

  // Extract the loop if the entry block doesn't branch to the loop header.
  TerminatorInst *EntryTI =
    L->getHeader()->getParent()->getEntryBlock().getTerminator();
  if (!isa<BranchInst>(EntryTI) ||
      !cast<BranchInst>(EntryTI)->isUnconditional() ||
      EntryTI->getSuccessor(0) != L->getHeader())
    ShouldExtractLoop = true;
  else {
    // Check to see if any exits from the loop are more than just return
    // blocks.
    SmallVector<BasicBlock*, 8> ExitBlocks;
    L->getExitBlocks(ExitBlocks);
    for (unsigned i = 0, e = ExitBlocks.size(); i != e; ++i)
      if (!isa<ReturnInst>(ExitBlocks[i]->getTerminator())) {
        ShouldExtractLoop = true;
        break;
      }
  }
  if (ShouldExtractLoop) {
    if (NumLoops == 0) return Changed;
    --NumLoops;
	CallInst * Call;
    if ( (Call = BranchedExtractLoop(DT, L)) != 0) {
      Changed = true;
      // After extraction, the loop is replaced by a function call, so
      // we shouldn't try to run any more loop passes on it.
	  
	  if(LoopFunctionCalls)
	      LoopFunctionCalls->push_back(Call);
      LPM.deleteLoopFromQueue(L);
    }
    ++NumBranchedExtracted;
  }

  return Changed;
}



 

namespace {
  /// SingleLoopExtractor - For bugpoint.
  struct SingleBranchedLoopExtractor : public BranchedLoopExtractor {
    static char ID; // Pass identification, replacement for typeid
    SingleBranchedLoopExtractor() : BranchedLoopExtractor(1) {}
  };
} // End anonymous namespace

char SingleBranchedLoopExtractor::ID = 0;

INITIALIZE_PASS(SingleBranchedLoopExtractor, "loop-extract-with-branch-single",
                "Extract at most one loop into a new function and add branch", false, false)

// createSingleLoopExtractorPass - This pass extracts one natural loop from the
// program into a function if it can.  This is used by bugpoint.
//
Pass *llvm::createSingleBranchedLoopExtractorPass() {
  return new SingleBranchedLoopExtractor();
}


