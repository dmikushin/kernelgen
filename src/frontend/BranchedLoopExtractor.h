#include "llvm/Analysis/Dominators.h"
#include "llvm/Analysis/LoopInfo.h"
namespace llvm
{
void initializeSingleBranchedLoopExtractorPass(PassRegistry&);
void initializeBranchedLoopExtractorPass(PassRegistry&);


Pass *createBranchedLoopExtractorPass();
Pass *createSingleBranchedLoopExtractorPass();
Pass *createBranchedLoopExtractorPass(std::vector<CallInst *> & LFC);
CallInst* BranchedExtractLoop(DominatorTree& DT,LoopInfo &LI, Loop *L);
}

