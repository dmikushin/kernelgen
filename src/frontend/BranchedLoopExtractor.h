#include "llvm/Analysis/Dominators.h"
#include "llvm/Analysis/LoopInfo.h"
namespace llvm
{
void initializeSingleBranchedLoopExtractorPass(PassRegistry&);
void initializeBranchedLoopExtractorPass(PassRegistry&);
  CallInst* BranchedExtractLoop(DominatorTree& DT, Loop *L,
                        bool AggregateArgs = false);

Pass *createBranchedLoopExtractorPass();
Pass *createSingleBranchedLoopExtractorPass();
Pass *createBranchedLoopExtractorPass(std::vector<CallInst *> & LFC);
						
}