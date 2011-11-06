#include "llvm/Analysis/Dominators.h"
#include "llvm/Analysis/LoopInfo.h"
namespace llvm
{
void initializeSingleBranchedLoopExtractorPass(PassRegistry&);
void initializeBranchedLoopExtractorPass(PassRegistry&);
  Function* BranchedExtractLoop(DominatorTree& DT, Loop *L,
                        bool AggregateArgs = false);
Pass *createBranchedLoopExtractorPass();
Pass *createSingleBranchedLoopExtractorPass();
						
}