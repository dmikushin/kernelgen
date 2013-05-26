//===- BranchedLoopExtractor.h - Extract each loop into a new function ----===//
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
// program if it can, insert call to this function before loop and adds branch
// to switch between original loop execution and function call.
//
//===----------------------------------------------------------------------===//

#include "llvm/Analysis/Dominators.h"
#include "llvm/Analysis/LoopInfo.h"
namespace llvm {
void initializeSingleBranchedLoopExtractorPass(PassRegistry &);
void initializeBranchedLoopExtractorPass(PassRegistry &);

Pass *createBranchedLoopExtractorPass();
Pass *createSingleBranchedLoopExtractorPass();
Pass *createBranchedLoopExtractorPass(std::vector<CallInst *> &LFC);
CallInst *BranchedExtractLoop(DominatorTree &DT, LoopInfo &LI, Loop *L,
                              bool isBranched = true);
}
