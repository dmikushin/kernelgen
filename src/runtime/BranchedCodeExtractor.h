//===- BranchedCodeExtractor.cpp - Pull code region into a new function ----===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the interface to tear out a code region, such as an
// individual loop or a parallel section, into a new function, replacing it with
// a call to the new function.
//
//===----------------------------------------------------------------------===//

#include "llvm/Instructions.h"
#include "llvm/Analysis/Dominators.h"
#include "llvm/Analysis/LoopInfo.h"

namespace llvm
{
	/// ExtractBasicBlock - slurp a natural loop into a brand new function
	CallInst* BranchedExtractLoop(DominatorTree &DT, Loop *L, bool AggregateArgs);
	
	CallInst* BranchedExtractBlocks(DominatorTree &DT, 
		         std::vector<BasicBlock *> BlocksToExtract, bool AggregateArgs);
}

