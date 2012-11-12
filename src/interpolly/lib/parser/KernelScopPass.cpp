//===- ScopPass.cpp - The base class of Passes that operate on Polly IR ---===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains the definitions of the ScopPass members.
//
//===----------------------------------------------------------------------===//

#include "KernelScopPass.h"
#include "KernelScopInfo.h"

using namespace llvm;
namespace kernelgen
{
bool KernelScopPass::runOnFunction(Function &F)
{
	S = 0;
	KernelScopInfo * KSI = NULL;
	assert( KSI = &getAnalysis<KernelScopInfo>());
	S = KSI->scop;
	if(S)
		return runOnScop(*S);
	return false;
}

void KernelScopPass::print(raw_ostream &OS, const Module *M) const
{
	if (S)
		printScop(OS);
}

void KernelScopPass::getAnalysisUsage(AnalysisUsage &AU) const
{
	AU.addRequiredTransitive<KernelScopInfo>();
	AU.setPreservesAll();
}
}
