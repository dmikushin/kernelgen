//===- helperMixed.h - AsFermi parser functions ---------------------------===//
//
//     KernelGen -- A prototype of LLVM-based auto-parallelizing Fortran/C
//        compiler for NVIDIA GPUs, targeting numerical modeling code.
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains AsFermi parser functions.
//
//===----------------------------------------------------------------------===//

#ifndef helperParseDefined
#define helperParseDefined

void hpParseBreakDirectiveIntoParts();
void hpParseBreakInstructionIntoComponents();
int hpParseComputeInstructionNameIndex(SubString &name);
int hpParseFindInstructionRuleArrayIndex(int Index);
int hpParseComputeDirectiveNameIndex(SubString &name);
int hpParseFindDirectiveRuleArrayIndex(int Index);
void hpParseApplyModifier(ModifierRule &rule);
void hpParseProcessPredicate();

#else
#endif
