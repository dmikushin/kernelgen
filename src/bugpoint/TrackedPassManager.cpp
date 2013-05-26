//===- TrackedPassManager.cpp - track failing passes with bugpoint --------===//
//
//     KernelGen -- A prototype of LLVM-based auto-parallelizing Fortran/C
//        compiler for NVIDIA GPUs, targeting numerical modeling code.
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements a PassManager hook for calling bugpoint on crashing
// passes, right during the application execution.
//
//===----------------------------------------------------------------------===//

#include "TrackedPassManager.h"

PassTracker *tracker = NULL;
