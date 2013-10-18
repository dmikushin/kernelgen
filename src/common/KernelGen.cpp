//===- KernelGen.cpp - core KernelGen API implementation ------------------===//
//
//     KernelGen -- A prototype of LLVM-based auto-parallelizing Fortran/C
//        compiler for NVIDIA GPUs, targeting numerical modeling code.
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements basic KernelGen API.
//
//===----------------------------------------------------------------------===//

#include "KernelGen.h"

kernelgen::Settings kernelgen::settings;
