//===- kernelgen-version.c - KernelGen version printer --------------------===//
//
//     KernelGen -- A prototype of LLVM-based auto-parallelizing Fortran/C
//        compiler for NVIDIA GPUs, targeting numerical modeling code.
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements KernelGen version printer.
//
//===----------------------------------------------------------------------===//

#include <stdio.h>

#include "kernelgen-version.h"

int main(void) { printf("0.2nvptx/" KERNELGEN_VERSION); return 0; }
