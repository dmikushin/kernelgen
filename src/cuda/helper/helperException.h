//===- helperException.h - AsFermi exceptions handlers --------------------===//
//
//     KernelGen -- A prototype of LLVM-based auto-parallelizing Fortran/C
//        compiler for NVIDIA GPUs, targeting numerical modeling code.
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains AsFermi exceptions handlers.
//
//===----------------------------------------------------------------------===//

#ifndef helperExceptionDefined
#define helperExceptionDefined

void hpExceptionHandler(int e);
void hpInstructionErrorHandler(int e);
void hpDirectiveErrorHandler(int e);
void hpWarning(int e);

#else
#endif
