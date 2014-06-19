//===- Launch.cpp - Kernels launcher API ----------------------------------===//
//
//     KernelGen -- A prototype of LLVM-based auto-parallelizing Fortran/C
//        compiler for NVIDIA GPUs, targeting numerical modeling code.
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements public interface for getting KernelGen work stream.
//
//===----------------------------------------------------------------------===//

#include "Runtime.h"

void kernelgen_get_stream(void** stream)
{
	if (stream)
		*stream = kernelgen::runtime::cuda_context->getSecondaryStream();
}
