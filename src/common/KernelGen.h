//===- KernelGen.h - core KernelGen API implementation --------------------===//
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

#ifndef KERNELGEN_H
#define KERNELGEN_H

#include "ELF.h"
#include "Settings.h"
#include "Temp.h"

// Define to load kernels lazily. Means instead of reading and verifying
// modules during application startup, they are instead loaded in the place
// of first use.
#define KERNELGEN_LOAD_KERNELS_LAZILY

namespace kernelgen {

extern Settings settings;

} // kernelgen

#endif // KERNELGEN_H
