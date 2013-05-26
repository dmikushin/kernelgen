//===- Memory.h - Device memory pool for NVIDIA GPUs ----------------------===//
//
//     KernelGen -- A prototype of LLVM-based auto-parallelizing Fortran/C
//        compiler for NVIDIA GPUs, targeting numerical modeling code.
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains functions to support the dynamic memory heap in GPU
// global memory. The rationale is to replace builtin malloc/free calls, since
// they are incompatible with concurrent kernels execution.
//
//===----------------------------------------------------------------------===//

#define kernelgen_free_device(...)                                             \
  kernelgen_free_device(__VA_ARGS__, unsigned int *__kernelgen_memory)
#include "free.h"
#undef kernelgen_free_device

struct free_callback_t {
  void *ptr;
};

extern "C" __attribute__((global)) void
kernelgen_free(int *callback, unsigned int *__kernelgen_memory) {
  free_callback_t *cb = (free_callback_t *)callback;
  kernelgen_free_device(cb->ptr, __kernelgen_memory);
}
