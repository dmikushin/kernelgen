//===- kernelgen_interop.h - KernelGen host-device interoperation layer ---===//
//
//     KernelGen -- A prototype of LLVM-based auto-parallelizing Fortran/C
//        compiler for NVIDIA GPUs, targeting numerical modeling code.
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines a plugin interface for external optimization tools,
// willing to insert pre- and postprocessing handlers into the compilation
// steps.
//
//===----------------------------------------------------------------------===//

typedef void (*kernelgen_after_ptx_t)(std::string &ptx,
                                      const std::string &kernel_name);

typedef void (*kernelgen_after_cubin_t)(std::string &cubin,
                                        const std::string &kernel_name);
