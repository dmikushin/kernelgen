//===- Util.h - Old KernelGen utilities (deprecated) ----------------------===//
//
//     KernelGen -- A prototype of LLVM-based auto-parallelizing Fortran/C
//        compiler for NVIDIA GPUs, targeting numerical modeling code.
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef KERNELGEN_UTIL_H
#define KERNELGEN_UTIL_H

#ifdef __cplusplus

#include <iostream>

#include <list>
#include <string>

// Execute the specified command in the system shell, supplying
// input stream content and returning results from output and
// error streams.
int execute(std::string command, std::list<std::string> args,
            std::string in = "", std::string *out = NULL,
            std::string *err = NULL);

#endif

#endif // KERNELGEN_UTIL_H
