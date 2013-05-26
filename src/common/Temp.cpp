//===- Temp.cpp - API for manipulating temporary files -------------------===//
//
//     KernelGen -- A prototype of LLVM-based auto-parallelizing Fortran/C
//        compiler for NVIDIA GPUs, targeting numerical modeling code.
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements KernelGen API for manipulating temporary files.
//
//===----------------------------------------------------------------------===//

#include "Temp.h"

#include "llvm/Object/Archive.h"
#include "llvm/Support/FileSystem.h"

#include <iostream>
#include <unistd.h>

#include "KernelGen.h"

using namespace kernelgen::utils;
using namespace llvm;
using namespace llvm::object;
using namespace llvm::sys;
using namespace llvm::sys::fs;
using namespace std;

TempFile Temp::getFile(string mask, bool closefd) {
  // Open unique file for the given mask.
  int fd;
  SmallString<128> filename_vector;
  if (error_code err = unique_file(mask, fd, filename_vector))
    THROW("Cannot open unique temp file " << err, err.value());

  // Store filename.
  string filename = (StringRef) filename_vector;

  if (closefd)
    close(fd);

  // Create output file tracker.
  string err;
  tool_output_file file(filename.c_str(), err, raw_fd_ostream::F_Binary);
  if (!err.empty())
    THROW("Cannot create output file tracker " << err, filename);

  return TempFile(filename, fd, file);
}
