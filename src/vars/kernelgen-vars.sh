##===- kernelgen-vars.sh - KernelGen environment variables setup ----------===//
##
##     KernelGen -- A prototype of LLVM-based auto-parallelizing Fortran/C
##        compiler for NVIDIA GPUs, targeting numerical modeling code.
##
## This file is distributed under the University of Illinois Open Source
## License. See LICENSE.TXT for details.
##
##===----------------------------------------------------------------------===//
##
## This file implements KernelGen enviromnet variables setup.
##
##===----------------------------------------------------------------------===//

bindir=CMAKE_INSTALL_PREFIX/bin
[ -z "$PATH" ] || PATH=":${PATH}"
  PATH="${bindir}$PATH"
export PATH
