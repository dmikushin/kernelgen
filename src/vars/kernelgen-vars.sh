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

export PATH=$PATH:BOOTSTRAP_ROOT/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:BOOTSTRAP_ROOT/lib:BOOTSTRAP_ROOT/lib32:BOOTSTRAP_ROOT/lib64
