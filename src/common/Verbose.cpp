//===- Verbose.cpp - KernelGen verbose output API -------------------------===//
//
//     KernelGen -- A prototype of LLVM-based auto-parallelizing Fortran/C
//        compiler for NVIDIA GPUs, targeting numerical modeling code.
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements KernelGen verbose output API.
//
//===----------------------------------------------------------------------===//

#include "Verbose.h"

namespace kernelgen {

Verbose::Color Verbose::Black      ( 0 );
Verbose::Color Verbose::Red        ( 1 );
Verbose::Color Verbose::Green      ( 2 );
Verbose::Color Verbose::Yellow     ( 3 );
Verbose::Color Verbose::Blue       ( 4 );
Verbose::Color Verbose::Magenta    ( 5 );
Verbose::Color Verbose::Cyan       ( 6 );
Verbose::Color Verbose::White      ( 7 );
Verbose::Color Verbose::SavedColor ( 8 );
Verbose::Color Verbose::Reset      ( 9 );

Verbose::Mode Verbose::Always      (-2 );
Verbose::Mode Verbose::Default     (-1 );
Verbose::Mode Verbose::Disable     ( 0 );
Verbose::Mode Verbose::Summary     ( 1 << 0 );
Verbose::Mode Verbose::Sources     ( 1 << 1 );
Verbose::Mode Verbose::ISA         ( 1 << 2 );
Verbose::Mode Verbose::DataIO      ( 1 << 3 );
Verbose::Mode Verbose::Hostcall    ( 1 << 4 );
Verbose::Mode Verbose::Polly       ( 1 << 5 );
Verbose::Mode Verbose::Perf        ( 1 << 6 );
Verbose::Mode Verbose::Alloca      ( 1 << 7 );

Verbose::Action Verbose::Flush     ( 0 );

} // kernelgen
