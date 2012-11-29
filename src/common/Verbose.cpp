/*
 * KernelGen - the LLVM-based compiler with GPU kernels generation over C backend.
 *
 * Copyright (c) 2011 Dmitry Mikushin
 *
 * This software is provided 'as-is', without any express or implied warranty.
 * In no event will the authors be held liable for any damages arising
 * from the use of this software.
 * Permission is granted to anyone to use this software for any purpose,
 * including commercial applications, and to alter it and redistribute it freely,
 * subject to the following restrictions:
 *
 * 1. The origin of this software must not be misrepresented;
 * you must not claim that you wrote the original software.
 * If you use this software in a product, an acknowledgment
 * in the product documentation would be appreciated but is not required.
 * 2. Altered source versions must be plainly marked as such,
 * and must not be misrepresented as being the original software.
 * 3. This notice may not be removed or altered from any source distribution.
 */

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
