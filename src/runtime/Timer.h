//===- Timer.h - KernelGen time measurement API ---------------------------===//
//
//     KernelGen -- A prototype of LLVM-based auto-parallelizing Fortran/C
//        compiler for NVIDIA GPUs, targeting numerical modeling code.
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements time measurement API.
//
//===----------------------------------------------------------------------===//

#include <time.h>

namespace kernelgen { namespace runtime {

class timer
{
        timespec time_start, time_stop;
        bool started;

public :

        static timespec get_resolution();

        timer(bool start = true);

        timespec start();
        timespec stop();

        double get_elapsed(timespec* start = NULL);
};

} }

