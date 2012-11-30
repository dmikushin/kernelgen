//===- Timer.cpp - KernelGen time measurement API -------------------------===//
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

#include "Util.h"

using namespace kernelgen::runtime;

#include <stdint.h>
#include <time.h>

#define CLOCKID CLOCK_REALTIME
//#define CLOCKID CLOCK_MONOTONIC
//#define CLOCKID CLOCK_PROCESS_CPUTIME_ID
//#define CLOCKID CLOCK_THREAD_CPUTIME_ID

static double get_diff(timespec& start, timespec& stop)
{
	int64_t seconds = stop.tv_sec - start.tv_sec;
	int64_t nanoseconds = stop.tv_nsec - start.tv_nsec;

	if (stop.tv_nsec < start.tv_nsec)
	{
		seconds--;
		nanoseconds = (1000000000 - start.tv_nsec) + stop.tv_nsec;
	}

	return (double)0.000000001 * nanoseconds + seconds;
}

timer::timer(bool start) : started(false)
{
	if (start)
	{
		clock_gettime(CLOCKID, &time_start);
		started = true;
	}
}

timespec get_resoltion()
{
	timespec val;
	clock_getres(CLOCKID, &val);
}

timespec timer::start()
{
	clock_gettime(CLOCKID, &time_start);
	started = true;
	return time_start;
}

timespec timer::stop()
{
	clock_gettime(CLOCKID, &time_stop);
	started = false;
	return time_stop;
}

double timer::get_elapsed(timespec* start)
{
	if (started) stop();
	if (start) return get_diff(*start, time_stop);
	return get_diff(time_start, time_stop);	
}

