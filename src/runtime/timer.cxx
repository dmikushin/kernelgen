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

#include "util.h"

using namespace kernelgen::runtime;

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

