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

#include <stdio.h>
#include <time.h>

#define CLOCKID CLOCK_REALTIME
//#define CLOCKID CLOCK_MONOTONIC
//#define CLOCKID CLOCK_PROCESS_CPUTIME_ID
//#define CLOCKID CLOCK_THREAD_CPUTIME_ID

// Get the built-in timer resolution.
void util_get_timer_resolution(util_time_t* val)
{
	if ((sizeof(int64_t) == sizeof(time_t)) &&
		(sizeof(int64_t) == sizeof(long)))
		clock_getres(CLOCKID, (struct timespec *)val);
	else
	{
		struct timespec t;
		clock_getres(CLOCKID, &t);
		val->seconds = t.tv_sec;
		val->nanoseconds = t.tv_nsec;
	}
}

// Get the built-in timer value.
void util_get_time(util_time_t* val)
{
	if ((sizeof(int64_t) == sizeof(time_t)) &&
		(sizeof(int64_t) == sizeof(long)))
		clock_gettime(CLOCKID, (struct timespec *)val);
	else
	{
		struct timespec t;
		val->seconds = 0;
		val->nanoseconds = 0;
		clock_gettime(CLOCKID, &t);
		val->seconds = t.tv_sec;
		val->nanoseconds = t.tv_nsec;
	}
}

// Get the built-in timer measured values difference.
double util_get_time_diff(
	util_time_t* val1, util_time_t* val2)
{
	int64_t seconds = val2->seconds - val1->seconds;
	int64_t nanoseconds = val2->nanoseconds - val1->nanoseconds;
	
	if (val2->nanoseconds < val1->nanoseconds)
	{
		seconds--;
		nanoseconds = (1000000000 - val1->nanoseconds) + val2->nanoseconds;
	}
	
	return (double)0.000000001 * nanoseconds + seconds;
}

// Print the built-in timer measured values difference.
void util_print_time_diff(
	util_time_t* val1, util_time_t* val2)
{
	int64_t seconds = val2->seconds - val1->seconds;
	int64_t nanoseconds = val2->nanoseconds - val1->nanoseconds;
	
	if (val2->nanoseconds < val1->nanoseconds)
	{
		seconds--;
		nanoseconds = (1000000000 - val1->nanoseconds) + val2->nanoseconds;
	}
	if (sizeof(uint64_t) == sizeof(long))
		printf("%ld.%09ld", (long)seconds, (long)nanoseconds);
	else
		printf("%lld.%09lld", seconds, nanoseconds);

}

