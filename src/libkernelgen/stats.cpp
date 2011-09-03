/*
 * KGen - the LLVM-based compiler with GPU kernels generation over C backend.
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

#include "stats.h"

#include <assert.h>

// Stats output file descriptor.
FILE* kernelgen_stats_output_file = NULL;

long kernelgen_stats_verbose = 1 << 3;

// Record start measured execution marker.
extern "C" void kernelgen_record_time_start(struct kernelgen_launch_stats_t* stats)
{
	assert((stats->started == 0) && "Timer was already started!");
	stats->started = 1;
	kernelgen_get_time(&stats->start);
}

// Record finish measured execution marker
// and time result.
extern "C" void kernelgen_record_time_finish(struct kernelgen_launch_stats_t* stats)
{
	assert((stats->started == 1) && "Timer was not started!");
	kernelgen_time_t finish;
	kernelgen_get_time(&finish);

	double time = kernelgen_get_time_diff(&stats->start, &finish);
	stats->time.push_back(time);
	stats->started = 0;
}

extern "C" int kernelgen_discard(struct kernelgen_launch_config_t* l,
	struct kernelgen_launch_stats_t* host, struct kernelgen_launch_stats_t* device)
{
	// Discard kernel, if there were 10 invocations,
	// slower than host version.
	if (host->time.size() == 100)
	{	
		double avg_host = 0.0, avg_device = 0.0;
		
		for (std::list<double>::iterator it = host->time.begin();
			it != host->time.end(); it++)
			avg_host += *it;
		for (std::list<double>::iterator it = device->time.begin();
			it != device->time.end(); it++)
			avg_device += *it;
		
		if (avg_host <= avg_device)
		{
			kernelgen_print_stats(KERNELGEN_STATS_SLOWER,
				"-\t%s\t%f\t%f\n", l->kernel_name, avg_host, avg_device);
			return 0;
		}
		
		host->time.clear();
		device->time.clear();

		kernelgen_print_debug(kernelgen_stats_verbose,
			"%s host time = %f sec, device time = %f sec\n",
			l->kernel_name, avg_host, avg_device);
		
		kernelgen_print_stats(KERNELGEN_STATS_FASTER,
			"+\t%s\t%f\t%f\n", l->kernel_name, avg_host, avg_device);
	}
	
	return 0;
}
