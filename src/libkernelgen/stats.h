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

#ifndef STATS_H

#include "kernelgen_int.h"

#ifdef __cplusplus

#include <list>

// Defines kernel launching statistics.
struct kernelgen_launch_stats_t
{
	kernelgen_time_t start;
	std::list<double> time;
};

extern "C"
{
#endif

// Record start measured execution marker.
void kernelgen_record_time_start(struct kernelgen_launch_stats_t* stats);

// Record finish measured execution marker
// and time result.
void kernelgen_record_time_finish(struct kernelgen_launch_stats_t* stats);

int kernelgen_discard(struct kernelgen_launch_config_t* l,
	struct kernelgen_launch_stats_t* host, struct kernelgen_launch_stats_t* device);

#ifdef __cplusplus
}
#endif

#endif // STATS_H
