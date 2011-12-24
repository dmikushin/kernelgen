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

#ifndef KERNELGEN_INTEROP_H
#define KERNELGEN_INTEROP_H

/*
 * The following list of definitions are
 * possible execution states with respect to the
 * main device kernel. Once main kernel is launched
 * and is working, its state becomes ACTIVE. During
 * execution main kernel may request to launch another
 * kernel (LOOPCALL) or perform a host call (HOSTCALL).
 * While other kernel or host call is running, the
 * main kernel is suspended, and will continue
 * immidiately after the request finishes.
 * Main kernel runs to the end, and before returning
 * sets state to INACTIVE.
 */

// Main kernel is inactive (finished execution).
#define KERNELGEN_STATE_INACTIVE	0

// Main kernel is running.
#define KERNELGEN_STATE_ACTIVE		1

// Main kernel requested to execute another kernel.
#define KERNELGEN_STATE_LOOPCALL	2

// Main kernel requested
#define KERNELGEN_STATE_HOSTCALL	3

// Defines callback status structure.
struct kernelgen_callback_t
{
	int lock;
	int state;
	unsigned char* name;
	int szarg;
	int* arg;
};

#include <string.h>

// Defines the dynamic memory pool configuration.
typedef struct
{
	// The pointer to the memory pool (a single
	// large array, allocated with cudaMalloc).
	char* pool;
	
	// The size of the memory pool.
	size_t szpool;
	
	// The size of the used memory.
	size_t szused;

	// The number of MCB records in pool.
	int count;
}
kernelgen_memory_t;

#endif // KERNELGEN_INTEROP_H

