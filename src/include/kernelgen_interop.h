//===- kernelgen_interop.h - KernelGen host-device interoperation layer ---===//
//
//     KernelGen -- A prototype of LLVM-based auto-parallelizing Fortran/C
//        compiler for NVIDIA GPUs, targeting numerical modeling code.
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements target code generation for supported architectures.
//
//===----------------------------------------------------------------------===//

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
#define KERNELGEN_STATE_INACTIVE       -1

// Main kernel state is unknown/unset (this value
// is unused for security reasons).
#define KERNELGEN_STATE_UNKNOWN         0

// Main kernel is running.
#define KERNELGEN_STATE_ACTIVE          1

// Main kernel requested to execute another kernel.
#define KERNELGEN_STATE_LOOPCALL        2

// Main kernel requested
#define KERNELGEN_STATE_HOSTCALL        3

// Loop kernel not launched, main kernel should use
// the fallback branch.
#define KERNELGEN_STATE_FALLBACK        4

#ifdef __cplusplus
namespace kernelgen
{
	struct Kernel;
}
#else
struct Kernel;
#endif

struct CallbackData;

// Defines callback status structure.
#pragma pack(push, 1)
struct kernelgen_callback_t
{
	// The synchronization lock shared between
	// the "main" and "monitor" kernels for atomic
	// read/write.
	int lock;
	
	// The callback state (see defines above).
	int state;
	
	// The callback kernel.
#ifdef __cplusplus
	kernelgen::Kernel* kernel;
#else
	struct Kernel* kernel;
#endif
	
	// The size of callback data (see
	// kernelgen_callback_data_t).
	int szdata;

	// The size of integer-only function arguments
	// in callback data.
	int szdatai;
	
	// The callback data.
	struct CallbackData* data;
};
#pragma pack(pop)

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
	size_t count;
}
kernelgen_memory_t;

#endif // KERNELGEN_INTEROP_H

