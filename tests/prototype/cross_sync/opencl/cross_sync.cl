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

__kernel void gpu_kernel(
	__global float* data, size_t size, int npasses,
	__global int* lock, __global int* finish,
	__global int* pmaxidx, __global float* pmaxval)
{
	*finish = 0;

	for (int ipass = 0; ipass < npasses; ipass++)
	{
		// Run some time-consuming work.
		for (int i = 1; i < size; i++)
			data[i] = data[i - 1];
		data[0] = data[size - 1];
	
		int maxidx = 0;
		float maxval = data[0];
		for (int i = 1; i < size; i++)
			if (data[i] >= maxval)
			{
				maxval = data[i];
				maxidx = i;
			}
		*pmaxidx = maxidx;
		*pmaxval = maxval;

		// Thread runs when lock = 0 and gets blocked
		// on lock = 1.
		
		// Lock thread.
		atomic_cmpxchg(lock, 0, 1);

		// Wait for unlock.
		while (atomic_cmpxchg(lock, 0, 0)) continue;
	}

	// Lock thread.
	atomic_cmpxchg(lock, 0, 1);

	*finish = 1;
}

__kernel void gpu_monitor(__global int* lock)
{
	// Unlock blocked gpu kernel associated
	// with lock. It simply waits for lock
	// to be dropped to zero.
	atomic_cmpxchg(lock, 1, 0);

	// Wait for lock to be set.
	// When lock is set this thread exits,
	// and CPU monitor thread gets notified
	// by synchronization.
	while (!atomic_cmpxchg(lock, 1, 1)) continue;
}

