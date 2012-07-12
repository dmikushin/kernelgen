#include "kernelgen_runtime.h"

__attribute__((device)) __attribute__((always_inline)) int __iAtomicCAS(
        volatile int *p, int compare, int val)
{
	int ret;
	asm volatile (
		"atom.cas.b32    %0, [%1], %2, %3; \n\t"
		: "=r"(ret) : "l"(p), "r"(compare), "r"(val)
	);
	return ret;
}

extern "C" __attribute__((global)) void kernelgen_monitor(int* callback)
{
	// Unlock blocked gpu kernel associated with lock.
	// It simply waits for lock to be dropped to zero.
	__iAtomicCAS(&((struct kernelgen_callback_t*)callback)->lock, 1, 0);

	// Wait for lock to be set. When lock is set this thread exits,
	// and CPU monitor thread gets notified by synchronization.
	while (!__iAtomicCAS(&((struct kernelgen_callback_t*)callback)->lock, 1, 1))
		continue;
}

