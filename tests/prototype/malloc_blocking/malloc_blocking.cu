__device__ int spinval;
__device__ void spin(int count)
{
	volatile int *spinptr = &spinval;
	while(count--)
	{
		*spinptr = count;

		// ~500 cycles timeout on Fermi
		__threadfence();
	}
}

__global__ void kernel1(int* lock)
{
	// Wait for unlock.
	while (atomicCAS(lock, 0, 0))
		spin(10);
}

__global__ void kernel2(int* lock)
{
	// Unlock.
	atomicCAS(lock, 1, 0);
}

__global__ void kernel3(int* lock)
{
	// Device-malloc call.
	int** buffer = (int**)malloc(10);
	buffer[1] = lock;

	// Unlock.
	atomicCAS(lock, 1, 0);
}

#include <assert.h>
#include <cuda.h>
#include <stdio.h>

static void usage(const char* filename)
{
	printf("Usage: %s <mode>\n", filename);
	printf("mode = 0: launch kernel1 and kernel2 without device-malloc (will succeed)\n");
	printf("mode = 1: launch kernel1 and kernel3 with device-malloc (will hang)\n");
}

int main (int argc, char* argv[])
{
	if (argc != 2)
	{
		usage(argv[0]);
		return 0;
	}

	int mode = atoi(argv[1]);
	if ((mode < 0) || (mode > 1))
	{
		usage(argv[0]);
		return 0;
	}

	// Initialize lock.
	int* lock = NULL;
	cudaError_t cuerr = cudaMalloc((void**)&lock, sizeof(int));
	assert(cuerr == cudaSuccess);
	int one = 1;
	cuerr = cudaMemcpy(lock, &one, sizeof(int), cudaMemcpyHostToDevice);
	assert(cuerr == cudaSuccess);

	// Create streams.
	cudaStream_t stream1, stream2;
	cuerr = cudaStreamCreate(&stream1);
	assert(cuerr == cudaSuccess);
	cuerr = cudaStreamCreate(&stream2);
	assert(cuerr == cudaSuccess);

	if (mode == 0)
	{
		// Launch first kernel.
		kernel1<<<1, 1, 0, stream1>>>(lock);
		cuerr = cudaGetLastError();
		assert(cuerr == cudaSuccess);

		// Launch second kernel.
		kernel2<<<1, 1, 0, stream2>>>(lock);
		cuerr = cudaGetLastError();
		assert(cuerr == cudaSuccess);
	}
	if (mode == 1)
	{
		// Launch first kernel.
		kernel1<<<1, 1, 0, stream1>>>(lock);
		cuerr = cudaGetLastError();
		assert(cuerr == cudaSuccess);

		// Launch third kernel (with malloc).
		kernel3<<<1, 1, 0, stream2>>>(lock);
		cuerr = cudaGetLastError();
		assert(cuerr == cudaSuccess);
	}

	cuerr = cudaDeviceSynchronize();
	assert(cuerr == cudaSuccess);

	cuerr = cudaStreamDestroy(stream1);
	assert(cuerr == cudaSuccess);
	cuerr = cudaStreamDestroy(stream2);
	assert(cuerr == cudaSuccess);

	cuerr = cudaFree(lock);
	assert(cuerr == cudaSuccess);

	return 0;
}
