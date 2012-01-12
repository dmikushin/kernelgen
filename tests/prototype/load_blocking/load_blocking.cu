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
	while (atomicCAS(lock, 0, 0)) spin(10);
}

/*extern "C" __global__ void kernel2(int* lock)
{
	// Unlock.
	atomicCAS(lock, 1, 0);
}*/

#include <assert.h>
#include <cuda.h>
#include <stdio.h>

static void usage(const char* filename)
{
	printf("Usage: %s <mode>\n", filename);
	printf("mode = 0: launch kernel1 before kernel2 load (will hang)\n");
	printf("mode = 1: launch kernel1 after kernel2 load (will succeed)\n");
}

int main (int argc, char* argv[])
{
	if (argc != 2)
	{
		usage(argv[0]);
		return 0;
	}

	int mode = atoi(argv[1]);
	if ((mode < 0) || (mode > 2))
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
	}

	// Load second kernel.
	CUmodule module;
	CUresult err = cuModuleLoad(&module, "kernel2.ptx");
	assert(err == CUDA_SUCCESS);
	CUfunction kernel2;
	err = cuModuleGetFunction(&kernel2, module, "kernel2");

	if (mode == 1)
	{
		// Launch first kernel.
		kernel1<<<1, 1, 0, stream1>>>(lock);
		cuerr = cudaGetLastError();
		assert(cuerr == cudaSuccess);
	}

	struct { unsigned int x, y, z; } gridDim, blockDim;
	gridDim.x = 1; gridDim.y = 1; gridDim.z = 1;
	blockDim.x = 1; blockDim.y = 1; blockDim.z = 1;
	size_t szshmem = 0;
	void* kernel2_args[] = { (void*)&lock };
	err = cuLaunchKernel(kernel2,
		gridDim.x, gridDim.y, gridDim.z,
		blockDim.x, blockDim.y, blockDim.z, szshmem,
		stream2, kernel2_args, NULL);
	assert(err == CUDA_SUCCESS);

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
