#include "kernelgen_memory.h"

#include <stdio.h>

// Setup the device global memory pool initial configuration.
void kernelgen_memory_init(size_t szpool)
{
	// First, fill config on host.
	kernelgen_memory_t config_host;

	// Allocate pool and flush it to zero.
	cudaError_t cuerr = cudaMalloc((void**)&config_host.pool, szpool);
	if (cuerr != cudaSuccess)
	{
		fprintf(stderr, "Cannot allocate device memory pool: %s\n",
			cudaGetErrorString(cuerr));
		return;
	}
	cuerr = cudaMemset(config_host.pool, 0, szpool);
	if (cuerr != cudaSuccess)
	{
		fprintf(stderr, "Cannot initialize device memory pool: %s\n",
			cudaGetErrorString(cuerr));
		return;
	}

	config_host.szused = 0;
	config_host.szpool = szpool;
	config_host.count = 0;

	// Copy the resulting config to the special
	// device variable.
	kernelgen_memory_t* config_device;
	cuerr = cudaGetSymbolAddress((void**)&config_device, "kernelgen_memory");
	if (cuerr != cudaSuccess)
	{
		fprintf(stderr, "Cannot find kernelgen_memory on device: %s\n",
			cudaGetErrorString(cuerr));
		return;
	}
	cuerr = cudaMemcpy(config_device, &config_host,
		sizeof(kernelgen_memory_t), cudaMemcpyHostToDevice);
	if (cuerr != cudaSuccess)
	{
		fprintf(stderr, "Cannot copy device memory pool configuration: %s\n",
			cudaGetErrorString(cuerr));
		return;
	}
}

__global__ void kernel()
{
	char* A = (char*)kernelgen_malloc(1);
	A[0] = 'A';
	char* B = (char*)kernelgen_malloc(2);
	B[1] = 'B';
	char* C = (char*)kernelgen_malloc(3);
	C[1] = 'C';
	kernelgen_free(B);
	char* D = (char*)kernelgen_malloc(1);
	D[0] = 'D';
}

int main(int argc, char* argv[])
{
	size_t szpool = 80;
	kernelgen_memory_init(szpool);

	kernel<<<1,1>>>();
	cudaError_t cuerr = cudaDeviceSynchronize();
	if (cuerr != cudaSuccess)
	{
		fprintf(stderr, "Cannot synchronize test kernel: %s\n",
			cudaGetErrorString(cuerr));
		return -1;
	}

	// Check the pool contents.
	kernelgen_memory_t config_host, *config_device;
	cuerr = cudaGetSymbolAddress((void**)&config_device, "kernelgen_memory");
	if (cuerr != cudaSuccess)
	{
		fprintf(stderr, "Cannot find kernelgen_memory on device: %s\n",
			cudaGetErrorString(cuerr));
		return -1;
	}
	cuerr = cudaMemcpy(&config_host, config_device,
		sizeof(kernelgen_memory_t), cudaMemcpyDeviceToHost);
	if (cuerr != cudaSuccess)
	{
		fprintf(stderr, "Cannot copy back device memory pool configuration: %s\n",
			cudaGetErrorString(cuerr));
		return -1;
	}
	char* pool = (char*)malloc(szpool);
	cuerr = cudaMemcpy(pool, config_host.pool, szpool, cudaMemcpyDeviceToHost);
	for (int i = 0; i < szpool; i++)
		if ((pool[i] >= 'A') && (pool[i] <= 'Z'))
			printf("%c", pool[i]);
		else
			printf(" ");
	printf("\n");
	for (int i = 0; i < szpool; i++)
		printf("%d", i % 8);
	printf("\n");

	return 0;
}

