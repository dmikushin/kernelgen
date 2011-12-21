#include <cuda.h>
#include <list>
#include <signal.h>
#include <stdio.h>
#include <sys/mman.h>
#include <sys/types.h>
#include <unistd.h>

std::list<void*> maps;

// signal handler
void sighandler(int code, siginfo_t *siginfo, void* ucontext)
{
	// Check if address is valid on GPU.
	void* addr = siginfo->si_addr;

	void* base;
	size_t size;
	CUresult cuerr = cuMemGetAddressRange((CUdeviceptr*)&base, &size, (CUdeviceptr)addr);
	if (cuerr == CUDA_SUCCESS)
	{
		void* map = mmap(base, size,
			PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS,
			-1, 0);
		if (map == (void*)-1)
		{
			fprintf(stderr, "Cannot map memory onto %p + %zu\n",
				base, size);
			return;
		}
		maps.push_back(map);

		printf("Mapped memory %p + %zu onto %p + %zu\n",
			map, size, base, size);

		cudaError_t cuerr = cudaMemcpy(base, base, size, cudaMemcpyDeviceToHost);
		if (cuerr != cudaSuccess)
		{
			fprintf(stderr, "Cannot copy GPU data to mapped memory: %s\n",
				cudaGetErrorString(cuerr));
			return;
		}
	}
}

__global__ void kernel(int* array)
{
	array[10] = 1313;
}

int main(int argc, char* argv[])
{
	// Set up signal handler.
	struct sigaction sa;
	sa.sa_flags = SA_SIGINFO;
	sigfillset(&sa.sa_mask);
	sa.sa_sigaction = sighandler;
	sigaction(SIGSEGV, &sa, 0);

	// Create GPU array.
	size_t length = 103 * sizeof(int);
	int* array = NULL;
	cudaError_t cuerr = cudaMalloc((void**)&array, length);
	if (cuerr != cudaSuccess)
	{
		fprintf(stderr, "Cannot create device memory array: %s\n",
			cudaGetErrorString(cuerr));
		return -1;
	}

	// Launch GPU kernel, assigning value to GPU array.
	kernel<<<1, 1>>>(array);
	cuerr = cudaDeviceSynchronize();
	if (cuerr != cudaSuccess)
	{
		fprintf(stderr, "Cannot synchronize GPU kernel: %s\n",
			cudaGetErrorString(cuerr));
		return -1;
	}

	printf("Trying to read: array[10] = %d\n", array[10]);

	for (std::list<void*>::iterator i = maps.begin(), e = maps.end(); i != e; i++)
	{
		void* map = *i;
		int err = munmap(map, length);
		if (err == -1)
		{
			fprintf(stderr, "Cannot unmap memory from %p + %zu\n",
				array, length);
			return -1;
		}
	}

	return 0;
}

