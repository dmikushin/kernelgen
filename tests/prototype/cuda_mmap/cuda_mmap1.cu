#include <stdio.h>
#include <sys/mman.h>

__global__ void kernel(int* array)
{
	array[10] = 1313;
}

int main(int argc, char* argv[])
{
	size_t length = 103 * sizeof(int);
	int* array = NULL;
	cudaError_t cuerr = cudaMalloc((void**)&array, length);
	kernel<<<1, 1>>>(array);
	cuerr = cudaDeviceSynchronize();

	void* map = mmap(array, length,
		PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS,
		-1, 0);
	if (map == (void*)-1)
	{
		fprintf(stderr, "Cannot map memory onto %p + %zu\n",
			array, length);
		return -1;
	}

	printf("Mapped memory %p + %zu onto %p + %zu\n",
		map, length, array, length);
	printf("Before assignment: array[10] = %d\n", array[10]);

	cuerr = cudaMemcpy(array, array, length, cudaMemcpyDeviceToHost);
	if (cuerr != cudaSuccess)
	{
		fprintf(stderr, "Cannot copy GPU data to mapped memory: %s\n",
			cudaGetErrorString(cuerr));
		return 1;
	}

	printf("After assignment: array[10] = %d\n", array[10]);

	int err = munmap(map, length);
	if (err == -1)
	{
		fprintf(stderr, "Cannot unmap memory from %p + %zu\n",
			array, length);
		return -1;
	}

	return 0;
}

