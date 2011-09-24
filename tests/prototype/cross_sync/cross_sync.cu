#include <cuda_runtime.h>
#include <malloc.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

__global__ void gpu_kernel(float* data, size_t size, int npasses,
	int* lock, int* finish, int* pmaxidx, float* pmaxval)
{
	printf("Test 3\n");

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
		
		printf("Test 1\n");
	
		// Lock thread.
		atomicCAS(lock, 0, 1);

		printf("Test 2\n");

		// Wait for unlock.
		while (atomicCAS(lock, 0, 0)) continue;
	}

	// Notify the target is finishing.	
	*finish = 1;

	// Lock thread.
	atomicCAS(lock, 0, 1);
}

__global__ void gpu_monitor(int* lock)
{
	// Unlock blocked gpu kernel associated
	// with lock. It simply waits for lock
	// to be dropped to zero.
	atomicCAS(lock, 1, 0);

	printf("Hello!\n");

	// Wait for lock to be set.
	// When lock is set this thread exits,
	// and CPU monitor thread gets notified
	// by synchronization.
	while (!atomicCAS(lock, 1, 1)) continue;
}

typedef struct
{
	cudaStream_t stream;
	pthread_barrier_t barrier;
	int* maxidx;
	float* maxval;
	int* lock;
	int* finish;
}
monitor_params_t;

void* cpu_monitor(void* arg)
{
	int first = 1;
	monitor_params_t* params = (monitor_params_t*)arg;

	while (1)
	{
		// Launch GPU monitoring kernel.
		gpu_monitor<<<1, 1, 1, params->stream>>>(params->lock);
		cudaError_t custat = cudaGetLastError();
		if (custat != cudaSuccess)
		{
			fprintf(stderr, "Cannot launch target GPU kernel: %s\n",
				cudaGetErrorString(custat));
			pthread_exit((void*)1);
			return NULL;
		}

		// Synchronize main thread and monitor thread at barrier.
		if (first)
		{
			pthread_barrier_wait(&params->barrier);
			pthread_barrier_wait(&params->barrier);
			first = 0;
		}

		// Wait for GPU monitoring kernel to finish.
		custat = cudaStreamSynchronize(params->stream);
		if (custat != cudaSuccess)
		{
			fprintf(stderr, "Cannot synchronize GPU kernel: %s\n",
				cudaGetErrorString(custat));
			pthread_exit((void*)1);
			return NULL;
		}

		// Do something with GPU data.
		int maxidx = 0;
		custat = cudaMemcpy(&maxidx, params->maxidx, sizeof(int),
			cudaMemcpyDeviceToHost);
		if (custat != cudaSuccess)
		{
			fprintf(stderr, "Cannot get GPU maxidx value: %s\n",
				cudaGetErrorString(custat));
			pthread_exit((void*)1);
			return NULL;
		}
		float maxval = 0.0;
		custat = cudaMemcpy(&maxval, params->maxval, sizeof(float),
			cudaMemcpyDeviceToHost);
		if (custat != cudaSuccess)
		{
			fprintf(stderr, "Cannot get GPU maxval value: %s\n",
				cudaGetErrorString(custat));
			pthread_exit((void*)1);
			return NULL;
		}
		printf("max value = %f @ index = %d\n", maxval, maxidx);		
		
		// Check if target GPU kernel has finished.
		int finish = 0;
		custat = cudaMemcpy(&finish, params->finish, sizeof(int),
			cudaMemcpyDeviceToHost);
		if (custat != cudaSuccess)
		{
			fprintf(stderr, "Cannot get GPU finish value: %s\n",
				cudaGetErrorString(custat));
			pthread_exit((void*)1);
			return NULL;
		}
		if (finish) break;
	}	

	return NULL;
}

int main(int argc, char* argv[])
{
	if (argc != 3)
	{
		printf("%s <size> <npasses>\n", argv[0]);
		return 0;
	}

	int count = 0;
	cudaError_t custat = cudaGetDeviceCount(&count);
	if (custat != cudaSuccess)
	{
		fprintf(stderr, "Cannot get CUDA device count: %s\n",
			cudaGetErrorString(custat));
		return 1;
	}
	if (!count)
	{
		fprintf(stderr, "No CUDA devices found\n");
		return 1;
	}

	monitor_params_t params;

	// Create stream where monitoring kernel will be
	// executed.
	custat = cudaStreamCreate(&params.stream);
	if (custat != cudaSuccess)
	{
		fprintf(stderr, "Cannot create monitoring stream: %s\n",
			cudaGetErrorString(custat));
		return 1;
	}
	cudaStream_t stream;
	custat = cudaStreamCreate(&stream);
	if (custat != cudaSuccess)
	{
		fprintf(stderr, "Cannot create monitoring stream: %s\n",
			cudaGetErrorString(custat));
		return 1;
	}

	size_t size = atoi(argv[1]);
	int npasses = atoi(argv[2]);

	float* cpu_data = (float*)malloc(sizeof(float) * size);
	double dinvrandmax = (double)1.0 / RAND_MAX;
	for (int i = 0; i < size; i++)
		cpu_data[i] = rand() * dinvrandmax;

	float* gpu_data = NULL;
	custat = cudaMalloc((void**)&gpu_data, sizeof(float) * size);
	if (custat != cudaSuccess)
	{
		fprintf(stderr, "Cannot create GPU data buffer: %s\n",
			cudaGetErrorString(custat));
		return 1;
	}
	custat = cudaMemcpy(gpu_data, cpu_data, sizeof(float) * size,
		cudaMemcpyHostToDevice);
	if (custat != cudaSuccess)
	{
		fprintf(stderr, "Cannot fill GPU data buffer: %s\n",
			cudaGetErrorString(custat));
		return 1;
	}
	free(cpu_data);
	
	custat = cudaMalloc((void**)&params.maxidx, sizeof(int));
	if (custat != cudaSuccess)
	{
		fprintf(stderr, "Cannot create GPU maxidx buffer: %s\n",
			cudaGetErrorString(custat));
		return 1;
	}
	
	custat = cudaMalloc((void**)&params.maxval, sizeof(float));
	if (custat != cudaSuccess)
	{
		fprintf(stderr, "Cannot create GPU maxval buffer: %s\n",
			cudaGetErrorString(custat));
		return 1;
	}
	
	custat = cudaMalloc((void**)&params.finish, sizeof(int));
	if (custat != cudaSuccess)
	{
		fprintf(stderr, "Cannot create GPU finish buffer: %s\n",
			cudaGetErrorString(custat));
		return 1;
	}
	
	// Initialize thread locker variable.
	// Initial state is "locked". It will be dropped
	// by gpu side monitor that must be started *before*
	// target GPU kernel.
	custat = cudaMalloc((void**)&params.lock, sizeof(int));
	if (custat != cudaSuccess)
	{
		fprintf(stderr, "Cannot create GPU lock buffer: %s\n",
			cudaGetErrorString(custat));
		return 1;
	}
	int one = 1;
	custat = cudaMemcpy(params.lock, &one, sizeof(int),
		cudaMemcpyHostToDevice);
	if (custat != cudaSuccess)
	{
		fprintf(stderr, "Cannot initialize GPU lock buffer: %s\n",
			cudaGetErrorString(custat));
		return 1;
	}
	
	// Create barrier used to guarantee gpu monitoring
	// kernel would be started before target GPU kernel.
	int status = pthread_barrier_init(&params.barrier, NULL, 2);
	if (status)
	{
		fprintf(stderr, "Cannot initialize barrier\n");
		return 1;
	}
	
	// Start monitoring CPU thread.
	pthread_t thread;
	status = pthread_create(&thread, NULL, &cpu_monitor, &params);
	if (status)
	{
		fprintf(stderr, "Cannot create monitoring thread\n");
		return 1;
	}
	
	// Synchronize main thread and monitor thread at barrier.
	pthread_barrier_wait(&params.barrier);
	
	// Execute target GPU kernel.
	gpu_kernel<<<1, 1, 1, stream>>>(gpu_data, size, npasses, params.lock,
		params.finish, params.maxidx, params.maxval);
	custat = cudaGetLastError();
	if (custat != cudaSuccess)
	{
		fprintf(stderr, "Cannot launch target GPU kernel: %s\n",
			cudaGetErrorString(custat));
		return 1;
	}

	// Synchronize main thread and monitor thread at barrier.
	pthread_barrier_wait(&params.barrier);

	status = pthread_barrier_destroy(&params.barrier);
	if (status)
	{
		fprintf(stderr, "Cannot destroy monitoring thread\n");
		return 1;
	}

	custat = cudaThreadSynchronize();
	if (custat != cudaSuccess)
	{
		fprintf(stderr, "Cannot synchronize GPU kernel: %s\n",
			cudaGetErrorString(custat));
		return 1;
	}
	
	custat = cudaFree(gpu_data);
	if (custat != cudaSuccess)
	{
		fprintf(stderr, "Cannot release GPU data buffer: %s\n",
			cudaGetErrorString(custat));
		return 1;
	}
	custat = cudaFree(params.maxidx);
	if (custat != cudaSuccess)
	{
		fprintf(stderr, "Cannot release GPU maxidx buffer: %s\n",
			cudaGetErrorString(custat));
		return 1;
	}
	custat = cudaFree(params.maxval);
	if (custat != cudaSuccess)
	{
		fprintf(stderr, "Cannot release GPU maxval buffer: %s\n",
			cudaGetErrorString(custat));
		return 1;
	}
	custat = cudaFree(params.finish);
	if (custat != cudaSuccess)
	{
		fprintf(stderr, "Cannot release GPU finish buffer: %s\n",
			cudaGetErrorString(custat));
		return 1;
	}
	custat = cudaFree(params.lock);
	if (custat != cudaSuccess)
	{
		fprintf(stderr, "Cannot release GPU lock buffer: %s\n",
			cudaGetErrorString(custat));
		return 1;
	}

	custat = cudaStreamDestroy(params.stream);
	if (custat != cudaSuccess)
	{
		fprintf(stderr, "Cannot create monitoring stream: %s\n",
			cudaGetErrorString(custat));
		return 1;
	}
	custat = cudaStreamDestroy(stream);
	if (custat != cudaSuccess)
	{
		fprintf(stderr, "Cannot create monitoring stream: %s\n",
			cudaGetErrorString(custat));
		return 1;
	}

	void* retval;
	status = pthread_join(thread, &retval);
	if (status)
	{
		fprintf(stderr, "Cannot join monitoring thread\n");
		return 1;
	}

	return (int)(size_t)retval;
}

