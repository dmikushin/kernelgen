#include <cuda_runtime.h>
#include <cupti_events.h>
#include <malloc.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

__global__ void gpu_kernel(float* data, size_t size, int npasses,
	int* lock, int* pmaxidx, float* pmaxval)
{
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
	
		// Issue event.
		__prof_trigger(0);

		// Lock thread to wait for response.
		*lock = 1;
		while (atomicCAS(lock, 0, 0)) continue;
	}
}

__global__ void gpu_unlock(int* lock)
{
	// Unlock thread waiting for response
	// in main gpu kernel.
	atomicCAS(lock, 1, 0);
}

typedef struct
{
	int done;
	unsigned int timeout;
	pthread_mutex_t mutex;
	CUdevice device;
	CUcontext context;
	int* lock;
}
monitor_params_t;

void* monitor(void* arg)
{
	monitor_params_t* params = (monitor_params_t*)arg;

	CUpti_EventGroup eventGroup;
	CUptiResult status = cuptiEventGroupCreate(params->context, &eventGroup, 0);
	if (status != CUPTI_SUCCESS)
	{
		const char* msg; cuptiGetResultString(status, &msg);
		fprintf(stderr, "Cannot create CUPTI event group: %s\n", msg);
		pthread_exit((void*)1);
		return NULL;
	}

	CUpti_EventID eventId;	
	status = cuptiEventGetIdFromName(params->device, "prof_trigger_00", &eventId);
	if (status != CUPTI_SUCCESS)
	{
		const char* msg; cuptiGetResultString(status, &msg);
		fprintf(stderr, "Cannot get CUPTI event id from name: %s\n", msg);
		pthread_exit((void*)1);
		return NULL;
	}

	status = cuptiEventGroupAddEvent(eventGroup, eventId);
	if (status != CUPTI_SUCCESS)
	{
		const char* msg; cuptiGetResultString(status, &msg);
		fprintf(stderr, "Cannot add event to group: %s\n", msg);
		pthread_exit((void*)1);
		return NULL;				
	}

	status = cuptiEventGroupEnable(eventGroup);
	if (status != CUPTI_SUCCESS)
	{
		const char* msg; cuptiGetResultString(status, &msg);
		fprintf(stderr, "Cannot enable event group: %s\n", msg);
		pthread_exit((void*)1);
		return NULL;
	}

	while (1)
	{
		uint64_t value;
		size_t szevent = sizeof(value);
		status = cuptiEventGroupReadEvent(
			eventGroup, CUPTI_EVENT_READ_FLAG_NONE,
			eventId, &szevent, &value);
		if (status != CUPTI_SUCCESS)
		{
			const char* msg; cuptiGetResultString(status, &msg);
			fprintf(stderr, "Cannot read CUPTI event value: %s\n", msg);
			pthread_exit((void*)1);
			return NULL;
		}
		
		if (szevent != sizeof(value))
		{
			const char* msg; cuptiGetResultString(status, &msg);
			fprintf(stderr, "Incorrect length of read CUPTI event\n", msg);
			pthread_exit((void*)1);
			return NULL;
		}
	
		printf("%s: %llu\n", "prof_trigger_00", (unsigned long long)value);

		// Unlock kernel being monitored.
		gpu_unlock<<<1, 1, 1>>>(params->lock);
		cudaError_t custat = cudaGetLastError();
		if (custat != cudaSuccess)
		{
			fprintf(stderr, "Cannot launch GPU kernel: %s\n",
				cudaGetErrorString(custat));
			pthread_exit((void*)1);
			return NULL;
		}

		usleep(params->timeout);

		// Check if we need to finish monitoring.		
		pthread_mutex_lock(&params->mutex);
		if (params->done)
		{
			pthread_mutex_unlock(&params->mutex);
			pthread_exit((void*)0);
			break;
		}
		pthread_mutex_unlock(&params->mutex);
	}

	return NULL;
}

int main(int argc, char* argv[])
{
	if (argc != 4)
	{
		printf("%s <timeout> <size> <npasses>\n", argv[0]);
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
	params.timeout = atoi(argv[1]);
	params.done = 0;

	int status = pthread_mutex_init(&params.mutex, NULL);
	if (status)
	{
		fprintf(stderr, "Cannot create mutex\n");
		return 1;
	}

	CUresult cuerr = cuDeviceGet(&params.device, 0);
	if (cuerr != CUDA_SUCCESS)
	{
		fprintf(stderr, "Cannot get CUDA device\n");
		return 1;
	}

	char name[32];
	cuerr = cuDeviceGetName(name, 32, params.device);
	if (cuerr != CUDA_SUCCESS)
	{
		fprintf(stderr, "Cannot get CUDA device name\n");
		return 1;
	}

	cuerr = cuCtxCreate(&params.context, 0, params.device);
	if (cuerr != CUDA_SUCCESS)
	{
		fprintf(stderr, "Cannot create CUDA device context\n");
		return 1;
	}

	size_t size = atoi(argv[2]);
	int npasses = atoi(argv[3]);

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
	
	int* pmaxidx = NULL;
	custat = cudaMalloc((void**)&pmaxidx, sizeof(int));
	if (custat != cudaSuccess)
	{
		fprintf(stderr, "Cannot create GPU pmaxidx buffer: %s\n",
			cudaGetErrorString(custat));
		return 1;
	}
	
	float* pmaxval = NULL;
	custat = cudaMalloc((void**)&pmaxval, sizeof(float));
	if (custat != cudaSuccess)
	{
		fprintf(stderr, "Cannot create GPU pmaxval buffer: %s\n",
			cudaGetErrorString(custat));
		return 1;
	}
	
	custat = cudaMalloc((void**)&params.lock, sizeof(int));
	if (custat != cudaSuccess)
	{
		fprintf(stderr, "Cannot create GPU lock buffer: %s\n",
			cudaGetErrorString(custat));
		return 1;
	}

	// Start monitoring thread.
	pthread_t thread;
	status = pthread_create(&thread, NULL, &monitor, &params);
	if (status)
	{
		fprintf(stderr, "Cannot create monitoring thread\n");
		return 1;
	}
	
	// Execute GPU kernel.
	gpu_kernel<<<1, 1, 1>>>(gpu_data, size, npasses, params.lock, pmaxidx, pmaxval);
	custat = cudaGetLastError();
	if (custat != cudaSuccess)
	{
		fprintf(stderr, "Cannot launch GPU kernel: %s\n",
			cudaGetErrorString(custat));
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
	custat = cudaFree(pmaxidx);
	if (custat != cudaSuccess)
	{
		fprintf(stderr, "Cannot release GPU pmaxidx buffer: %s\n",
			cudaGetErrorString(custat));
		return 1;
	}
	custat = cudaFree(pmaxval);
	if (custat != cudaSuccess)
	{
		fprintf(stderr, "Cannot release GPU pmaxval buffer: %s\n",
			cudaGetErrorString(custat));
		return 1;
	}

	// Tell monitoring thread to finish monitoring.
	pthread_mutex_lock(&params.mutex);
	params.done = 1;
	pthread_mutex_unlock(&params.mutex);

	status = pthread_mutex_destroy(&params.mutex);
	if (status)
	{
		fprintf(stderr, "Cannot destroy mutex\n");
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

