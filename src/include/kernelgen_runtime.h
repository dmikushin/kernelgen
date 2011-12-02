#include <stddef.h>

extern __attribute__((device)) void* malloc(size_t);
extern __attribute__((device)) void free(void*);

__attribute__((device)) void kernelgen_hostcall(unsigned char* name, unsigned int* args)
{
}

__attribute__((device)) int kernelgen_launch(unsigned char* name, unsigned int* args)
{
	return -1;
}

