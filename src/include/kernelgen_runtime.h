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

#ifndef KERNELGEN_RUNTIME_H
#define KERNELGEN_RUNTIME_H

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

#endif // KERNELGEN_RUNTIME_H

