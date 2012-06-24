/*
 * KGen - the LLVM-based compiler with GPU kernels generation over C backend.
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

#include <malloc.h>
#include <stdio.h>
#include <stdlib.h>

#define real float

//const char *a = "aa", *z = "zz";

void sincos_(int* nx, int* ny, int* nz, real* x, real* y, real* xy);

int main(int argc, char* argv[])
{
	if (argc != 4)
	{
		printf("Usage: %s <nx> <ny> <nz>\n", argv[0]);
		return 0;
	}

	int nx = atoi(argv[1]), ny = atoi(argv[2]), nz = atoi(argv[3]);
	real* x = (real*)malloc(sizeof(real) * nx * ny * nz);
	real* y = (real*)malloc(sizeof(real) * nx * ny * nz);
	real* xy = (real*)malloc(sizeof(real) * nx * ny * nz);
	double invrmax = 1.0 / RAND_MAX;
	for (int i = 0; i < nx * ny * nz; i++)
	{
		x[i] = rand() * invrmax;
		y[i] = rand() * invrmax;
	}
	
	sincos_(&nx, &ny, &nz, x, y, xy);
	sincos_(&nx, &ny, &nz, x, y, xy);

	nx--; ny--; nz--;
	sincos_(&nx, &ny, &nz, x, y, xy);

	free(x);
	free(y);
	free(xy);
	
	return 0;
}

