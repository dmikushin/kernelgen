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

#include <gforscale.h>
#include <malloc.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#define real float

struct pair { real x, y; };

void derived_(int* m, int* n, struct pair* xy_in, real* xy);

int main(int argc, char* argv[])
{
	if (argc != 3)
	{
		printf("Usage: %s <m> <n>\n", argv[0]);
		return 0;
	}
	
	int m = atoi(argv[1]);
	int n = atoi(argv[2]);
	struct pair* xy_in =
		(struct pair*)malloc(sizeof(struct pair) * m * n);
	real* xy_out = (real*)malloc(sizeof(real) * m * n);
	double invrmax = 1.0 / RAND_MAX;
	for (int i = 0; i < m * n; i++)
	{
		xy_in[i].x = rand() * invrmax;
		xy_in[i].y = rand() * invrmax;
	}
	
	derived_(&m, &n, xy_in, xy_out);

	free(xy_in);
	free(xy_out);

	gforscale_status_t status = gforscale_get_last_error();
	if (status.value != gforscale_success)
	{
		printf("%s\n", gforscale_get_error_string(status));
		return 1;
	}
	
	return 0;
}

