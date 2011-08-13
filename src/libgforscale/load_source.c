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
#include <sys/stat.h>

// Load contents of the specified text file.
int gforscale_load_source(const char* filename, char** source, size_t* szsource)
{
	if (!filename)
	{
		fprintf(stderr, "Invalid filename pointer\n");
		return 1;
	}
	if (!source)
	{
		fprintf(stderr, "Invalid source pointer\n");
		return 1;
	}
	if (!szsource)
	{
		fprintf(stderr, "Invalid size pointer\n");
		return 1;
	}
	FILE * fp = fopen(filename, "r");
	if (!fp)
	{
		fprintf(stderr, "Cannot open file %s\n", filename);
		return 1;
	}
	struct stat st; stat(filename, &st);
	*szsource = st.st_size;
	*source = (char*)malloc(sizeof(char) * *szsource);
	fread(*source, *szsource, 1, fp);
	int ierr = ferror(fp);
	fclose(fp);
	if (ierr)
	{
		fprintf(stderr, "Error reading from %s, code = %d", filename, ierr);
		return 1;
	}
	return 0;
}

