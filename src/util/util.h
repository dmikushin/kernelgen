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

#ifndef UTIL_H
#define UTIL_H

#include <gelf.h>

#ifdef __cplusplus
extern "C"
{
#endif

#pragma pack(push, 1)

// The built-in timer value type.
typedef struct
{
	int64_t seconds;
	int64_t nanoseconds;
}
util_time_t;

#pragma pack(pop)

void util_get_timer_resolution(util_time_t* val);

// Get the built-in timer value.
void util_get_time(util_time_t* val);

// Get the built-in timer measured values difference.
double util_get_time_diff(
	util_time_t* val1, util_time_t* val2);

// Print the built-in timer measured values difference.
void util_print_time_diff(
	util_time_t* val1, util_time_t* val2);

// Load contents of the specified text file.
int util_load_source(
	const char* filename, char** source, size_t* szsource);

// Load the specified ELF image symbol raw data.
int util_elf_read(const char* filename, const char* symname,
	char** symdata, size_t* symsize);

// Load the specified ELF executable header.
int util_elf_read_eheader(
	const char* executable, GElf_Ehdr* ehdr);

// Create ELF image containing symbol with the specified name,
// associated data content and its length. Certain ELF properties
// could be taken from the specified reference executable, if not NULL.
int util_elf_write(const char* filename, GElf_Ehdr* ehdr,
	const char* symname, const char* symdata, size_t length);

// Create ELF image containing multiple symbols with the specified names,
// associated data contents and their lengths. Certain ELF properties
// could be taken from the specified reference executable, if not NULL.
int util_elf_write_many(const char* filename, GElf_Ehdr* ehdr,
	int count, ...);

#ifdef __cplusplus
}
#endif

#endif // UTIL_H

