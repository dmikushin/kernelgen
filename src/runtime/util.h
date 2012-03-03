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

#ifndef KERNELGEN_UTIL_H
#define KERNELGEN_UTIL_H

#include <gelf.h>

#ifdef __cplusplus

#include <iostream>

#define THROW(message) { std::cerr << __FILE__ << ":" << __LINE__ << " " << message << endl; throw; }

#include <list>
#include <string>

// Execute the specified command in the system shell, supplying
// input stream content and returning results from output and
// error streams.
int execute(std::string command, std::list<std::string> args,
	std::string in = "", std::string* out = NULL, std::string* err = NULL);

namespace kernelgen { namespace runtime {

class timer
{
	timespec time_start, time_stop;
	bool started;

public :

	static timespec get_resolution();

	timer(bool start = true);

	timespec start();
	timespec stop();

	double get_elapsed(timespec* start = NULL);
};

} }

#endif

#ifdef __cplusplus
extern "C"
{
#endif

// Load the specified ELF image symbol raw data.
int util_elf_read(const int fd, const char* symname,
	char** symdata, size_t* symsize);

// Load the specified ELF image symbols raw data.
int util_elf_read_many(const int fd, const int count, 
	const char** symnames, char** symdatas, size_t* symsizes);

// Load the specified ELF executable header.
int util_elf_read_eheader(
	const char* executable, GElf_Ehdr* ehdr);

// Create ELF image containing symbol with the specified name,
// associated data content and its length.
int util_elf_write(const int fd, const int arch,
	const char* symname, const char* symdata, size_t length);

// Create ELF image containing multiple symbols with the specified names,
// associated data contents and their lengths.
int util_elf_write_many(const int fd, const int arch,
	const int count, ...);

int util_elf_find(const int fd, const char* symname_pattern,
	char*** symnames, int* count);

#ifdef __cplusplus
}
#endif

#endif // KERNELGEN_UTIL_H

