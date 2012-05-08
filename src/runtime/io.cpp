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
 * 
 */

#include "util.h"
#include "io.h"

#include <cstdlib>
#include <fcntl.h>
#include <string.h>

using namespace std;
using namespace util::io;

cfiledesc::cfiledesc() : fd(-1), filename("") { }

const string& cfiledesc::getFilename() const { return filename; }

int cfiledesc::getFileDesc() const { return fd; }

cfiledesc::cfiledesc(std::string& filename, int flags) : fd(-1), filename(filename)
{
	fd = open(filename.c_str(), flags);
	if (fd == -1) THROW("Cannot open file " << filename);
}

cfiledesc::~cfiledesc()
{
	if (fd >= 0) close(fd);
}

cfiledesc cfiledesc::mktemp(string prefix)
{
	cfiledesc fd;
	string prefix_xxx = prefix + "XXXXXX";
	char* c_prefix_xxx = new char[prefix_xxx.size() + 1];
	strcpy(c_prefix_xxx, prefix_xxx.c_str());
	fd.fd = mkstemp(c_prefix_xxx);
	fd.filename = c_prefix_xxx;
	delete[] c_prefix_xxx;
	return fd;
}

