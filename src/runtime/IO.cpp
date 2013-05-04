//===- IO.cpp - KernelGen old temp files API (deprecated) -----------------===//
//
//     KernelGen -- A prototype of LLVM-based auto-parallelizing Fortran/C
//        compiler for NVIDIA GPUs, targeting numerical modeling code.
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "KernelGen.h"

#include "Util.h"
#include "IO.h"

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

