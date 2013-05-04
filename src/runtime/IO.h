//===- IO.h - KernelGen old temp files API (deprecated) -------------------===//
//
//     KernelGen -- A prototype of LLVM-based auto-parallelizing Fortran/C
//        compiler for NVIDIA GPUs, targeting numerical modeling code.
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef KERNELGEN_IO_H
#define KERNELGEN_IO_H

#include <string>

namespace util { namespace io {

class cfiledesc
{
	int fd;
	std::string filename;

	cfiledesc();
public :
	const std::string& getFilename() const;
	
	int getFileDesc() const;

	cfiledesc(std::string& filename, int flags);

	~cfiledesc();
	
	static cfiledesc mktemp(std::string prefix);
};

} } // namespace

#endif // KERNELGEN_IO_H

