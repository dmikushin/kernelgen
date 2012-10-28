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

#include "Temp.h"

#include "llvm/Object/Archive.h"
#include "llvm/Support/FileSystem.h"

#include <iostream>

#include "runtime.h"

using namespace kernelgen::utils;
using namespace llvm;
using namespace llvm::object;
using namespace llvm::sys;
using namespace llvm::sys::fs;
using namespace std;

TempFile Temp::getFile(string mask, bool closefd)
{
	// Open unique file for the given mask.
	int fd;
	SmallString<128> filename_vector;
	if (error_code err = unique_file(mask, fd, filename_vector))
	{
		if (verbose)
			cerr << "Error " << err.value() << " at " << __FILE__ << ":" << __LINE__ << endl;
		throw err;
	}

	// Store filename.
	string filename = (StringRef)filename_vector;

	if (closefd) close(fd);

	// Create output file tracker.
	string err;
	tool_output_file file(filename.c_str(), err, raw_fd_ostream::F_Binary);
	if (!err.empty())
	{
		if (verbose)
			cerr << "Error " << err.c_str() << " at " << __FILE__ << ":" << __LINE__ << endl;
		throw err;
	}

	return TempFile(filename, fd, file);
}
