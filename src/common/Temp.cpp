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
