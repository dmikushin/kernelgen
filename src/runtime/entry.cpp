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

#include "runtime.h"
#include "util/elf.h"

#include <iostream>
#include <cstdlib>

// Regular main entry.
extern "C" int __regular_main(int argc, char* argv[]);

using namespace kernelgen;
using namespace std;
using namespace util::elf;

// Kernels runmode (target).
int kernelgen::runmode = -1;

// Verbose output.
bool kernelgen::verbose = false;

// The pool of already loaded kernels.
// After kernel is loaded, we pin it here
// for futher references.
std::map<string, kernel_t> kernelgen::kernels;

int main(int argc, char* argv[])
{
	char* crunmode = getenv("kernelgen_runmode");
	if (crunmode)
	{	
		runmode = atoi(crunmode);

		// Check requested verbosity level.
		char* cverbose = getenv("kernelgen_verbose");
		if (cverbose) verbose = (bool)atoi(cverbose);
		
		if (verbose)
		{
			switch (runmode)
			{
			case KERNELGEN_RUNMODE_NATIVE :
				cout << "Using KernelGen/NATIVE" << endl;
				break;
			case KERNELGEN_RUNMODE_CUDA :
				cout << "Using KernelGen/CUDA" << endl;
				break;
			case KERNELGEN_RUNMODE_OPENCL :
				cout << "Using KernelGen/OpenCL" << endl;
				break;
			}
			cout << endl;
		}

		// Build kernels index.
		if (verbose) cout << "Building kernels index ..." << endl;
		celf e("/proc/self/exe", "");
		cregex regex("^__kernelgen_.*$", REG_EXTENDED | REG_NOSUB);
		vector<csymbol*> symbols = e.getSymtab()->find(regex);
		for (vector<csymbol*>::iterator i = symbols.begin(),
			ie = symbols.end(); i != ie; i++)
		{
			csymbol* symbol = *i;
			const char* data = symbol->getData();
			const string& name = symbol->getName();
			kernel_t kernel;
			kernel.name = name;
			kernel.source = data;
			kernels[name] = kernel;
			if (verbose) cout << name << endl;
		}
		if (verbose) cout << endl;

		// Invoke entry point kernel.
		int szargs[2] = { sizeof(int), sizeof(char**) };
		kernel_t kernel = kernels["__kernelgen_main"];
		return kernelgen_launch((char*)&kernel, 2, szargs, argc, argv);
	}
	
	// Chain to entry point of the regular binary.
	return __regular_main(argc, argv);
}

