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

#include "bind.h"
#include "io.h"
#include "util.h"
#include "runtime.h"

#include <fstream>

using namespace kernelgen::bind::cuda;
using namespace util::io;
using namespace std;

#define PTX_LOG_SIZE 1024

char* kernelgen::runtime::nvopencc(string source, string name)
{
	// Dump generated kernel object to first temporary file.
	cfiledesc tmp1 = cfiledesc::mktemp("/tmp/");
	{
		fstream tmp_stream;
		tmp_stream.open(tmp1.getFilename().c_str(),
			fstream::binary | fstream::out | fstream::trunc);
		tmp_stream << source;
		tmp_stream.close();
	}

	// Compile CUDA code in temporary file.
	cfiledesc tmp2 = cfiledesc::mktemp("/tmp/");
	{
		string nvopencc = "nvopencc";
		std::list<string> nvopencc_args;
		nvopencc_args.push_back("-D__CUDA_DEVICE_FUNC__");
		nvopencc_args.push_back("-I/opt/kernelgen/include");
		nvopencc_args.push_back("-include");
		nvopencc_args.push_back("kernelgen_runtime.h");
		nvopencc_args.push_back("-x");
		nvopencc_args.push_back("c");
		nvopencc_args.push_back("-TARG:compute_20");
		nvopencc_args.push_back("-m64");
		nvopencc_args.push_back("-OPT:ftz=1");
		nvopencc_args.push_back("-CG:ftz=1");
		nvopencc_args.push_back("-CG:prec_div=0");
		nvopencc_args.push_back("-CG:prec_sqrt=0");
		nvopencc_args.push_back(tmp1.getFilename());
		nvopencc_args.push_back("-o");
		nvopencc_args.push_back(tmp2.getFilename());
		if (verbose)
		{
			cout << nvopencc;
                        for (std::list<string>::iterator it = nvopencc_args.begin();
                                it != nvopencc_args.end(); it++)
                                cout << " " << *it;
                        cout << endl;
                }
                execute(nvopencc, nvopencc_args, "", NULL, NULL);
	}

	// Load PTX into string.
	string ptx;
	{
		std::ifstream tmp_stream(tmp2.getFilename().c_str());
		tmp_stream.seekg(0, std::ios::end);   
		ptx.reserve(tmp_stream.tellg());
		tmp_stream.seekg(0, std::ios::beg);

		ptx.assign((std::istreambuf_iterator<char>(tmp_stream)),
			std::istreambuf_iterator<char>());
		tmp_stream.close();
	}

	cout << ptx;
	
	// Load PTX from string into module.
	void* module;
	char log[PTX_LOG_SIZE] = "", elog[PTX_LOG_SIZE] = "";
	int options[] =
	{
		CU_JIT_INFO_LOG_BUFFER, CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES,
		CU_JIT_ERROR_LOG_BUFFER, CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES
	};
	void* values[] =
	{
		&log, (void*)PTX_LOG_SIZE,
		&elog, (void*)PTX_LOG_SIZE
	};
	int err = cuModuleLoadDataEx(&module, ptx.c_str(), 4, options, values);
	if (verbose)
		cout << log << endl;
	if (err)
		THROW("Error in cuModuleLoadData " << err << " " << elog);

	kernel_func_t kernel_func = NULL;
	err = cuModuleGetFunction((void**)&kernel_func, module, name.c_str());
	if (err)
		THROW("Error in cuModuleGetFunction " << err);

	if (verbose)
		cout << "Loaded '" << name << "' at: " << (void*)kernel_func << endl;
	
	return (char*)kernel_func;
}

