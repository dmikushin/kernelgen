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

#include "kernelgen.h"
#include "util.h"

#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <malloc.h>
#include <string.h>

using namespace std;

int main(int argc, char* argv[])
{
	//
	// Behave like compiler if no arguments.
	//
	if (argc == 1)
	{
		cout << "kernelgen: no input files" << endl;
		return 0;
	}

	//
	// Enable or disable verbose output.
	//
	int verbose = 1;

	//
	// Switch to bypass the kernelgen pipe and use regular compiler only.
	//
	int bypass = 0;
	
	//
	// Target architecture.
	//
	int arch = 64;

	//
	// The regular compiler used for host-side source code.
	//
	const char* host_compiler = "gfortran";

	//
	// Supported source code files extensions.
	//
	list<string> source_ext;
	source_ext.push_back(".c");
	source_ext.push_back(".f");
	source_ext.push_back(".f90");
	source_ext.push_back(".F");
	source_ext.push_back(".F90");

	//
	// Split kgen args from other args in the command line.
	//
	list<string> args, kgen_args;
	for (int i = 1; i < argc; i++)
	{
		char* arg = argv[i];
		if (!strncmp(arg, "-Wk,", 4))
			kgen_args.push_back(arg);
		else
			args.push_back(arg);

		// In case of 32-bit compilation on 64-bit,
		// invoke object mergering command with 32-bit flag.		
		if (!strcmp(arg, "-m32"))
			arch = 32;
	}

	//
	// Interpret kgen args.
	//
	for (list<string>::iterator it = kgen_args.begin(); it != kgen_args.end(); it++)
	{
		const char* arg = (*it).c_str();
		
		if (!strcmp(arg, "-Wk,--bypass"))
			bypass = 1;
		if (!strcmp(arg, "-Wk,--verbose"))
			verbose = 1;
		if (!strncmp(arg, "-Wk,--host-compiler=", 20))
			host_compiler = arg + 20;
	}

	//
	// Find source code input.
	// FIXME There could be multiple source files supplied,
	// currently this case is unhandled.
	//
	string input = "";
	for (list<string>::iterator it1 = args.begin(); (it1 != args.end()) && !input.size(); it1++)
	{
		const char* arg = (*it1).c_str();
		for (list<string>::iterator it2 = source_ext.begin(); it2 != source_ext.end(); it2++)
		{
			const char* ext = (*it2).c_str();
			if (!strcmp(arg + strlen(arg) - strlen(ext), ext))
			{
				input = *it1;
				break;
			}
		}
	}

	//
	// Find output file in args.
	// There could be "-c" or "-o" option or both.
	// With "-c" source is compiled only, producing by default
	// an object file with same basename as source.
	// With "-o" source could either compiled only (with additional
	// "-c") or fully linked, but in both cases output is sent to
	// explicitly defined file after "-o" option.
	//
	string output = "";
	for (list<string>::iterator it = args.begin(); it != args.end(); it++)
	{
		const char* arg = (*it).c_str();
		if (!strcmp(arg, "-o"))
		{
			it++;
			output = *it;
			break;
		}
	}
	for (list<string>::iterator it = args.begin(); (it != args.end()) && !output.size(); it++)
	{
		const char* arg = (*it).c_str();
		if (!strcmp(arg, "-c"))
		{
			// Trim path.
			it++;
			arg = (*it).c_str();
			output = *it;
			for (int i = output.size(); i >= 0; i--)
			{
				if (output[i] == '/')
				{
					output = arg + i + 1;
					break;
				}
			}

			// Replace source extension with object extension.
			for (int i = output.size(); i >= 0; i--)
			{
				if (output[i] == '.')
				{
					output[i + 1] = 'o';
					output[i + 2] = '\0';
					break;
				}
			}
		}
	}

	//
	// Only execute the regular host compiler, if required.
	//
	if (bypass)
		return execute(host_compiler, args, "", NULL, NULL);

	//
	// Linker used to merge multiple objects into single one.
	//
	string merge = "ld";
	list<string> merge_args;

	//
	// Temporary files location prefix.
	//
	string fileprefix = "/tmp/";

	//
	// Interpret kernelgen compile options.
	//
	for (list<string>::iterator iarg = kgen_args.begin(),
		iearg = kgen_args.end(); iarg != iearg; iarg++)
	{
		const char* arg = (*iarg).c_str();		
		if (!strcmp(arg, "-Wk,--keep"))
			fileprefix = "";
	}

	if (arch == 32)
		merge_args.push_back("-melf_i386");
	merge_args.push_back("--unresolved-symbols=ignore-all");
	merge_args.push_back("-r");
	merge_args.push_back("-o");

	//
	// Do only regular compilation for file extensions
	// we do not know. Also should cover the case of linking.
	//
	if (input.size())
		return compile(args, kgen_args,
			merge, merge_args, input, output,
			verbose, arch, host_compiler, fileprefix);
	else
		return link(args, kgen_args,
			merge, merge_args, input, output,
			verbose, arch, host_compiler, fileprefix);
}

