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

#include "llvm/LLVMContext.h"
#include "llvm/Module.h"
#include "llvm/PassManager.h"
#include "llvm/Support/FormattedStream.h"
#include "llvm/Support/IRReader.h"
#include "llvm/Support/Host.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/ToolOutputFile.h"
#include "llvm/Target/TargetData.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetRegistry.h"
#include "llvm/Target/TargetSelect.h"

#include "io.h"
#include "util.h"
#include "runtime.h"

#include <dlfcn.h>
#include <fstream>
#include <list>

using namespace util::io;
using namespace llvm;
using namespace std;

static auto_ptr<TargetMachine> mcpu;

void kernelgen::runtime::compile(
	int runmode, kernel_t* kernel, int* args)
{
	// Load LLVM IR source into module.
	LLVMContext &context = getGlobalContext();
	SMDiagnostic diag;
	MemoryBuffer* buffer = MemoryBuffer::getMemBuffer(kernel->source);
	Module* m = ParseIR(buffer, diag, context);
	if (!m)
		THROW(kernel->name << ":" << diag.getLineNo() << ": " <<
			diag.getLineContents() << ": " << diag.getMessage());
	m->setModuleIdentifier(kernel->name + "_module");
	
	//m->dump();
	
	// Emit target assembly and binary image, depending
	// on runmode.
	switch (runmode)
	{
		case KERNELGEN_RUNMODE_NATIVE :
		{
			// Create target machine and get its target data.
			if (!mcpu.get())
			{
				InitializeAllTargets();
				InitializeAllTargetMCs();
				InitializeAllAsmPrinters();
				InitializeAllAsmParsers();

				Triple triple(m->getTargetTriple());
				if (triple.getTriple().empty())
					triple.setTriple(sys::getHostTriple());
				string err;
				InitializeAllTargets();
				const Target* target = TargetRegistry::lookupTarget(triple.getTriple(), err);
				if (!target)
					THROW("Error auto-selecting target for module '" << err << "'." << endl <<
						"Please use the -march option to explicitly pick a target.");
				mcpu.reset(target->createTargetMachine(
					triple.getTriple(), "", "", Reloc::PIC_, CodeModel::Default));
				if (!mcpu.get())
					THROW("Could not allocate target machine");
			}
			
			const TargetData* tdata = mcpu.get()->getTargetData();
			PassManager manager;
			manager.add(new TargetData(*tdata));

			// Override default to generate verbose assembly.
			mcpu.get()->setAsmVerbosityDefault(true);

			// Setup output stream.
			string bin_string;
			raw_string_ostream bin_stream(bin_string);
			formatted_raw_ostream stream(bin_stream);

			// Ask the target to add backend passes as necessary.
			if (mcpu.get()->addPassesToEmitFile(manager, stream,
				TargetMachine::CGFT_ObjectFile, CodeGenOpt::Aggressive))
				THROW("Target does not support generation of this file type");

			manager.run(*m);
			
			// Flush the resulting object binary to the
			// underlying string.
			stream.flush();

			//cout << bin_string;

			// Dump generated kernel object to first temporary file.
			cfiledesc tmp1 = cfiledesc::mktemp("/tmp/");
			{
				fstream tmp_stream;
				tmp_stream.open(tmp1.getFilename().c_str(),
					fstream::binary | fstream::out | fstream::trunc);
				tmp_stream << bin_string;
				tmp_stream.close();
			}
			
			// Link first and second objects together into third one.
			cfiledesc tmp3 = cfiledesc::mktemp("/tmp/");
			{
				string linker = "ld";
				std::list<string> linker_args;
				linker_args.push_back("-shared");
				linker_args.push_back("-o");
				linker_args.push_back(tmp3.getFilename());
				linker_args.push_back(tmp1.getFilename());
				if (verbose)
				{
					cout << linker;
					for (std::list<string>::iterator it = linker_args.begin();
						it != linker_args.end(); it++)
						cout << " " << *it;
					cout << endl;
				}
				execute(linker, linker_args, "", NULL, NULL);
			}

			// Load linked image and extract kernel entry point.
			void* handle = dlopen(tmp3.getFilename().c_str(),
				RTLD_NOW | RTLD_GLOBAL | RTLD_DEEPBIND);
			if (!handle)
				THROW("Cannot dlopen " << dlerror());

			kernel_func_t kernel_func = (kernel_func_t)dlsym(handle, kernel->name.c_str());
			if (!kernel_func)
				THROW("Cannot dlsym " << dlerror());
			
			if (verbose)
				cout << "Loaded '" << kernel->name << "' at: " << (void*)kernel_func << endl;

			kernel_func(args);

			break;
		}
		case KERNELGEN_RUNMODE_CUDA :
		{
			break;
		}
		case KERNELGEN_RUNMODE_OPENCL :
		{
			break;
		}
		default :
			THROW("Unknown runmode " << runmode);
	}	
}

