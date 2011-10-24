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

#include "util/util.h"
#include "runtime.h"

using namespace llvm;
using namespace std;

static auto_ptr<TargetMachine> mcpu;

void kernelgen::runtime::compile(int runmode, char* source, char** binary)
{
	// Load LLVM IR source into module.
	LLVMContext &context = getGlobalContext();
	SMDiagnostic diag;
	MemoryBuffer* buffer =
		MemoryBuffer::getMemBuffer(source);
	auto_ptr<Module> module;
	module.reset(ParseIR(buffer, diag, context));

	// Emit target assembly and binary image, depending
	// on runmode.
	switch (runmode)
	{
		case KERNELGEN_RUNMODE_NATIVE :
		{
			// Create target machine and get its target data.
			if (!mcpu.get())
			{
				Triple triple(module.get()->getTargetTriple());
				if (triple.getTriple().empty())
					triple.setTriple(sys::getHostTriple());
				string err;
				InitializeAllTargets();
				const Target* target = TargetRegistry::lookupTarget(triple.getTriple(), err);
				if (!target)
					THROW("Error auto-selecting target for module '" << err << "'." << endl <<
						"Please use the -march option to explicitly pick a target.");
				mcpu.reset(target->createTargetMachine(
					triple.getTriple(), "", "", Reloc::Default, CodeModel::Default));
				if (!mcpu.get())
					THROW("Could not allocate target machine");
			}
			
			// Insert third extra argument - an array of original
			// function arguments sizes. Note for now we just insert
			// pointer to uninitialized array. Values will be set
			// later, depending on target machine.
			/*std::vector<Constant*> sizes;
			for (int i = 0; i != nargs; i++)
			{
				Value* arg = call->getArgOperand(i);
				int size = tdata->getTypeStoreSize(arg->getType());
				sizes.push_back(ConstantInt::get(int32Ty, size));
			}*/

			const TargetData* tdata = mcpu.get()->getTargetData();
			PassManager manager;
			manager.add(new TargetData(*tdata));

			// Override default to generate verbose assembly.
			mcpu.get()->setAsmVerbosityDefault(true);

			// Setup output stream.
			std::string error;
			unsigned flags = raw_fd_ostream::F_Binary;
			auto_ptr<tool_output_file> fdout;
			fdout.reset(new tool_output_file("-", error, flags));
			if (!error.empty())
				THROW("Cannot create output stream : " << error);
			formatted_raw_ostream stream(fdout.get()->os());

			// Ask the target to add backend passes as necessary.
			if (mcpu.get()->addPassesToEmitFile(manager, stream,
				TargetMachine::CGFT_ObjectFile, CodeGenOpt::Aggressive))
				THROW("Target does not support generation of this file type");
			
			manager.run(*module.get());
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

