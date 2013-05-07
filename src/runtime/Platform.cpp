//===- Platform.cpp - KernelGen runtime API --------------------------------===//
//
//     KernelGen -- A prototype of LLVM-based auto-parallelizing Fortran/C
//        compiler for NVIDIA GPUs, targeting numerical modeling code.
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines KernelGen target platforms.
//
//===----------------------------------------------------------------------===//

#include "Platform.h"
#include "Runtime.h"

#include "llvm/MC/MCContext.h"
#include "llvm/Support/Host.h"

using namespace llvm;
using namespace std;

namespace kernelgen {
namespace runtime {
TargetPlatforms platforms;

TargetPlatform::TargetPlatform(const Target* target, TargetMachine* machine,
		StringRef triple) : target(target), machine(machine), triple(triple)
{
	const MCAsmInfo *MAI = target->createMCAsmInfo(triple);
	if (!MAI)
		THROW("Unable to create target asm info");
	const MCRegisterInfo *MRI = target->createMCRegInfo(triple);
	if (!MRI)
		THROW("Unable to create target register info!");
	mccontext.reset(new MCContext(*MAI, *MRI, 0));
	mangler.reset(new Mangler(*mccontext.get(), *machine->getDataLayout()));
}

// Get target platform for the specified runmode.
TargetPlatform* TargetPlatforms::operator[](int runmode) {
	TargetPlatform* platform = platforms[runmode].get();
	if (platform)
		return platform;

	InitializeAllTargets();
	InitializeAllTargetMCs();
	InitializeAllAsmPrinters();
	InitializeAllAsmParsers();

	switch (runmode) {
	case KERNELGEN_RUNMODE_NATIVE: {
		Triple triple;
		triple.setTriple(sys::getDefaultTargetTriple());
		string err;
		TargetOptions options;
		const Target* target = TargetRegistry::lookupTarget(triple.getTriple(), err);
		if (!target)
			THROW("Error auto-selecting native target: " << err << endl);
		TargetMachine* machine = target->createTargetMachine(triple.getTriple(),
				"", "", options, Reloc::PIC_, CodeModel::Default);
		if (!machine)
			THROW("Could not allocate target machine");

		// Override default to generate verbose assembly.
		machine->setAsmVerbosityDefault(true);

		platform = new TargetPlatform(target, machine, triple.getTriple());
		platforms[KERNELGEN_RUNMODE_NATIVE].reset(platform);
		return platform;
	}
	case KERNELGEN_RUNMODE_CUDA: {
		const Target* target = NULL;
		Triple triple(monitor_module->getTargetTriple());
		for (TargetRegistry::iterator it = TargetRegistry::begin(), ie =
				TargetRegistry::end(); it != ie; ++it) {
			if (!strcmp(it->getName(), "nvptx64")) {
				target = &*it;
				break;
			}
		}

		if (!target)
			THROW("LLVM is built without NVPTX Backend support");

		stringstream sarch;
		sarch << cuda_context->getSubarch();
		TargetMachine* machine = target->createTargetMachine(
				triple.getTriple(), sarch.str(), "+drv_cuda",
				TargetOptions(), Reloc::PIC_,
				CodeModel::Default, CodeGenOpt::Aggressive);
		if (!machine)
			THROW("Could not allocate target machine");

		// Override default to generate verbose assembly.
		machine->setAsmVerbosityDefault(true);

		platform = new TargetPlatform(target, machine, triple.getTriple());
		platforms[KERNELGEN_RUNMODE_CUDA].reset(platform);
		return platform;
	}
	default:
		THROW("Unknown runmode " << RUNMODE, RUNMODE);
		break;
	}
}
}
}
