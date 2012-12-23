//===- kernelgen-plugin.cpp - KernelGen GCC plugin ------------------------===//
//
// KernelGen -- A prototype of LLVM-based auto-parallelizing Fortran/C
// compiler for NVIDIA GPUs, targeting numerical modeling code.
//
// This file is distributed under the GNU Lesser General Public License.
//
// Copyright (C) 2011, 2012 Dmitry Mikushin, dmitry@kernelgen.org
//
// This program is free software; you can redistribute it and/or
// modify it under the terms of the GNU Lesser General Public License
// as published by the Free Software Foundation; either version 2
// of the License, or (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program; if not, write to the Free Software
// Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.
//
//===----------------------------------------------------------------------===//
//
// This file interfaces KernelGen compile-time passes with GCC fronend, using
// GCC plugin API.
//
//===----------------------------------------------------------------------===//

#include "llvm/Constants.h"
#include "llvm/Instructions.h"
#include "llvm/Linker.h"
#include "llvm/LLVMContext.h"
#include "llvm/Module.h"
#include "llvm/PassManager.h"
#include "llvm/Analysis/Verifier.h"
#include "llvm/Bitcode/ReaderWriter.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/FormattedStream.h"
#include "llvm/Support/Host.h"
#include "llvm/Support/IRReader.h"
#include "llvm/Support/ManagedStatic.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/PluginLoader.h"
#include "llvm/Support/PrettyStackTrace.h"
#include "llvm/Support/Program.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/Signals.h"
#include "llvm/Support/TargetRegistry.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/ToolOutputFile.h"
#include "llvm/Support/TypeBuilder.h"
#include "llvm/Target/TargetData.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetOptions.h"
#include "llvm/Transforms/IPO.h"
#include "llvm/Transforms/IPO/PassManagerBuilder.h"
#include "llvm/Transforms/Scalar.h"
#include "llvm/Transforms/Utils/Cloning.h"
#include "llvm/LinkAllPasses.h"

#include <gmp.h>
#include <iostream>
#include <list>
#include <vector>

#include "KernelGen.h"
#include "BranchedLoopExtractor.h"
#include "Timer.h"
#include "TrackedPassManager.h"
 
extern "C"
{
	#include "gcc-plugin.h"
	#include "cp/cp-tree.h"
	#include "langhooks.h"
	#include "tree-flow.h"
}

using namespace kernelgen;
using namespace kernelgen::utils;
using namespace llvm;
using namespace std;

TimingInfo TI("libkernelgen-ct.so");

int plugin_is_GPL_compatible;

// Information about this plugin.
// Users can access this using "gcc --help -v".
static struct plugin_info info = {
	KERNELGEN_VERSION,
	NULL
};

static int verbose = 0;

Pass* createFixPointersPass();
Pass* createMoveUpCastsPass();

extern string dragonegg_result;

extern cl::opt<bool> EnablePRE;
extern cl::opt<bool> EnableLoadPRE;
extern cl::opt<bool> DisableLoadsDeletion;
extern cl::opt<bool> DisablePromotion;

static void addKernelgenPasses(const PassManagerBuilder &Builder, PassManagerBase &PM)
{
	PM.add(createFixPointersPass());
	PM.add(createInstructionCombiningPass());
	PM.add(createMoveUpCastsPass());
	PM.add(createInstructionCombiningPass());
	PM.add(createBasicAliasAnalysisPass());
	PM.add(createGVNPass()); 
	//PM.add(createEarlyCSEPass());
	PM.add(createBranchedLoopExtractorPass());
	PM.add(createVerifierPass());
}

// A fallback function to be called in case kernelgen-enabled
// compilation process fails by some reason. This function
// must be defined by the the gcc fronend.
extern "C" void fallback(void*);

// The parent gcc instance already compiled the source code.
// Here we need to compile the same source code to LLVM IR and
// attach it to the assembly as extra string global variables.
extern "C" void callback (void*, void*)
{
	PassTracker* tracker = new PassTracker(main_input_filename, &fallback, NULL);

	LLVMContext &context = getGlobalContext();
	SMDiagnostic diag;
	MemoryBuffer* buffer1 = MemoryBuffer::getMemBuffer(dragonegg_result);
	Module* m = ParseIR(buffer1, diag, context);

	//
	// 1) Extract loops into new functions. Apply some optimization
	// passes to the resulting module.
	//
	{
		TimeRegion TCompile(TI.getTimer("Loops extraction"));
		{
			PassManager manager;
			manager.add(new TargetData(m));
			manager.add(createFixPointersPass());
			manager.add(createInstructionCombiningPass());
			manager.add(createMoveUpCastsPass());
			manager.add(createInstructionCombiningPass());
			manager.add(createEarlyCSEPass());
			manager.add(createCFGSimplificationPass());
			manager.run(*m);
		}
		{
			EnableLoadPRE.setValue(false);
			DisableLoadsDeletion.setValue(true);
			DisablePromotion.setValue(true);
			PassManager manager;
			manager.add(new TargetData(m));
			manager.add(createBasicAliasAnalysisPass());
			manager.add(createLICMPass());
			manager.add(createGVNPass());
			manager.run(*m);

		}
		{
			PassManager manager;
			manager.add(createBranchedLoopExtractorPass());
			manager.add(createCFGSimplificationPass());
			manager.run(*m);
		}
	}
		
	verifyModule(*m);
	VERBOSE(Verbose::Sources << *m << Verbose::Default);

	//
	// 2) Embed the resulting module into object file.
	//
	{
		TimeRegion TCompile(TI.getTimer("Embedding LLVM IR into object"));

		// The name of the symbol to hold LLVM IR source.
		string string_name = "__kernelgen_";
		string_name += main_input_filename;
		
		// The LLVM IR source data.
		SmallVector<char, 128> moduleBitcode;
		raw_svector_ostream moduleBitcodeStream(moduleBitcode);
		WriteBitcodeToFile(m, moduleBitcodeStream);
		moduleBitcodeStream.flush();

		// Create the constant string with the specified content.
		tree index_type = build_index_type(size_int(moduleBitcode.size()));
		tree const_char_type = build_qualified_type(
			unsigned_char_type_node, TYPE_QUAL_CONST);
		tree string_type = build_array_type(const_char_type, index_type);
		string_type = build_variant_type_copy(string_type);
		TYPE_STRING_FLAG(string_type) = 1;
		tree string_val = build_string(moduleBitcode.size(), moduleBitcode.data());
		TREE_TYPE(string_val) = string_type;

		// Create a global string variable and assign it with a
		// previously created constant value.
		tree var = create_tmp_var_raw(string_type, NULL);
		char* tmpname = tmpnam(NULL) + strlen(P_tmpdir);
		if (*tmpname == '/') tmpname++;
		DECL_NAME (var) = get_identifier (tmpname);
		DECL_SECTION_NAME (var) = build_string (11, ".kernelgen");
		TREE_PUBLIC (var) = 1;
		TREE_STATIC (var) = 1;
		TREE_READONLY (var) = 1;
		DECL_INITIAL (var) = string_val;
		varpool_finalize_decl (var);
	}

	delete tracker;

	llvm_shutdown();
}
 
extern "C" int plugin_init (
	plugin_name_args* info, plugin_gcc_version* ver)
{
	// Turn on time stats, if requested on gcc side.
	if (time_report || !quiet_flag || flag_detailed_statistics)
		llvm::TimePassesIsEnabled = true;

	// Register callbacks.
	register_callback(info->base_name, PLUGIN_INFO, NULL, &info);
	register_callback(info->base_name, PLUGIN_FINISH_UNIT, &callback, 0);
	
	return 0;
}
