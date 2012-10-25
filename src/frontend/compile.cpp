/*
 * KernelGen - the LLVM-based compiler with GPU kernels generation over C backend.
 *
 * Copyright (c) 2012 Dmitry Mikushin
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

#include "llvm/Constants.h"
#include "llvm/Instructions.h"
#include "llvm/Linker.h"
#include "llvm/LLVMContext.h"
#include "llvm/Module.h"
#include "llvm/PassManager.h"
#include "llvm/Analysis/Verifier.h"
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

#include "BranchedLoopExtractor.h"
#include "TrackedPassManager.h"
 
extern "C"
{
	#include "gcc-plugin.h"
	#include "cp/cp-tree.h"
	#include "langhooks.h"
	#include "tree-flow.h"
}

using namespace llvm;
using namespace std;

int plugin_is_GPL_compatible;

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

	verifyModule(*m);
	if (verbose) m->dump();

	//
	// 2) Embed the resulting module into object file.
	//
	{
		// The name of the symbol to hold LLVM IR source.
		string string_name = "__kernelgen_";
		string_name += main_input_filename;
		
		// The LLVM IR source data.
		string string_data;
		raw_string_ostream stream_data(string_data);
		stream_data << (*m);
	
		// Create the constant string with the specified content.
		tree index_type = build_index_type(size_int(string_data.length()));
		tree const_char_type = build_qualified_type(
			unsigned_char_type_node, TYPE_QUAL_CONST);
		tree string_type = build_array_type(const_char_type, index_type);
		string_type = build_variant_type_copy(string_type);
		TYPE_STRING_FLAG(string_type) = 1;
		tree string_val = build_string(string_data.length(), string_data.data());
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

	PluginLoader loader;
	loader.operator =("libkernelgen-opt.so");

	// Register callback.
	register_callback (info->base_name, PLUGIN_FINISH_UNIT, &callback, 0);
	
	// Enable or disable verbose output.
	char* cverbose = getenv("kernelgen_verbose");
	if (cverbose) verbose = atoi(cverbose);

	return 0;
}

