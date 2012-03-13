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
#include "llvm/LLVMContext.h"
#include "llvm/Module.h"
#include "llvm/PassManager.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/IRReader.h"
#include "llvm/Transforms/IPO/PassManagerBuilder.h"
#include "llvm/Support/raw_os_ostream.h"
#include "llvm/Support/TypeBuilder.h"
#include "llvm/Transforms/IPO.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/Transforms/Scalar.h"

#include "BranchedLoopExtractor.h"

#include <gmp.h>
#include <iostream>
 
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

Pass* createFixPointersPass();
Pass* createMoveUpCastsPass();

extern string dragonegg_result;

// The parent gcc instance already compiled the source code.
// Here we need to compile the same source code to LLVM IR and
// attach it to the assembly as extra string global variables.
extern "C" void callback (void*, void*)
{
	//
	// 1) Append "always inline" attribute to all existing functions.
	//
	LLVMContext &context = getGlobalContext();
	SMDiagnostic diag;
	MemoryBuffer* buffer1 = MemoryBuffer::getMemBuffer(dragonegg_result);
	auto_ptr<Module> m;
	m.reset(ParseIR(buffer1, diag, context));
	for (Module::iterator f = m.get()->begin(), fe = m.get()->end(); f != fe; f++)
	{
		Function* func = f;
		if (func->isDeclaration()) continue;

		const AttrListPtr attr = func->getAttributes();
		const AttrListPtr attr_new = attr.addAttr(~0U, Attribute::AlwaysInline);
		func->setAttributes(attr_new);
	}
	
	//
	// 2) Inline calls and extract loops into new functions.
	//
	{
		std::vector<CallInst*> LoopFuctionCalls;
		PassManager manager;
                manager.add(createFixPointersPass());
		manager.add(createInstructionCombiningPass());
                manager.add(createMoveUpCastsPass());
		manager.add(createInstructionCombiningPass());
		manager.add(createBranchedLoopExtractorPass(LoopFuctionCalls));
		manager.run(*m.get());
	}

	//
	// 3) Apply optimization passes to the resulting common
	// module.
	//
	{
		PassManager manager;
		//manager.add(createLowerSetJmpPass());
		PassManagerBuilder builder;
		builder.Inliner = createFunctionInliningPass();
		builder.OptLevel = 3;
		builder.DisableSimplifyLibCalls = true;
		builder.populateModulePassManager(manager);
		manager.run(*m.get());
	}
	 
	//
	// 4) Embed the resulting module into object file.
	//
	{
		// The name of the symbol to hold LLVM IR source.
		string string_name = "__kernelgen_";
		string_name += main_input_filename;
		
		// The LLVM IR source data.
		string string_data;
		raw_string_ostream stream_data(string_data);
		stream_data << (*m.get());
	
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
		TREE_PUBLIC (var) = 0;
		TREE_STATIC (var) = 1;
		TREE_READONLY (var) = 1;
		DECL_INITIAL (var) = string_val;
		varpool_finalize_decl (var);
	}
}
 
extern "C" int plugin_init (
	plugin_name_args* info, plugin_gcc_version* ver)
{
	// Register callback.
	register_callback (info->base_name, PLUGIN_FINISH_UNIT, &callback, 0);
	
	return 0;
}

