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

#include <gmp.h>
#include <iostream>
 
extern "C"
{
#include "gcc-plugin.h"
#include "cp/cp-tree.h"
#include "langhooks.h"
#include "tree-flow.h"
}
 
using namespace std;
 
int plugin_is_GPL_compatible;

extern "C" void callback (void*, void*)
{
	// The global variable content.
	string string_data = "hello!";

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
	DECL_NAME (var) = get_identifier ("hello");
	TREE_PUBLIC (var) = 0;
	TREE_STATIC (var) = 1;
	TREE_READONLY (var) = 1;
	DECL_INITIAL (var) = string_val;
	varpool_finalize_decl (var);
}
 
extern "C" int plugin_init (
	plugin_name_args* info, plugin_gcc_version* ver)
{
	cerr << "starting " << info->base_name << endl;

	// Register callback.
	register_callback (info->base_name, PLUGIN_FINISH_UNIT, &callback, 0);
	
	return 0;
}

