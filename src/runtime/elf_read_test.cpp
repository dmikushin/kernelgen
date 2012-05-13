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
 * 
 * Created with help of:
 * 	"libelf by Example" by Joseph Koshy
 * 	"Executable and Linkable Format (ELF)"
 * 	people on #gcc@irc.oftc.net
 */

#include "elf.h"

#include <iostream>

using namespace util::elf;
using namespace std;

int main(int argc, char **argv)
{
	if (argc != 3)
	{
		cout << "Purpose: read value of the specified symbol from entire ELF image" << endl;
		cout << "Usage: " << argv[0] << " <filename> <symname>" << endl;
		return 0;
	}
	
	string filename = argv[1];
	string symname = argv[2];

	celf e(filename, "");
	cregex regex("^" + symname + "$", REG_EXTENDED | REG_NOSUB);
	vector<csymbol*> symbols = e.getSymtab()->find(regex);
	if (!symbols.size())
	{
		cerr << "Cannot find symbol " << symname << " in " << filename << endl;
		return 1;
	}
	
	csymbol* symbol = *symbols.begin();
	const char* data = symbol->getData();

	cout << "Found symbol " << symname << " in " << filename << ":" << endl;
	cout << data << endl;

	return 0;
}

