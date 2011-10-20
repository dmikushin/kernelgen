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

#ifndef KERNELGEN_ELF_H
#define KERNELGEN_ELF_H

#include <gelf.h>
#include <iostream>
#include <libelf.h>
#include <regex.h>
#include <string>
#include <unistd.h>
#include <vector>

#define THROW(message) { std::cerr << __FILE__ << ":" << __LINE__ << " " << message << endl; throw; }

namespace util { namespace elf {

// Defines regular expression processor.
class cregex
{
	regex_t regex;
public :
	bool matches(std::string value);

	cregex(std::string pattern, int flags);
	~cregex();
};

class celf;

// Defines ELF section.
class csection
{
protected :
	const celf* e;
	Elf_Scn* scn;
	GElf_Shdr shdr;

	csection();
	csection(const celf* e, Elf_Scn* scn);
public :
	const GElf_Shdr& getHeader() const;

	friend class celf;
};

class csymtab;

// Defines ELF symbol.
class csymbol
{
	const celf* e;
	std::string name;
	char* data;
	size_t size;
	int shndx;
	bool data_loaded;
	
	csymbol();
public :
	const std::string& getName() const;
	const char* getData();
	size_t getSize() const;

	csymbol(const celf* e, std::string name,
		char* data, size_t size, int shndx);
	
	~csymbol();
	
	friend class csymtab;
};

// Defines ELF symbol table section.
class csymtab : public csection
{
	int nsymbols;
	csymbol* symbols;
public :

	// Find symbols names by the specified pattern.
	std::vector<csymbol*> find(cregex& regex) const;

	csymtab(csection& section);
	~csymtab();
};

// Defines ELF image section.
class celf
{
	int fd;
	Elf* e;
	std::vector<csection> sections;
	csymtab* symtab;
public :
	const csymtab* getSymtab() const;

	celf(int fd, int flag);
	~celf();
	
	friend class csymtab;
	friend class csymbol;
};

} } // namespace

#endif // KERNELGEN_ELF_H

