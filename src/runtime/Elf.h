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

#include "io.h"
#include <gelf.h>
#include <libelf.h>
#include <regex.h>
#include <string>
#include <vector>

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
	celf* e;
	Elf_Scn* scn;
	std::string name;

	csection();
	csection(celf* e, Elf_Scn* scn, std::string name);
public :
	void addSymbol(std::string name, const char* data, size_t length);

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
	bool data_loaded, data_allocated;
	
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

	csymtab(const csection* section);
	~csymtab();
};

// Defines ELF image section.
class celf
{
	// Fields for the underlying input and output
	// file descriptors.
	util::io::cfiledesc *ifd;
	bool managed_fd;
	std::string ofilename;

	Elf* e;
	csection* sections_array;
	std::vector<csection*> sections;
	csymtab* symtab;
	GElf_Ehdr header;
	bool opened;
	
	void open();
public :
	const csymtab* getSymtab();
	const GElf_Ehdr* getHeader();
	csection* getSection(std::string name);

	void setSymtab32(Elf32_Sym* sym, int count);
	void setSymtab64(Elf64_Sym* sym, int count);

	void setStrtab(GElf_Ehdr* ehdr, const char* content, size_t length);
	void setData(const char* content, size_t length);

	celf(std::string ifilename, std::string ofilename);
	celf(util::io::cfiledesc* ifd, std::string ofilename);
	~celf();
	
	friend class csection;
	friend class csymtab;
	friend class csymbol;
};

} } // namespace

#endif // KERNELGEN_ELF_H

