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

using namespace std;
using namespace util::elf;

bool cregex::matches(string value)
{
	if (!regexec(&regex, value.c_str(), (size_t) 0, NULL, 0)) return true;
	return false;
}

cregex::cregex(string pattern, int flags)
{
	// Compile regular expression out of the specified
	// string pattern.
	if (regcomp(&regex, pattern.c_str(), flags))
		THROW("Invalid regular expression " << pattern);
}

cregex::~cregex()
{
	regfree(&regex);
}

const GElf_Shdr& csection::getHeader() const
{
	return shdr;
}

csection::csection(const celf* e, Elf_Scn* scn) : e(e), scn(scn)
{
	if (!scn) THROW("Invalid ELF section");

	if (!gelf_getshdr(scn, &shdr))
		THROW("gelf_getshdr() failed: " << elf_errmsg(-1));
}

csymbol::csymbol() { }

const std::string& csymbol::getName() const { return name; }

const char* csymbol::getData()
{
	if (data_loaded) return data;

	// If object is not fully linked, address value
	// could be representing offset, not absolute address.
	// TODO: set condition on when it happens
	
	// Navigate to section pointed by symbol.
	GElf_Shdr shdr;
	Elf_Scn* scn = NULL;
	for (int i = 0; (i < shndx) &&
		((scn = elf_nextscn(e->e, scn)) != NULL); i++)
	{
		if (!gelf_getshdr(scn, &shdr))
			THROW("gelf_getshdr() failed: " << elf_errmsg(-1));
	}
	if (!scn) THROW("Invalid section index: symbol.st_shndx");
	
	// Load actual data from file.
	size_t position = shdr.sh_offset + (size_t)data;
	if (lseek(e->fd, position, SEEK_SET) == -1)
		THROW("Cannot set file position to " << position);
	data = new char[size + 1];
	if (read(e->fd, data, size) == -1)
		THROW("Cannot read section data from file");
	data[size] = '\0';

	data_loaded = true;
	return data;
}

size_t csymbol::getSize() const { return size; }
	
csymbol::csymbol(const celf* e, std::string name,
	char* data, size_t size, int shndx) :
	e(e), name(name), data(data), size(size),
	shndx(shndx), data_loaded(false) { }

csymbol::~csymbol()
{
	if (data_loaded) delete[] data;
}

// Find symbols names by the specified pattern.
vector<csymbol*> csymtab::find(cregex& regex) const
{
	vector<csymbol*> result;
	for (int i = 0; i < nsymbols; i++)
	{
		csymbol* symbol = symbols + i;
		const string& name = symbol->getName();
		if (regex.matches(name))
			result.push_back(symbol);
	}
	return result;
}
	
csymtab::csymtab(csection& section) : csection(section), symbols(NULL), nsymbols(0)
{
	Elf_Data* data = elf_getdata(scn, NULL);
	if (!data) THROW("elf_getdata() failed: " << elf_errmsg(-1));

	// Load symbols.
	if (shdr.sh_size && !shdr.sh_entsize)
		THROW("Cannot get the number of symbols");
	if (shdr.sh_size)
		nsymbols = shdr.sh_size / shdr.sh_entsize;
	symbols = new csymbol[nsymbols];
	for (int i = 0; i < nsymbols; i++)
	{
		GElf_Sym symbol;
		if (!gelf_getsym(data, i, &symbol))
			THROW("gelf_getsym() failed: " << elf_errmsg(-1));
		char* name = elf_strptr(
			e->e, shdr.sh_link, symbol.st_name);
		if (!name)
			THROW("elf_strptr() failed: " << elf_errmsg(-1));

		char* data = (char*)(size_t)symbol.st_value;
		size_t size = symbol.st_size;
		int shndx = symbol.st_shndx;

		new (symbols + i) csymbol(e, name, data, size, shndx);
	}
}

csymtab::~csymtab()
{
	delete[] symbols;
}

const csymtab* celf::getSymtab() const { return symtab; }

celf::celf(int fd, int flag) : e(NULL), fd(fd), symtab(NULL)
{
	if (fd < 0)
		THROW("Invalid file descriptor");

	if (elf_version(EV_CURRENT) == EV_NONE)
		THROW("ELF library initialization failed: " << elf_errmsg(-1));

	// Load elf.
	e = elf_begin(fd, ELF_C_READ, NULL);
	if (!e)
		THROW("elf_begin() failed: " << elf_errmsg(-1));

	// Load sections.
	Elf_Scn* scn = NULL;
	while ((scn = elf_nextscn(e, scn)) != NULL)
		sections.push_back(csection(this, scn));

	// Search and load symbols table.
	for (vector<csection>::iterator i = sections.begin(),
		ie = sections.end(); i != ie; i++)
	{
		csection& section = *i;
		const GElf_Shdr& shdr = section.getHeader();
		if (shdr.sh_type == SHT_SYMTAB)
		{			
			symtab = new csymtab(section);
			break;
		}
	}
}

celf::~celf()
{
	if (e) elf_end(e);
	if (symtab) delete symtab;
}

