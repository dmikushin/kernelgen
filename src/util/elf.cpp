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

#include "util.h"
#include "elf.h"

#include <fcntl.h>

using namespace std;
using namespace util::elf;
using namespace util::io;

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

void csection::addSymbol(std::string symname, std::string symdata)
{
	if (elf_version(EV_CURRENT) == EV_NONE)
		THROW("ELF library initialization failed: " << elf_errmsg(-1));

	// Create temporary ELF object.
	cfiledesc fd = cfiledesc::mktemp("/tmp/");
	celf e_tmp(NULL, fd.getFilename());
	e_tmp.e = elf_begin(fd.getFileDesc(), ELF_C_WRITE, NULL);
	if (!e_tmp.e) THROW("elf_begin() failed: " << elf_errmsg(-1));
	
	const GElf_Ehdr* ehdr = e->getHeader();
	GElf_Ehdr* ehdr_tmp = (GElf_Ehdr*)gelf_newehdr(e_tmp.e,
		(ehdr->e_machine == EM_X86_64) ? ELFCLASS64 : ELFCLASS32);
	if (!ehdr_tmp) THROW("gelf_newehdr() failed: " << elf_errmsg(-1));
	
	ehdr_tmp->e_machine = ehdr->e_machine;
	ehdr_tmp->e_type = ehdr->e_type;
	ehdr_tmp->e_version = ehdr->e_version;

	// Put together names of ELF sections and name of the target symbol.
	const string sections = "\0.strtab\0.symtab\0" + name + "\0";

	string strings = sections + symname;

	if (ehdr_tmp->e_machine == EM_X86_64)
	{
		// Symbol table size always starts with one
		// undefined symbol.
		Elf64_Sym sym[2];
		sym[0].st_value = 0;
		sym[0].st_size = 0;
		sym[0].st_info = 0;
		sym[0].st_other = 0;
		sym[0].st_shndx = STN_UNDEF;
		sym[0].st_name = 0;
		sym[1].st_value = 0;
		sym[1].st_size = symdata.size();
		sym[1].st_info = ELF32_ST_INFO(STB_GLOBAL, STT_OBJECT);
		sym[1].st_other = 0;
		sym[1].st_shndx = 3;
		sym[1].st_name = sections.size();

		e_tmp.setSymtab64(sym, 2);
	}
	else
	{
		// Symbol table size always starts with one
		// undefined symbol.
		Elf32_Sym sym[2];
		sym[0].st_value = 0;
		sym[0].st_size = 0;
		sym[0].st_info = 0;
		sym[0].st_other = 0;
		sym[0].st_shndx = STN_UNDEF;
		sym[0].st_name = 0;
		sym[1].st_value = 0;
		sym[1].st_size = symdata.size();
		sym[1].st_info = ELF64_ST_INFO(STB_GLOBAL, STT_OBJECT);
		sym[1].st_other = 0;
		sym[1].st_shndx = 3;
		sym[1].st_name = sections.size();

		e_tmp.setSymtab32(sym, 2);
	}

	// Write string table and data.	
	e_tmp.setStrtab(ehdr_tmp, strings);
	e_tmp.setData(symdata);
	
	if (elf_update(e_tmp.e, ELF_C_WRITE) < 0)
		THROW("elf_update() failed: " << elf_errmsg(-1));
	
	// Merge temporary ELF with section's parent
	string merge = "ld";
	list<string> merge_args;
	if (ehdr_tmp->e_machine != EM_X86_64)
		merge_args.push_back("-melf_i386");
	merge_args.push_back("--unresolved-symbols=ignore-all");
	merge_args.push_back("-r");
	merge_args.push_back("-o");
	cfiledesc tmp = cfiledesc::mktemp("/tmp/");
	if (e->ifd->getFilename() != e->ofilename)
		merge_args.push_back(e->ofilename);
	else
		merge_args.push_back(tmp.getFilename());
	merge_args.push_back(e->ifd->getFilename());
	merge_args.push_back(e_tmp.ofilename);

	cout << merge;
	for (list<string>::iterator it = merge_args.begin();
		it != merge_args.end(); it++)
		cout << " " << *it;
	cout << endl;
	execute(merge, merge_args, "", NULL, NULL);

	// Move temporary result in place of original file 
	if (e->ifd->getFilename() == e->ofilename)
	{
		list<string> mv_args;
		mv_args.push_back(tmp.getFilename());
		mv_args.push_back(e->ofilename);
		int status = execute("mv", mv_args, "", NULL, NULL);
		if (status) THROW("Error while moving object");
	}
}

csection::csection() { }

csection::csection(celf* e, Elf_Scn* scn, string name) :
	e(e), scn(scn), name(name)
{
	if (!e) THROW("Invalid ELF handle");
	if (!scn) THROW("Invalid ELF section");
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
	if (lseek(e->ifd->getFileDesc(), position, SEEK_SET) == -1)
		THROW("Cannot set file position to " << position);
	data = new char[size + 1];
	if (read(e->ifd->getFileDesc(), data, size) == -1)
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
	
csymtab::csymtab(const csection* section) : csection(*section), symbols(NULL), nsymbols(0)
{
	Elf_Data* data = elf_getdata(scn, NULL);
	if (!data) THROW("elf_getdata() failed: " << elf_errmsg(-1));

	// Load symbols.
	GElf_Shdr shdr;
	if (gelf_getshdr(scn, &shdr) != &shdr)
		THROW("getshdr() failed: " << elf_errmsg(-1));
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

const csymtab* celf::getSymtab()
{
	if (!opened) open();
	return symtab;
}

const GElf_Ehdr* celf::getHeader()
{
	if (!opened) open();
	return &header;
}

csection* celf::getSection(string name)
{
	if (!opened) open();
	for (vector<csection*>::iterator i = sections.begin(),
		ie = sections.end(); i != ie; i++)
	{
		csection* section = *i;
		if (section->name == name)
			return section;

	}
	THROW("Cannot find section " << name);
	return NULL;
}

void celf::setStrtab(GElf_Ehdr* ehdr, string content)
{
	Elf_Scn* scn = elf_newscn(e);
	if (!scn) THROW("elf_newscn() failed: " << elf_errmsg(-1));

	Elf_Data* data = elf_newdata(scn);
	if (!data) THROW("elf_newdata() failed: " << elf_errmsg(-1));
	
	data->d_align = 1;
	data->d_buf = (void*)content.c_str();
	data->d_off = 0LL;
	data->d_size = content.size();
	data->d_type = ELF_T_BYTE;
	data->d_version = EV_CURRENT;

	GElf_Shdr *shdr, shdr_buf;
	shdr = gelf_getshdr(scn, &shdr_buf);
	if (!shdr) THROW("gelf_getshdr() failed: " << elf_errmsg(-1));
	
	shdr->sh_name = 1;
	shdr->sh_type = SHT_STRTAB;
	shdr->sh_entsize = 0;

	ehdr->e_shstrndx = elf_ndxscn(scn);

	if (!gelf_update_ehdr(e, ehdr))
		THROW("gelf_update_ehdr() failed: " << elf_errmsg (-1));

	if (!gelf_update_shdr(scn, shdr))
		THROW("gelf_update_shdr() failed: " << elf_errmsg (-1));
}

void celf::setData(string symdata)
{
	Elf_Scn* scn = elf_newscn(e);
	if (!scn) THROW("elf_newscn() failed: " << elf_errmsg(-1));

	Elf_Data* data = elf_newdata(scn);
	if (!data) THROW("elf_newdata() failed: " << elf_errmsg(-1));
	
	data->d_align = 1;
	data->d_buf = (void*)symdata.c_str();
	data->d_off = 0LL;
	data->d_size = symdata.size();
	data->d_type = ELF_T_BYTE;
	data->d_version = EV_CURRENT;

	GElf_Shdr *shdr, shdr_buf;
	shdr = gelf_getshdr(scn, &shdr_buf);
	if (!shdr) THROW("gelf_getshdr() failed: " << elf_errmsg(-1));
	
	shdr->sh_name = 17;
	shdr->sh_type = SHT_PROGBITS;
	shdr->sh_flags = SHF_ALLOC | SHF_WRITE;
	shdr->sh_entsize = 0;

	if (!gelf_update_shdr(scn, shdr))
		THROW("gelf_update_shdr() failed: " << elf_errmsg (-1));
}

template<typename Elf_Sym>
static void setSymtab(Elf* e, Elf_Sym* sym, int count)
{
	Elf_Scn* scn = elf_newscn(e);
	if (!scn) THROW("elf_newscn() failed: " << elf_errmsg(-1));

	GElf_Shdr *shdr, shdr_buf;
	shdr = gelf_getshdr(scn, &shdr_buf);
	if (!shdr) THROW("gelf_getshdr() failed: " << elf_errmsg(-1));

	shdr->sh_name = 9;
	shdr->sh_type = SHT_SYMTAB;
	shdr->sh_entsize = sizeof(Elf_Sym);
	shdr->sh_size = shdr->sh_entsize * count;
	shdr->sh_link = 2;

	Elf_Data* data = elf_newdata(scn);
	if (!data) THROW("elf_newdata() failed: " << elf_errmsg(-1));

	data->d_align = 1;
	data->d_off = 0LL;
	data->d_buf = sym;
	data->d_type = ELF_T_BYTE;
	data->d_size = shdr->sh_entsize * count;
	data->d_version = EV_CURRENT;

	if (!gelf_update_shdr(scn, shdr))
		THROW("gelf_update_shdr() failed: " << elf_errmsg (-1));
}

void celf::setSymtab32(Elf32_Sym* sym, int count)
{
	setSymtab<Elf32_Sym>(e, sym, count);
}

void celf::setSymtab64(Elf64_Sym* sym, int count)
{
	setSymtab<Elf64_Sym>(e, sym, count);
}

void celf::open()
{
	if (!ifd) THROW("Cannot open ELF without input file descriptor specified");

	if (elf_version(EV_CURRENT) == EV_NONE)
		THROW("ELF library initialization failed: " << elf_errmsg(-1));

	// Load elf.
	e = elf_begin(ifd->getFileDesc(), ELF_C_READ, NULL);
	if (!e) THROW("elf_begin() failed: " << elf_errmsg(-1));

	// Get elf header.
	if (!gelf_getehdr(e, &header))
		THROW("gelf_getehdr() failed: " << elf_errmsg(-1));

	// Load index of sections names table.
	size_t shstrndx;
	if (elf_getshdrstrndx(e, &shstrndx))
		THROW("elf_getshdrstrndx() failed: " << elf_errmsg(-1));

	// Count sections.
	int nsections = 0;
	Elf_Scn* scn = NULL;
	while ((scn = elf_nextscn(e, scn)) != NULL)
		nsections++;
	sections_array = new csection[nsections];

	// Load sections.
	scn = NULL;
	for (int i = 0; i < nsections; i++)
	{
		scn = elf_nextscn(e, scn);

		// Get section name.
		GElf_Shdr shdr;
		if (gelf_getshdr(scn, &shdr) != &shdr)
			THROW("getshdr() failed: " << elf_errmsg(-1));
		char* name = NULL;
		if ((name = elf_strptr(e, shstrndx, shdr.sh_name)) == NULL)
			THROW("elf_strptr() failed: " << elf_errmsg(-1));

		csection* section = new (sections_array + i) csection(this, scn, name);
		sections.push_back(section);
		
		// Check if section is symbols table.
		if (!symtab && (shdr.sh_type == SHT_SYMTAB))
			symtab = new csymtab(section);
	}	
	opened = true;
}

celf::celf(string ifilename, string ofilename) :
	opened(false), e(NULL), symtab(NULL), sections_array(NULL),
	ifd(NULL), ofilename(ofilename), managed_fd(true)
{
	if (ifilename != "")
		ifd = new cfiledesc(ifilename, O_RDONLY);
}

celf::celf(cfiledesc* ifd, string ofilename) :
	opened(false), e(NULL), symtab(NULL), sections_array(NULL),
	ifd(ifd), ofilename(ofilename), managed_fd(false)
{
}

celf::~celf()
{
	if (e) elf_end(e);
	if (symtab) delete symtab;
	if (sections_array) delete[] sections_array;
	if (!opened)
	{
		// Delete lazily loaded sections.
		for (vector<csection*>::iterator i = sections.begin(),
			ie = sections.end(); i != ie; i++)
		{
			csection* section = *i;
			delete section;
		}
	}
	if (managed_fd)
	{
		// Delete filedescs if they are managed
		// by celf.
		delete ifd;
	}
}

