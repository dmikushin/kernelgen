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
 */

#include "Cuda.h"

#include <algorithm>
#include <cstddef>
#include <cstdio>
#include <cstring>
#include <fcntl.h>
#include <gelf.h>
#include <map>
#include <string>
#include <vector>
#include <unistd.h>

// The list of first 5 predefined sections.
#define NULL_SECTION_INDEX      0 // NULL
#define SHSTRTAB_SECTION_INDEX  1 // .shstrtab
#define STRTAB_SECTION_INDEX    2 // .strtab
#define SYMTAB_SECTION_INDEX    3 // .symtab
#define NV_INFO_SECTION_INDEX   4 // .nv.info

using namespace std;

struct Elf_Datas
{
	int old_index, new_index;

	string name, info;

	GElf_Shdr shdr;
	Elf_Data *data;
	size_t shname_off;

	Elf_Datas() : old_index(-1) { }
};

struct Elf_Symbols
{
	int old_index[2], new_index;

	GElf_Sym sym[2];
	size_t strname_off;

	Elf_Symbols()
	{
		old_index[0] = -1;
		old_index[1] = -1;
	}
};

struct NVInfo
{
	unsigned int Ident1; // 0x00081204
	unsigned int GlobalSymbolIndex1;
	unsigned int MinStackSize;

	unsigned int Ident2; // 0x00081104
	unsigned int GlobalSymbolIndex2;
	unsigned int MinFrameSize;
};

struct NVInfo_Ext
{
	string symbol1;
	string symbol2;

	struct NVInfo nvinfo;
};

struct Elf_PHeader
{
	string name;

	GElf_Phdr phdr;
};

static void parse_elf_sections(int elfclass,
	const char* cubin, Elf* e, int ielf, int shstrndx,
	map<string, Elf_Datas>& sections, map<string, Elf_Symbols>& symbols,
	map<string, int>& strings,
	size_t& szdata, size_t& sznames, size_t& szstrings, size_t& nstrings,
	vector<string>& relocations, vector<NVInfo_Ext>& nvinfos)
{
	// First, locate and handle the symbol table.
	Elf_Scn* scn = elf_nextscn(e, NULL);
	int strndx;
	Elf_Data* symtab_data = NULL;
	for (int i = 1; scn != NULL; scn = elf_nextscn(e, scn), i++)
	{
		// Get section header.
		GElf_Shdr shdr;
		if (!gelf_getshdr(scn, &shdr)) {
			fprintf(stderr, "gelf_getshdr() failed for %s: %s\n",
				cubin, elf_errmsg(-1));
			throw;
		}

		// If section is not a symbol table:
		if (shdr.sh_type != SHT_SYMTAB) continue;

		// Load symbols.
		symtab_data = elf_getdata(scn, NULL);
		if (!symtab_data)
		{
			fprintf(stderr, "Expected %s data section for %s\n",
				".symtab", cubin);
			throw;
		}
		if (shdr.sh_size && !shdr.sh_entsize)
			fprintf(stderr, "Cannot get the number of symbols for %s\n",
				cubin);
		int nsymbols = 0;
		if (shdr.sh_size)
			nsymbols = shdr.sh_size / shdr.sh_entsize;
		strndx = shdr.sh_link;
		for (int i = 0; i < nsymbols; i++)
		{
			GElf_Sym sym;
			if (!gelf_getsym(symtab_data, i, &sym))
			{
				fprintf(stderr, "gelf_getsym() failed for %s: %s\n",
					cubin, elf_errmsg(-1));
				throw;
			}
			char* name = elf_strptr(e, strndx, sym.st_name);
			if (!name)
			{
				fprintf(stderr, "Cannot get the name of %d-th symbol for %s: %s\n",
					i, cubin, elf_errmsg(-1));
				throw;
			}
			//printf("symbol: \"%s\"\n", name);
			Elf_Symbols& symbol = symbols[name];
			if (symbol.old_index[ielf] != -1)
			{
				fprintf(stderr, "Duplicate symbols are not allowed for %s: \"%s\"\n",
					cubin, name);
				throw;
			}
			symbol.sym[ielf] = sym;
			symbol.old_index[ielf] = i;
		}
	}

	// Handle all other tables.
	scn = elf_nextscn(e, NULL);
	for (int i = 1; scn != NULL; scn = elf_nextscn(e, scn), i++)
	{
		// Get section header.
		GElf_Shdr shdr;
		if (!gelf_getshdr(scn, &shdr)) {
			fprintf(stderr, "gelf_getshdr() failed for %s: %s\n",
				cubin, elf_errmsg(-1));
			throw;
		}

		// Get name.
		char* name = NULL;
		if ((name = elf_strptr(e, shstrndx, shdr.sh_name)) == NULL)
		{
			fprintf(stderr, "Cannot get the name of section %d of %s\n",
				i, cubin);
			throw;
		}

		// Decode .nv.info section.
		if (!strcmp(name, ".nv.info"))
		{
			Elf_Data* data = elf_getdata(scn, NULL);
			if (!data)
			{
				fprintf(stderr, "Expected %s data section for %s\n",
					".nv.info", cubin);
				throw;
			}

			int ninfos = shdr.sh_size / sizeof(NVInfo);
			int offset = nvinfos.size();
			nvinfos.resize(offset + ninfos);
			NVInfo* info = (NVInfo*)data->d_buf;
			for (int k = 0; k < ninfos; k++)
			{
				NVInfo& nvinfo = info[k];

				// Find symbol name by first global index.
				GElf_Sym sym;
				if (!gelf_getsym(symtab_data, nvinfo.GlobalSymbolIndex1, &sym))
				{
					fprintf(stderr, "gelf_getsym() failed for %s: %s\n",
						cubin, elf_errmsg(-1));
					throw;
				}
				char* name1 = elf_strptr(e, strndx, sym.st_name);
				if (!name)
				{
					fprintf(stderr, "Cannot get the name of %d-th symbol for %s: %s\n",
						i, cubin, elf_errmsg(-1));
					throw;
				}

				// Find symbol name by second global index.
				if (!gelf_getsym(symtab_data, nvinfo.GlobalSymbolIndex2, &sym))
				{
					fprintf(stderr, "gelf_getsym() failed for %s: %s\n",
						cubin, elf_errmsg(-1));
					throw;
				}
				char* name2 = elf_strptr(e, strndx, sym.st_name);
				if (!name)
				{
					fprintf(stderr, "Cannot get the name of %d-th symbol for %s: %s\n",
						i, cubin, elf_errmsg(-1));
					throw;
				}

				// Store nvinfo record.
				nvinfos[offset + k].nvinfo = nvinfo;
				nvinfos[offset + k].symbol1 = name1;
				nvinfos[offset + k].symbol2 = name2;

				/*printf("nvinfo: ident1 = 0x%x\n", nvinfo.Ident1);
				printf("nvinfo: global symbol1 = %s\n", name1);
				printf("nvinfo: min stack size = %d\n", nvinfo.MinStackSize);
				printf("nvinfo: ident2 = 0x%x\n", nvinfo.Ident2);
				printf("nvinfo: global symbol2 = %s\n", name2);
				printf("nvinfo: min frame size = %d\n", nvinfo.MinFrameSize);*/
			}
			continue;
		}

		// More special handling for certain types of sections:
		switch (shdr.sh_type)
		{
		case SHT_REL:

			// If section is a relocation table:
			{
				if (!symtab_data)
				{
					fprintf(stderr, "Cannot parse relocations, since .symtab is not found for %s\n",
						cubin);
					throw;
				}

				// Load relocations.
				Elf_Data* data = elf_getdata(scn, NULL);
				if (!data)
				{
					fprintf(stderr, "Expected %s data section for %s\n",
						".symtab", cubin);
					throw;
				}
				if (shdr.sh_size && !shdr.sh_entsize)
					fprintf(stderr, "Cannot get the number of symbols for %s\n",
						cubin);
				int nrelocs = 0;
				if (shdr.sh_entsize)
					nrelocs = shdr.sh_size / shdr.sh_entsize;
				size_t offset = relocations.size();
				relocations.resize(offset + nrelocs);
				for (int k = 0; k < nrelocs; k++)
				{
					GElf_Rel rel;
					if (!gelf_getrel(data, k, &rel))
					{
						fprintf(stderr, "gelf_getrel() failed for %s: %s\n",
							cubin, elf_errmsg(-1));
						throw;
					}
					int i = 0;
					switch (elfclass)
					{
					case ELFCLASS32:
						i = ELF32_R_SYM(rel.r_info);
						break;
					case ELFCLASS64:
						i = ELF64_R_SYM(rel.r_info);
						break;
					}

					// Find symbol name by its index.
					GElf_Sym sym;
					if (!gelf_getsym(symtab_data, i, &sym))
					{
						fprintf(stderr, "gelf_getsym() failed for %s: %s\n",
							cubin, elf_errmsg(-1));
						throw;
					}
					char* name = elf_strptr(e, strndx, sym.st_name);
					if (!name)
					{
						fprintf(stderr, "Cannot get the name of %d-th symbol for %s: %s\n",
							i, cubin, elf_errmsg(-1));
						throw;
					}
					relocations[k + offset] = name;
					//printf("reloc: \"%s\"\n", name);
				}
			}
			break;

		case SHT_SYMTAB:

			{
				// Skip symbol table, which is already handled.
				continue;
			}
			break;

		case SHT_STRTAB:

			// If section is a string table:
			{
				// Deal with sections names separately.
				if (!strcmp(name, ".shstrtab")) continue;

				if (!strcmp(name, ".strtab"))
				{
					Elf_Data* data = elf_getdata(scn, NULL);
					if (!data)
					{
						fprintf(stderr, "Expected %s data section for %s\n",
							name, cubin);
						throw;
					}
					int i = 0;
					do
					{
						char* ptr = (char*)data->d_buf + i;
						string str(ptr);
						size_t size = strlen(ptr) + 1;
						if (strings.find(str) == strings.end())
						{
							strings[str] = nstrings++;
							szstrings += size;
						}
						i += size;
					}
					while (i < data->d_size);

					continue;
				}

				fprintf(stderr, "Unknown string table section for %s: %s\n",
					cubin, name);
				throw;
			}
			break;
		}

		// Get data.
		Elf_Data* data = elf_getdata(scn, NULL);

		// Write name, section and data.
		Elf_Datas& datas = sections[name];
		if (datas.old_index != -1)
		{
			fprintf(stderr, "Duplicate sections are not allowed for %s: \"%s\"\n",
				cubin, name);
			throw;
		}
		datas.name = name;
		datas.shdr = shdr;
		datas.data = data;
		datas.old_index = i;

		// New index is shifted by the number of heading special sections.
		datas.new_index = sections.size() + NV_INFO_SECTION_INDEX;

		// Remap section info value.
		datas.info = "";
		if (strncmp(name, ".text.", strlen(".text.")))
		{
			// Get section by info index.
			Elf_Scn* scn = NULL;
			if ((scn = elf_getscn(e, shdr.sh_info)) == NULL)
			{
				fprintf(stderr, "Cannot find section at index %d\n",
					shdr.sh_info);
				throw;
			}

			// Get section header.
			GElf_Shdr shdr;
			if (!gelf_getshdr(scn, &shdr)) {
				fprintf(stderr, "gelf_getshdr() failed for %s: %s\n",
					cubin, elf_errmsg(-1));
				throw;
			}

			// Get section name.
			char* name = elf_strptr(e, shstrndx, shdr.sh_name);
			if (!name)
			{
				fprintf(stderr, "Cannot get the name of %d-th symbol for %s: %s\n",
					i, cubin, elf_errmsg(-1));
				throw;
			}

			datas.info = name;
			//printf("info: %s\n", name);
		}

		// Account data size in total data size.
		szdata += shdr.sh_size;

		// Account section name size.
		sznames += strlen(name) + 1;
	}
}

static void parse_elf_program_headers(int elfclass,
	const char* cubin, Elf* e, GElf_Ehdr& ehdr,
	map<string, Elf_Datas>& sections, vector<Elf_PHeader>& pheaders)
{
	// Build a temporary "offset"-"section name" map.
	map<size_t, string> offsets;
	for (map<string, Elf_Datas>::iterator i = sections.begin(), ie = sections.end(); i != ie; i++)
	{
		offsets[i->second.shdr.sh_offset] = i->first;
		//printf("%zu -> %s\n", (size_t)i->second.shdr.sh_offset, i->first.c_str());
	}

	size_t offset = pheaders.size();
	pheaders.resize(offset + ehdr.e_phnum);
	for (int i = 0; i < ehdr.e_phnum; i++)
	{
		GElf_Phdr phdr;
		if (!gelf_getphdr(e, i, &phdr))
		{
			fprintf(stderr, "gelf_getphdr() failed for %s: %s\n",
				cubin, elf_errmsg(-1));
			throw;
		}

		pheaders[offset + i].phdr = phdr;

		if (phdr.p_type != PT_LOAD) continue;

		// TODO: Don't know what to do with 0 offset, that takes place sometimes.
		if (phdr.p_offset == 0) continue;

		// Locate section name by the given offset.
		if (offsets.find(phdr.p_offset) == offsets.end())
		{
			fprintf(stderr, "PHeader refers an offset, where symbol could not be found: %zu\n",
				(size_t)phdr.p_offset);
			throw;
		}
		string& name = offsets[phdr.p_offset];
		pheaders[offset + i].name = name;
		//printf("%s -> %zu\n", name.c_str(), phdr.p_offset);
	}
}

template<typename Elf_Sym>
static void parse_elf_symbols(
	const char* cubin1, Elf* e1, int shstrndx1, const char* cubin2, Elf* e2, int shstrndx2,
	int nsymbols, size_t& szsymbol, map<string, Elf_Symbols>& symbols, vector<char>& symbols_data,
	map<string, Elf_Datas>& sections)
{
	szsymbol = sizeof(Elf_Sym);
	size_t szsymbols = szsymbol * nsymbols;
	symbols_data.resize(szsymbols);
	{
		Elf_Sym* symbols_data_ptr = (Elf_Sym*)&symbols_data[0];
		int isym = 0;
		for (map<string, Elf_Symbols>::iterator i = symbols.begin(), ie = symbols.end(); i != ie; i++, isym++)
		{
			Elf_Symbols& syms = i->second;
			syms.new_index = isym;
			GElf_Sym* sym = NULL;
			char* name = NULL;
			if (syms.old_index[0] != -1)
			{
				sym = &syms.sym[0];
				if (sym->st_shndx == STN_UNDEF)
				{
					memcpy(symbols_data_ptr, sym, szsymbol);
					symbols_data_ptr++;
					continue;
				}

				// Find the name of section symbol was originally linked to.
				Elf_Scn* scn = NULL;
				if ((scn = elf_getscn(e1, sym->st_shndx)) == NULL)
				{
					fprintf(stderr, "Cannot find section at index %d\n",
						sym->st_shndx);
					throw;
				}

				// Get section header.
				GElf_Shdr shdr;
				if (!gelf_getshdr(scn, &shdr)) {
					fprintf(stderr, "gelf_getshdr() failed for %s: %s\n",
						cubin1, elf_errmsg(-1));
					throw;
				}

				// Get name.
				if ((name = elf_strptr(e1, shstrndx1, shdr.sh_name)) == NULL)
				{
					fprintf(stderr, "Cannot get the name of section %d of %s\n",
						sym->st_shndx, cubin1);
					throw;
				}
			}
			else if (syms.old_index[1] != -1)
			{
				sym = &syms.sym[1];
				if (sym->st_shndx == STN_UNDEF)
				{
					memcpy(symbols_data_ptr, sym, szsymbol);
					symbols_data_ptr++;
					continue;
				}

				// Find the name of section symbol was originally linked to.
				Elf_Scn* scn = NULL;
				if ((scn = elf_getscn(e2, sym->st_shndx)) == NULL)
				{
					fprintf(stderr, "Cannot find section at index %d\n",
						sym->st_shndx);
					throw;
				}

				// Get section header.
				GElf_Shdr shdr;
				if (!gelf_getshdr(scn, &shdr)) {
					fprintf(stderr, "gelf_getshdr() failed for %s: %s\n",
						cubin2, elf_errmsg(-1));
					throw;
				}

				// Get name.
				if ((name = elf_strptr(e2, shstrndx2, shdr.sh_name)) == NULL)
				{
					fprintf(stderr, "Cannot get the name of section %d of %s\n",
						sym->st_shndx, cubin2);
					throw;
				}
			}
			else
			{
				fprintf(stderr, "Both symbols containers cannot be empty for symbol \"%s\"\n",
					i->first.c_str());
				throw;
			}
			sym->st_name = syms.strname_off;
			Elf_Datas& datas = sections[name];
			if (datas.old_index == -1)
			{
				printf("section \"%s\" does not exist\n", name);
				throw;
			}
			sym->st_shndx = sections[name].new_index;
			memcpy(symbols_data_ptr, sym, szsymbol);
			symbols_data_ptr++;
			//printf("name: %s\n", name);
		}
	}
}

// Merge two input CUBIN ELF images into single output image.
void kernelgen::bind::cuda::CUBIN::Merge(const char* input1, const char* input2, const char* output)
{
	int fd1 = -1, fd2 = -1, fd3 = -1;
	Elf *e1 = NULL, *e2 = NULL, *e3 = NULL;
	try
	{
		//
		// 0) Setup ELF version.
		//
		if (elf_version(EV_CURRENT) == EV_NONE)
		{
			fprintf(stderr, "Cannot initialize ELF library: %s\n",
				elf_errmsg(-1));
			throw;
		}

		//
		// 1) First, load two input ELF files and one output ELF file.
		//
		if ((fd1 = open(input1, O_RDONLY)) < 0) {
			fprintf(stderr, "Cannot open file %s\n", input1);
			throw;
		}
		if ((fd2 = open(input2, O_RDONLY)) < 0) {
			fprintf(stderr, "Cannot open file %s\n", input2);
			throw;
		}
		if ((fd3 = open(output, O_WRONLY | O_CREAT | O_TRUNC,
			S_IRUSR | S_IWUSR | S_IRGRP | S_IROTH)) < 0)
		{
			fprintf(stderr, "Cannot open file %s\n", output);
			throw;
		}
		if ((e1 = elf_begin(fd1, ELF_C_READ, e1)) == 0) {
			fprintf(stderr, "Cannot read ELF image from %s: %s\n",
				input1, elf_errmsg(-1));
			throw;
		}
		if ((e2 = elf_begin(fd2, ELF_C_READ, e2)) == 0) {
			fprintf(stderr, "Cannot read ELF image from %s: %s\n",
				input2, elf_errmsg(-1));
			throw;
		}
		if ((e3 = elf_begin(fd3, ELF_C_WRITE, e3)) == 0) {
			fprintf(stderr, "Cannot write ELF image from %s: %s\n",
				output, elf_errmsg(-1));
			throw;
		}
		size_t shstrndx1, shstrndx2;
		if (elf_getshdrstrndx(e1, &shstrndx1)) {
			fprintf(stderr, "elf_getshdrstrndx() failed for %s: %s\n",
				input1, elf_errmsg(-1));
			throw;
		}
		if (elf_getshdrstrndx(e2, &shstrndx2)) {
			fprintf(stderr, "elf_getshdrstrndx() failed for %s: %s\n",
				input2, elf_errmsg(-1));
			throw;
		}

		//
		// 2) Compare input ELF executable headers.
		//
		GElf_Ehdr ehdr1, ehdr2;
		if (!gelf_getehdr(e1, &ehdr1))
		{
			fprintf(stderr, "elf_getehdr() failed for %s: %s\n",
				input1, elf_errmsg(-1));
			throw;
		}
		if (!gelf_getehdr(e2, &ehdr2))
		{
			fprintf(stderr, "elf_getehdr() failed for %s: %s\n",
				input2, elf_errmsg(-1));
			throw;
		}
		// Disregarding the 8th identity that is ABI version. Not clear now,
		// when to use 5 and when to use 6.
		ehdr1.e_ident[8] = ehdr2.e_ident[8];
		if (memcmp(&ehdr1.e_ident, &ehdr2.e_ident, sizeof(unsigned char) * EI_NIDENT))
		{
			fprintf(stderr, "Idents of ELF images being merged mismatch\n");
			throw;
		}
		unsigned char elfclass = ((unsigned char*)&ehdr1)[EI_CLASS];
		switch (elfclass)
		{
		case ELFCLASS32 :
			if (memcmp(&ehdr1.e_type, &ehdr2.e_type, sizeof(Elf32_Half)))
			{
				fprintf(stderr, "Types of ELF images being merged mismatch\n");
				throw;
			}
			if (memcmp(&ehdr1.e_machine, &ehdr2.e_machine, sizeof(Elf32_Half)))
			{
				fprintf(stderr, "Machines of ELF images being merged mismatch\n");
				throw;
			}
			if (memcmp(&ehdr1.e_version, &ehdr2.e_version, sizeof(Elf32_Word)))
			{
				fprintf(stderr, "Versions of ELF images being merged mismatch\n");
				throw;
			}
			if (memcmp(&ehdr1.e_flags, &ehdr2.e_flags, sizeof(Elf32_Word)))
			{
				fprintf(stderr, "Flags of ELF images being merged mismatch\n");
				throw;
			}
			break;
		case ELFCLASS64:
			if (memcmp(&ehdr1.e_type, &ehdr2.e_type, sizeof(Elf64_Half)))
			{
				fprintf(stderr, "Types of ELF images being merged mismatch\n");
				throw;
			}
			if (memcmp(&ehdr1.e_machine, &ehdr2.e_machine, sizeof(Elf64_Half)))
			{
				fprintf(stderr, "Machines of ELF images being merged mismatch\n");
				throw;
			}
			if (memcmp(&ehdr1.e_version, &ehdr2.e_version, sizeof(Elf64_Word)))
			{
				fprintf(stderr, "Versions of ELF images being merged mismatch\n");
				throw;
			}
			/*if (memcmp(&ehdr1.e_flags, &ehdr2.e_flags, sizeof(Elf64_Word)))
			{
				fprintf(stderr, "Flags of ELF images being merged mismatch\n");
				throw;
			}*/
			break;
		}

		//
		// 3) Walk through both ELF images, record sections names, headers,
		// their sizes and pointers to data content.
		//
		map<string, Elf_Datas> sections;
		map<string, Elf_Symbols> symbols;
		map<string, int> strings;
		vector<string> relocations;
		vector<NVInfo_Ext> nvinfos;
		size_t szdata = 0, sznames = 0, szstrings = 0, nstrings = 0;
		parse_elf_sections(elfclass, input1, e1, 0, shstrndx1,
			sections, symbols, strings, szdata, sznames,
			szstrings, nstrings, relocations, nvinfos);
		parse_elf_sections(elfclass, input2, e2, 1, shstrndx2,
			sections, symbols, strings, szdata, sznames,
			szstrings, nstrings, relocations, nvinfos);

		//
		// 4) Parse program headers.
		//
		vector<Elf_PHeader> pheaders;
		parse_elf_program_headers(elfclass, input1, e1, ehdr1, sections, pheaders);
		parse_elf_program_headers(elfclass, input2, e2, ehdr2, sections, pheaders);

		//
		// 5) Create output ELF executable header and fill it with a copy
		// of input ELF executable header.
		//
		GElf_Ehdr* ehdr3;
		if ((ehdr3 = (GElf_Ehdr*)gelf_newehdr(e3, elfclass)) == NULL)
		{
			fprintf(stderr, "gelf_newehdr() failed for %s: %s\n",
				output, elf_errmsg(-1));
			throw;
		}
		switch (elfclass)
		{
		case ELFCLASS32:
			memcpy(ehdr3, &ehdr1, offsetof(Elf32_Ehdr, e_entry));
			break;
		case ELFCLASS64:
			memcpy(ehdr3, &ehdr1, offsetof(Elf64_Ehdr, e_entry));
			break;
		default:
			fprintf(stderr, "Unknown ELF class ident: %c\n", elfclass);
			throw;
		}
		ehdr3->e_flags = ehdr1.e_flags;

		//
		// 6) Add sections names string table to new ELF.
		//
		size_t offset = 1 +
			strlen(".shstrtab") + 1 +
			strlen(".strtab") + 1 +
			strlen(".symtab") + 1 +
			strlen(".nv.info") + 1;
		sznames += offset;
		vector<char> sections_data;
		sections_data.resize(sznames);
		char* sections_data_ptr = &sections_data[0];
		memcpy(sections_data_ptr, "", 1);
		sections_data_ptr += 1;
		memcpy(sections_data_ptr, ".shstrtab", strlen(".shstrtab") + 1);
		sections_data_ptr += strlen(".shstrtab") + 1;
		memcpy(sections_data_ptr, ".strtab", strlen(".strtab") + 1);
		sections_data_ptr += strlen(".strtab") + 1;
		memcpy(sections_data_ptr, ".symtab", strlen(".symtab") + 1);
		sections_data_ptr += strlen(".symtab") + 1;
		memcpy(sections_data_ptr, ".nv.info", strlen(".nv.info") + 1);
		sections_data_ptr += strlen(".nv.info") + 1;
		for (map<string, Elf_Datas>::iterator i = sections.begin(), ie = sections.end(); i != ie; i++)
		{
			const string& name = i->first;
			sections[name].shname_off = offset;
			offset += name.size() + 1;

			memcpy(sections_data_ptr, name.c_str(), name.size() + 1);
			sections_data_ptr += name.size() + 1;
		}
		{
			// Create new section.
			Elf_Scn* scn;
			if ((scn = elf_newscn(e3)) == NULL)
			{
				fprintf(stderr, "elf_newscn() failed for %s: %s\n",
					output, elf_errmsg(-1));
				throw;
			}

			// Get section header.
			GElf_Shdr shdr;
			if (!gelf_getshdr(scn, &shdr)) {
				fprintf(stderr, "gelf_getshdr() failed for %s: %s\n",
					output, elf_errmsg(-1));
				throw;
			}

			// Create output joint data section.
			Elf_Data* data;
			if ((data = elf_newdata(scn)) == NULL)
			{
				fprintf(stderr, "elf_newdata() failed for %s: %s\n",
					output, elf_errmsg(-1));
				throw;
			}
			data->d_buf = &sections_data[0];
			data->d_size = sznames;
			data->d_align = 1;
			data->d_off = 0LL;
			data->d_type = ELF_T_BYTE;
			data->d_version = EV_CURRENT;

			// Update section header.
			shdr.sh_name = 1;
			shdr.sh_type = SHT_STRTAB;
			shdr.sh_flags = SHF_STRINGS;
			shdr.sh_entsize = 0;
			shdr.sh_size = sznames;
			if (!gelf_update_shdr(scn, &shdr)) {
				fprintf(stderr, "gelf_update_shdr() failed for %s: %s\n",
					output, elf_errmsg(-1));
				throw;
			}

			// Update executable header.
			ehdr3->e_shstrndx = elf_ndxscn(scn);
			if (!gelf_update_ehdr(e3, ehdr3))
			{
				fprintf(stderr, "gelf_update_ehdr() failed for %s: %s\n",
					output, elf_errmsg (-1));
				throw;
			}
		}

		//
		// 7) Add symbols names string table to new ELF.
		//
		vector<char> strings_data;
		strings_data.resize(sznames + szstrings);
		offset = sznames;
		char* strings_data_ptr = &strings_data[0] + offset;
		memcpy(&strings_data[0], &sections_data[0], sznames);
		for (map<string, int>::iterator i = strings.begin(), ie = strings.end(); i != ie; i++)
		{
			const string& name = i->first;
			if (symbols.find(name) != symbols.end())
				symbols[name].strname_off = offset;

			offset += name.size() + 1;
			memcpy(strings_data_ptr, name.c_str(), name.size() + 1);
			strings_data_ptr += name.size() + 1;
		}
		{
			// Create new section.
			Elf_Scn* scn;
			if ((scn = elf_newscn(e3)) == NULL)
			{
				fprintf(stderr, "elf_newscn() failed for %s: %s\n",
					output, elf_errmsg(-1));
				throw;
			}

			// Get section header.
			GElf_Shdr shdr;
			if (!gelf_getshdr(scn, &shdr)) {
				fprintf(stderr, "gelf_getshdr() failed for %s: %s\n",
					output, elf_errmsg(-1));
				throw;
			}

			// Create output joint data section.
			Elf_Data* data;
			if ((data = elf_newdata(scn)) == NULL)
			{
				fprintf(stderr, "elf_newdata() failed for %s: %s\n",
					output, elf_errmsg(-1));
				throw;
			}
			data->d_buf = &strings_data[0];
			data->d_size = sznames + szstrings;
			data->d_align = 1;
			data->d_off = 0LL;
			data->d_type = ELF_T_BYTE;
			data->d_version = EV_CURRENT;

			// Update section header.
			shdr.sh_name = 1 + strlen(".shstrtab") + 1;
			shdr.sh_type = SHT_STRTAB;
			shdr.sh_flags = SHF_STRINGS;
			shdr.sh_entsize = 0;
			shdr.sh_size = sznames + szstrings;
			if (!gelf_update_shdr(scn, &shdr)) {
				fprintf(stderr, "gelf_update_shdr() failed for %s: %s\n",
					output, elf_errmsg(-1));
				throw;
			}
		}

		//
		// 8) Add symbols table to new ELF.
		//
		size_t szsymbol = 0;
		size_t nsymbols = symbols.size();
		vector<char> symbols_data;
		{
			// Add fictive .nv.info section.
			Elf_Datas& datas = sections[".nv.info"];
			datas.old_index = sections.size();
			datas.new_index = NV_INFO_SECTION_INDEX;
		}
		switch (elfclass)
		{
		case ELFCLASS32:
			parse_elf_symbols<Elf32_Sym>(
				input1, e1, shstrndx1, input2, e2, shstrndx2,
				nsymbols, szsymbol, symbols, symbols_data, sections);
			break;
		case ELFCLASS64:
			parse_elf_symbols<Elf64_Sym>(
				input1, e1, shstrndx1, input2, e2, shstrndx2,
				nsymbols, szsymbol, symbols, symbols_data, sections);
			break;
		default:
			fprintf(stderr, "Unknown ELF class ident: %c\n", elfclass);
			throw;
		}
		{
			// Remove fictive .nv.info section.
			sections.erase(".nv.info");
		}
		size_t szsymbols = nsymbols * szsymbol;
		{
			// Create new section.
			Elf_Scn* scn;
			if ((scn = elf_newscn(e3)) == NULL)
			{
				fprintf(stderr, "elf_newscn() failed for %s: %s\n",
					output, elf_errmsg(-1));
				throw;
			}

			// Get section header.
			GElf_Shdr shdr;
			if (!gelf_getshdr(scn, &shdr)) {
				fprintf(stderr, "gelf_getshdr() failed for %s: %s\n",
					output, elf_errmsg(-1));
				throw;
			}

			// Create output joint data section.
			Elf_Data* data;
			if ((data = elf_newdata(scn)) == NULL)
			{
				fprintf(stderr, "elf_newdata() failed for %s: %s\n",
					output, elf_errmsg(-1));
				throw;
			}
			data->d_buf = &symbols_data[0];
			data->d_size = szsymbols;
			data->d_align = 1;
			data->d_off = 0LL;
			data->d_type = ELF_T_BYTE;
			data->d_version = EV_CURRENT;

			// Update section header.
			shdr.sh_name = 1 + strlen(".shstrtab") + 1 + strlen(".strtab") + 1;
			shdr.sh_type = SHT_SYMTAB;
			shdr.sh_entsize = szsymbol;
			shdr.sh_size = szsymbols;
			shdr.sh_link = STRTAB_SECTION_INDEX;
			shdr.sh_info = STRTAB_SECTION_INDEX;
			if (!gelf_update_shdr(scn, &shdr)) {
				fprintf(stderr, "gelf_update_shdr() failed for %s: %s\n",
					output, elf_errmsg(-1));
				throw;
			}
		}

		//
		// 9) Add .nv.info section.
		//
		size_t ninfos = nvinfos.size();
		size_t szinfos = ninfos * sizeof(NVInfo);
		vector<NVInfo> nvinfos_data;
		nvinfos_data.resize(szinfos);
		NVInfo* nvinfos_data_ptr = &nvinfos_data[0];
		for (int i = 0; i < ninfos; i++)
		{
			NVInfo& nvinfo = nvinfos[i].nvinfo;
			nvinfo.GlobalSymbolIndex1 = symbols[nvinfos[i].symbol1].new_index;
			nvinfo.GlobalSymbolIndex2 = symbols[nvinfos[i].symbol2].new_index;
			*nvinfos_data_ptr++ = nvinfo;
		}
		{
			// Create new section.
			Elf_Scn* scn;
			if ((scn = elf_newscn(e3)) == NULL)
			{
				fprintf(stderr, "elf_newscn() failed for %s: %s\n",
					output, elf_errmsg(-1));
				throw;
			}

			// Get section header.
			GElf_Shdr shdr;
			if (!gelf_getshdr(scn, &shdr)) {
				fprintf(stderr, "gelf_getshdr() failed for %s: %s\n",
					output, elf_errmsg(-1));
				throw;
			}

			// Create output joint data section.
			Elf_Data* data;
			if ((data = elf_newdata(scn)) == NULL)
			{
				fprintf(stderr, "elf_newdata() failed for %s: %s\n",
					output, elf_errmsg(-1));
				throw;
			}
			data->d_buf = (char*)&nvinfos_data[0];
			data->d_size = szinfos;
			data->d_align = 1;
			data->d_off = 0LL;
			data->d_type = ELF_T_BYTE;
			data->d_version = EV_CURRENT;

			// Update section header.
			shdr.sh_name = 1 + strlen(".shstrtab") + 1 + strlen(".strtab") + 1 + strlen(".symtab") + 1;
			shdr.sh_type = SHT_LOPROC;
			shdr.sh_entsize = sizeof(NVInfo);
			shdr.sh_size = szinfos;
			shdr.sh_link = SYMTAB_SECTION_INDEX;
			if (!gelf_update_shdr(scn, &shdr)) {
				fprintf(stderr, "gelf_update_shdr() failed for %s: %s\n",
					output, elf_errmsg(-1));
				throw;
			}
		}

		//
		// 10) Rebase sections from map into array, to maintain the exact ordering.
		//
		vector<Elf_Datas> sections_vector;
		sections_vector.resize(sections.size());
		for (map<string, Elf_Datas>::iterator i = sections.begin(), ie = sections.end(); i != ie; i++)
		{
			Elf_Datas& content = i->second;
			sections_vector[content.new_index - NV_INFO_SECTION_INDEX - 1] = content;
		}

		//
		// 11) Fill the rest of ELF sections.
		//
		vector<char> pool;
		pool.resize(szdata);
		char* poolptr = &pool[0];
		memset(poolptr, 0, szdata);
		size_t relocations_offset = 0;
		for (int i = 0, ie = sections_vector.size(); i != ie; i++)
		{
			Elf_Datas& content = sections_vector[i];
			string& name = content.name;

			// Create new section.
			Elf_Scn* scn;
			if ((scn = elf_newscn(e3)) == NULL)
			{
				fprintf(stderr, "elf_newscn() failed for %s: %s\n",
					output, elf_errmsg(-1));
				throw;
			}

			// Get section header.
			GElf_Shdr shdr;
			if (!gelf_getshdr(scn, &shdr)) {
				fprintf(stderr, "gelf_getshdr() failed for %s: %s\n",
					output, elf_errmsg(-1));
				throw;
			}

			shdr = content.shdr;
			shdr.sh_name = content.shname_off;
			if (content.info != "")
				shdr.sh_info = sections[content.info].new_index;

			// Create section data.
			// Note we create section data even if it previously
			// did not exist. Otherwise, libelf will flush our
			// section size to zero, which is wrong.
			Elf_Data* data;
			if ((data = elf_newdata(scn)) == NULL)
			{
				fprintf(stderr, "elf_newdata() failed for %s: %s\n",
					output, elf_errmsg(-1));
				throw;
			}

			// Fill section data.
			if (content.data && content.data->d_buf)
			{
				*data = *content.data;
				memcpy(poolptr, content.data->d_buf, content.data->d_size);
			}

			data->d_version = EV_CURRENT;

			// Create a data buffer.
			data->d_buf = poolptr;
			data->d_size = shdr.sh_size;
			data->d_align = shdr.sh_addralign;
			if (content.data)
			{
				data->d_type = content.data->d_type;
				data->d_off = content.data->d_off;
				data->d_align = content.data->d_align;
			}

			// If relocations section, update symbols.
			if (shdr.sh_type == SHT_REL)
			{
				int nrelocs = 0;
				if (shdr.sh_entsize)
					nrelocs = shdr.sh_size / shdr.sh_entsize;
				for (int k = 0; k < nrelocs; k++)
				{
					GElf_Rel rel;
					if (!gelf_getrel(data, k, &rel))
					{
						fprintf(stderr, "gelf_getrel() failed for %s: %s\n",
							output, elf_errmsg(-1));
						throw;
					}
					int type;
					switch (elfclass)
					{
					case ELFCLASS32:
						type = ELF32_R_TYPE(rel.r_info);
						rel.r_info = ELF32_R_INFO(symbols[
							relocations[k + relocations_offset]].new_index, type);
						break;
					case ELFCLASS64:
						type = ELF64_R_TYPE(rel.r_info);
						rel.r_info = ELF64_R_INFO(symbols[
							relocations[k + relocations_offset]].new_index, type);
						break;
					}
					if (!gelf_update_rel(data, k, &rel))
					{
						fprintf(stderr, "gelf_update_rel() failed for %s: %s\n",
							output, elf_errmsg(-1));
						throw;
					}
				}
				relocations_offset += nrelocs;
			}

			// Update section header.
			if (!gelf_update_shdr(scn, &shdr)) {
				fprintf(stderr, "gelf_update_shdr() failed for %s: %s\n",
					output, elf_errmsg(-1));
				throw;
			}

			poolptr += shdr.sh_size;
			offset += shdr.sh_size;
		}

		//
		// 12) Add an ELF program header.
		//
		long unsigned int phdr;
		if ((phdr = gelf_newphdr(e3, pheaders.size())) == 0)
		{
			fprintf(stderr, "gelf_newphdr() failed for %s: %s\n",
				output, elf_errmsg(-1));
			throw;
		}

		//
		// 13) Commit changes into the underlying ELF binary.
		//
		if (elf_update(e3, ELF_C_WRITE) == -1) {
			fprintf(stderr, "elf_update() failed for %s: %s\n",
				output, elf_errmsg(-1));
			throw;
		}

		//
		// 14) Fill ELF program header.
		//
		if ((phdr = gelf_newphdr(e3, pheaders.size())) == 0)
		{
			fprintf(stderr, "gelf_newphdr() failed for %s: %s\n",
				output, elf_errmsg(-1));
			throw;
		}
		for (int i = 0, k = 0, ie = pheaders.size(); i != ie; i++)
		{
			// TODO: Don't know what to do with 0 offset, that takes place sometimes.
			if (pheaders[i].phdr.p_offset == 0) continue;

			if (pheaders[i].phdr.p_type != PT_LOAD) continue;

			// Get section index in new ELF.
			string& name = pheaders[i].name;
			int isection = sections[name].new_index;
			Elf_Scn* scn;
			if ((scn = elf_getscn(e3, isection)) == NULL)
			{
				fprintf(stderr, "elf_getscn() failed for %s: %s\n",
					output, elf_errmsg(-1));
				throw;
			}

			// Get section header.
			GElf_Shdr shdr;
			if (!gelf_getshdr(scn, &shdr)) {
				fprintf(stderr, "gelf_getshdr() failed for %s: %s\n",
					output, elf_errmsg(-1));
				throw;
			}

			// Determine the section offset in output ELF
			// and update program header with this value.
			pheaders[i].phdr.p_offset = shdr.sh_offset;
			if (!gelf_update_phdr(e3, k, &pheaders[i].phdr))
			{
				fprintf(stderr, "gelf_update_phdr() failed for %s: %s\n",
					output, elf_errmsg(-1));
				throw;
			}
			k++;
		}

		//
		// 14) Commit changes into the underlying ELF binary.
		//
		if (elf_update(e3, ELF_C_WRITE) == -1) {
			fprintf(stderr, "elf_update() failed for %s: %s\n",
				output, elf_errmsg(-1));
			throw;
		}

		elf_end(e1);
		elf_end(e2);
		elf_end(e3);
		close(fd1);
		close(fd2);
		close(fd3);
		e1 = NULL;
		e2 = NULL;
		e3 = NULL;
	} catch (...) {
		if (e1)
			elf_end(e1);
		if (e2)
			elf_end(e2);
		if (e3)
			elf_end(e3);
		if (fd1 >= 0)
			close(fd1);
		if (fd2 >= 0)
			close(fd2);
		if (fd3 >= 0)
			close(fd3);
		throw;
	}
}
