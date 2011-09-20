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

#include <fcntl.h>
#include <gelf.h>	
#include <libelf.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#define Elf_Sym Elf32_Sym
#define elf_write_symtab elf_write_symtab32

#include "elf_write_int.h"

#undef Elf_Sym
#undef elf_write_symtab

#define Elf_Sym Elf64_Sym
#define elf_write_symtab elf_write_symtab64

#include "elf_write_int.h"

static int elf_write_strtab(Elf* e, GElf_Ehdr* ehdr, char* strings, size_t length)
{
	Elf_Scn* scn = elf_newscn(e);
	if (!scn)
	{
		fprintf(stderr, "elf_newscn() failed: %s\n",
			elf_errmsg(-1));
		return 1;
	}

	Elf_Data* data = elf_newdata(scn);
	if (!data)
	{
		fprintf(stderr, "elf_newdata() failed: %s\n",
			elf_errmsg(-1));
		return 1;
	}
	
	data->d_align = 1;
	data->d_buf = strings;
	data->d_off = 0LL;
	data->d_size = length;
	data->d_type = ELF_T_BYTE;
	data->d_version = EV_CURRENT;

	GElf_Shdr *shdr, shdr_buf;
	shdr = gelf_getshdr(scn, &shdr_buf);
	if (!shdr)
	{
		fprintf(stderr, "gelf_getshdr() failed: %s\n",
			elf_errmsg(-1));
		return 1;
	}
	
	shdr->sh_name = 1;
	shdr->sh_type = SHT_STRTAB;
	shdr->sh_entsize = 0;

	ehdr->e_shstrndx = elf_ndxscn(scn);

	if (!gelf_update_ehdr(e, ehdr))
	{
		fprintf(stderr, "gelf_update_ehdr() failed: %s\n",
			elf_errmsg (-1));
		return 1;
	}

	if (!gelf_update_shdr(scn, shdr))
	{
		fprintf(stderr, "gelf_update_shdr() failed: %s\n",
			elf_errmsg (-1));
		return 1;
	}
	
	return 0;
}

static int elf_write_data(Elf* e, const char* symdata, size_t length)
{
	Elf_Scn* scn = elf_newscn(e);
	if (!scn)
	{
		fprintf(stderr, "elf_newscn() failed: %s\n",
			elf_errmsg(-1));
		return 1;
	}

	Elf_Data* data = elf_newdata(scn);
	if (!data)
	{
		fprintf(stderr, "elf_newdata() failed: %s\n",
			elf_errmsg(-1));
		return 1;
	}
	
	data->d_align = 1;
	data->d_buf = (char*)symdata;
	data->d_off = 0LL;
	data->d_size = length;
	data->d_type = ELF_T_BYTE;
	data->d_version = EV_CURRENT;

	GElf_Shdr *shdr, shdr_buf;
	shdr = gelf_getshdr(scn, &shdr_buf);
	if (!shdr)
	{
		fprintf(stderr, "gelf_getshdr() failed: %s\n",
			elf_errmsg(-1));
		return 1;
	}
	
	shdr->sh_name = 17;
	shdr->sh_type = SHT_PROGBITS;
	shdr->sh_flags = SHF_ALLOC | SHF_WRITE;
	shdr->sh_entsize = 0;

	if (!gelf_update_shdr(scn, shdr))
	{
		fprintf(stderr, "gelf_update_shdr() failed: %s\n",
			elf_errmsg (-1));
		return 1;
	}
	
	return 0;
}

// Create ELF image containing symbol with the specified name,
// associated data content and its length.
int util_elf_write(const int fd, const int arch,
	const char* symname, const char* symdata, size_t length)
{
	int status = 0;
	Elf* e = NULL;
	char* strings = NULL;

	if (fd < 0)
	{
		fprintf(stderr, "Invalid file descriptor\n");
		status = 1;
		goto finish;
	}

	if ((arch != 32) && (arch != 64))
	{
		fprintf(stderr, "Invalid architecture: %d\n", arch);
		status = 1;
		goto finish;
	}

	if (elf_version(EV_CURRENT) == EV_NONE)
	{
		fprintf(stderr, "ELF library initialization failed: %s\n",
			elf_errmsg(-1));
		status = 1;
		goto finish;
	}

	e = elf_begin(fd, ELF_C_WRITE, NULL);
	if (!e)
	{
		fprintf(stderr, "elf_begin() failed: %s\n",
			elf_errmsg(-1));
		status = 1;
		goto finish;
	}
	
	GElf_Ehdr* ehdr = (GElf_Ehdr*)gelf_newehdr(e,
		(arch == 64) ? ELFCLASS64 : ELFCLASS32);
	if (!ehdr)
	{
		fprintf(stderr, "gelf_newehdr() failed: %s\n",
			elf_errmsg(-1));
		status = 1;
		goto finish;
	}
	
	ehdr->e_machine = (arch == 64) ? EM_X86_64 : EM_386;
	ehdr->e_type = ET_REL;
	ehdr->e_version = EV_CURRENT;

	// Put together names of ELF sections and name of the target symbol.
	const char sections[] = "\0.strtab\0.symtab\0.data";
	size_t szsymname = strlen(symname) + 1;
	size_t szstrings = szsymname + sizeof(sections);
	strings = (char*)malloc(szstrings);
	memcpy(strings, sections, sizeof(sections));
	memcpy(strings + sizeof(sections), symname, szsymname);

	if (arch == 64)
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
		sym[1].st_size = length;
		sym[1].st_info = ELF32_ST_INFO(STB_GLOBAL, STT_OBJECT);
		sym[1].st_other = 0;
		sym[1].st_shndx = 3;
		sym[1].st_name = sizeof(sections);

		status = elf_write_symtab64(e, sym, 2);
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
		sym[1].st_size = length;
		sym[1].st_info = ELF64_ST_INFO(STB_GLOBAL, STT_OBJECT);
		sym[1].st_other = 0;
		sym[1].st_shndx = 3;
		sym[1].st_name = sizeof(sections);

		status = elf_write_symtab32(e, sym, 2);
	}
	if (status) goto finish;
	
	status = elf_write_strtab(e, ehdr, strings, szstrings);
	if (status) goto finish;

	status = elf_write_data(e, symdata, length);
	if (status) goto finish;
	
	if (elf_update(e, ELF_C_WRITE) < 0)
	{
		fprintf(stderr, "elf_update() failed: %s\n",
			elf_errmsg(-1));
		status = 1;
		goto finish;
	}

finish:

	if (strings) free(strings);

	if (e) elf_end(e);
	
	return status;
}

// Create ELF image containing multiple symbols with the specified names,
// associated data contents and their lengths.
// Duplicate ident and archtecture from the reference executable header.
int util_elf_write_many(const int fd, const int arch, const int count, ...)
{
	int status = 0;
	Elf* e = NULL;
	char* strings = NULL;
	char* data = NULL;

	Elf64_Sym* symbols64 = NULL;
	Elf32_Sym* symbols32 = NULL;

	if (fd < 0)
	{
		fprintf(stderr, "Invalid file descriptor\n");
		status = 1;
		goto finish;
	}

	if ((arch != 32) && (arch != 64))
	{
		fprintf(stderr, "Invalid architecture: %d\n", arch);
		status = 1;
		goto finish;
	}

	if (elf_version(EV_CURRENT) == EV_NONE)
	{
		fprintf(stderr, "ELF library initialization failed: %s\n",
			elf_errmsg(-1));
		status = 1;
		goto finish;
	}

	e = elf_begin(fd, ELF_C_WRITE, NULL);
	if (!e)
	{
		fprintf(stderr, "elf_begin() failed: %s\n",
			elf_errmsg(-1));
		status = 1;
		goto finish;
	}
	
	GElf_Ehdr* ehdr = (GElf_Ehdr*)gelf_newehdr(e,
		(arch == 64) ? ELFCLASS64 : ELFCLASS32);
	if (!ehdr)
	{
		fprintf(stderr, "gelf_newehdr() failed: %s\n",
			elf_errmsg(-1));
		status = 1;
		goto finish;
	}
	
	ehdr->e_machine = (arch == 64) ? EM_X86_64 : EM_386;
	ehdr->e_type = ET_REL;
	ehdr->e_version = EV_CURRENT;

	// Put together names of ELF sections and names of the target symbols.
	const char sections[] = "\0.strtab\0.symtab\0.data";
	va_list list;
	va_start(list, count);
	size_t szdata = 0, szsymname = 0;
	for (int i = 0; i < count; i++)
	{
		const char* symname = (const char*)va_arg(list, const char*);
		szsymname += strlen(symname) + 1;
		const char* symdata = (const char*)va_arg(list, const char*);
		size_t length = (size_t)va_arg(list, size_t);
		szdata += length;
	}
	va_end(list);
	size_t szstrings = szsymname + sizeof(sections);
	strings = (char*)malloc(szstrings);
	memcpy(strings, sections, sizeof(sections));
	va_start(list, count);
	for (int i = 0, offset = sizeof(sections); i < count; i++)
	{
		const char* symname = (const char*)va_arg(list, const char*);
		szsymname = strlen(symname) + 1;
		memcpy(strings + offset, symname, szsymname);
		offset += szsymname;
		const char* symdata = (const char*)va_arg(list, const char*);
		size_t length = (size_t)va_arg(list, size_t);
	}
	va_end(list);

	va_start(list, count);
	if (arch == 64)
	{
		// Symbol table size always starts with one
		// undefined symbol.
		symbols64 = (Elf64_Sym*)malloc(sizeof(Elf64_Sym) * (count + 1));
		symbols64[0].st_value = 0;
		symbols64[0].st_size = 0;
		symbols64[0].st_info = 0;
		symbols64[0].st_other = 0;
		symbols64[0].st_shndx = STN_UNDEF;
		symbols64[0].st_name = 0;
		for (int i = 1, data_offset = 0, name_offset = sizeof(sections); i < count + 1; i++)
		{
			const char* symname = (const char*)va_arg(list, const char*);
			const char* symdata = (const char*)va_arg(list, const char*);
			size_t length = (size_t)va_arg(list, size_t);
			symbols64[i].st_value = data_offset;
			symbols64[i].st_size = length;
			symbols64[i].st_info = ELF32_ST_INFO(STB_GLOBAL, STT_OBJECT);
			symbols64[i].st_other = 0;
			symbols64[i].st_shndx = 3;
			symbols64[i].st_name = name_offset;
			data_offset += length;
			name_offset += strlen(symname) + 1;
		}

		status = elf_write_symtab64(e, symbols64, count + 1);
	}
	else
	{
		// Symbol table size always starts with one
		// undefined symbol.
		symbols32 = (Elf32_Sym*)malloc(sizeof(Elf32_Sym) * (count + 1));
		symbols32[0].st_value = 0;
		symbols32[0].st_size = 0;
		symbols32[0].st_info = 0;
		symbols32[0].st_other = 0;
		symbols32[0].st_shndx = STN_UNDEF;
		symbols32[0].st_name = 0;
		for (int i = 1, data_offset = 0, name_offset = sizeof(sections); i < count + 1; i++)
		{
			const char* symname = (const char*)va_arg(list, const char*);
			const char* symdata = (const char*)va_arg(list, const char*);
			size_t length = (size_t)va_arg(list, size_t);
			symbols32[i].st_value = data_offset;
			symbols32[i].st_size = length;
			symbols32[i].st_info = ELF64_ST_INFO(STB_GLOBAL, STT_OBJECT);
			symbols32[i].st_other = 0;
			symbols32[i].st_shndx = 3;
			symbols32[i].st_name = name_offset;
			data_offset += length;
			name_offset += strlen(symname) + 1;
		}

		status = elf_write_symtab32(e, symbols32, count + 1);
	}
	va_end(list);
	if (status) goto finish;
	
	status = elf_write_strtab(e, ehdr, strings, szstrings);
	if (status) goto finish;

	va_start(list, count);
	data = (char*)malloc(szdata);
	for (int i = 0, offset = 0; i < count; i++)
	{
		const char* symname = (const char*)va_arg(list, const char*);
		const char* symdata = (const char*)va_arg(list, const char*);
		size_t length = (size_t)va_arg(list, size_t);
		memcpy(data + offset, symdata, length);
		offset += length;
	}
	va_end(list);

	status = elf_write_data(e, data, szdata);
	if (status) goto finish;
	
	if (elf_update(e, ELF_C_WRITE) < 0)
	{
		fprintf(stderr, "elf_update() failed: %s\n",
			elf_errmsg(-1));
		status = 1;
		goto finish;
	}

finish:

	if (strings) free(strings);
	if (data) free(data);
	
	if (symbols64) free(symbols64);
	if (symbols32) free(symbols32);
	
	if (e) elf_end(e);
	
	return status;
}

