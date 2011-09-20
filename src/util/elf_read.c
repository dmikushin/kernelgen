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
#include <malloc.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>

// Load the specified ELF image symbol raw data.
int util_elf_read(const int fd, const char* symname,
	char** symdata, size_t* symsize)
{
	if (!symname)
	{
		fprintf(stderr, "Invalid symbol name\n");
		return 1;
	}
	if (!symdata)
	{
		fprintf(stderr, "Invalid symbol data pointer\n");
		return 1;
	}
	if (!symsize)
	{
		fprintf(stderr, "Invalid symbol data size pointer\n");
		return 1;
	}
	
	GElf_Shdr shdr;
	Elf_Scn* scn = NULL;
	Elf_Data* symbols = NULL;
	
	int status = 0;
	
	if (fd < 0)
	{
		fprintf(stderr, "Invalid file descriptor\n");
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

	Elf* e = elf_begin(fd, ELF_C_READ, NULL);
	if (!e)
	{
		fprintf(stderr, "elf_begin() failed: %s\n",
			elf_errmsg(-1));
		status = 1;
		goto finish;
	}

	while ((scn = elf_nextscn(e, scn)) != NULL)
	{
		if (!gelf_getshdr(scn, &shdr))
		{
			fprintf(stderr, "gelf_getshdr() failed: %s\n",
				elf_errmsg(-1));
			status = 1;
			goto finish;
		}

		if (shdr.sh_type == SHT_SYMTAB)
		{
			symbols = elf_getdata(scn, NULL);
			if (!symbols)
			{
				fprintf(stderr, "elf_getdata() failed: %s\n",
					elf_errmsg(-1));
				status = 1;
				goto finish;
			}
			break;
		}
	}
	
	if (!scn)
	{
		fprintf(stderr, "Cannot find valid sections\n");
		status = 1;
		goto finish;
	}
	
	int nsymbols = 0;
	if (shdr.sh_entsize)
		nsymbols = shdr.sh_size / shdr.sh_entsize;	

	// In symbols table find the name of entire symbol
	// by the known name.
	int found = 0;
	for (int isymbol = 0; isymbol < nsymbols; isymbol++)
	{
		GElf_Sym symbol;
		gelf_getsym(symbols, isymbol, &symbol);
		char* name = elf_strptr(
			e, shdr.sh_link, symbol.st_name);
		if (!strcmp(name, symname))
		{
			*symdata = (char*)(size_t)symbol.st_value;
			*symsize = symbol.st_size;
			found = 1;
			break;
		}
	}
	if (!found)
	{
		printf("Cannot find symbol %s\n", symname);
		status = 1;
	}

finish:

	if (e) elf_end(e);

	return status;
}

// Load the specified ELF image symbols raw data.
int util_elf_read_many(const int fd, const int count,
	const char** symnames, char** symdatas, size_t* symsize)
{
	if (!symnames)
	{
		fprintf(stderr, "Invalid symbols names array\n");
		return 1;
	}
	if (!symdatas)
	{
		fprintf(stderr, "Invalid symbols data array pointer\n");
		return 1;
	}
	if (!symsize)
	{
		fprintf(stderr, "Invalid symbols data sizes array pointer\n");
		return 1;
	}
	
	GElf_Shdr shdr;
	Elf_Scn* scn = NULL;
	Elf_Data* symbols = NULL;
	
	int status = 0;
	
	if (fd < 0)
	{
		fprintf(stderr, "Invalid file descriptor\n");
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

	Elf* e = elf_begin(fd, ELF_C_READ, NULL);
	if (!e)
	{
		fprintf(stderr, "elf_begin() failed: %s\n",
			elf_errmsg(-1));
		status = 1;
		goto finish;
	}

	while ((scn = elf_nextscn(e, scn)) != NULL)
	{
		if (!gelf_getshdr(scn, &shdr))
		{
			fprintf(stderr, "gelf_getshdr() failed: %s\n",
				elf_errmsg(-1));
			status = 1;
			goto finish;
		}

		if (shdr.sh_type == SHT_SYMTAB)
		{
			symbols = elf_getdata(scn, NULL);
			if (!symbols)
			{
				fprintf(stderr, "elf_getdata() failed: %s\n",
					elf_errmsg(-1));
				status = 1;
				goto finish;
			}
			break;
		}
	}
	
	if (!scn)
	{
		fprintf(stderr, "Cannot find valid sections\n");
		status = 1;
		goto finish;
	}
	
	int nsymbols = 0;
	if (shdr.sh_entsize)
		nsymbols = shdr.sh_size / shdr.sh_entsize;	

	// In symbols table find the name of entire symbol
	// by the known name.
	int found = 0;
	for (int jsymbol = 0; jsymbol < count; jsymbol++)
	{
		const char* symname = symnames[jsymbol];			
		for (int isymbol = 0; isymbol < nsymbols; isymbol++)
		{
			GElf_Sym symbol;
			gelf_getsym(symbols, isymbol, &symbol);
			char* name = elf_strptr(
				e, shdr.sh_link, symbol.st_name);
			if (!strcmp(name, symname))
			{
				symdatas[jsymbol] = (char*)(size_t)symbol.st_value;
				symsize[jsymbol] = symbol.st_size;
				
				// If object is not fully linked, address value
				// could be representing offset, not absolute address.
				// TODO: set condition on when it happens
				scn = NULL;
				for (int i = 0; (i < symbol.st_shndx) &&
					((scn = elf_nextscn(e, scn)) != NULL); i++)
				{
					if (!gelf_getshdr(scn, &shdr))
					{
						fprintf(stderr, "gelf_getshdr() failed: %s\n",
							elf_errmsg(-1));
						status = 1;
						goto finish;
					}
				}
				if (!scn)
				{
					fprintf(stderr, "Invalid section index: %d\n",
						symbol.st_shndx);
					status = 1;
					goto finish;
				}
				
				// Load actual data from file.
				size_t offset = shdr.sh_offset + symbol.st_value;
				if (lseek(fd, offset, SEEK_SET) == -1)
				{
					fprintf(stderr, "Cannot set file position to %zu\n", offset);
					status = 1;
					goto finish;
				}
				symdatas[jsymbol] = (char*)malloc(symsize[jsymbol]);
				if (read(fd, symdatas[jsymbol], symsize[jsymbol]) == -1)
				{
					fprintf(stderr, "Cannot read section data from file\n");
					status = 1;
					goto finish;
				}
				
				found = 1;
				break;
			}
		}
		if (!found)
		{
			printf("Cannot find symbol %s\n", symname);
			status = 1;
		}
	}

finish:

	if (e) elf_end(e);

	return status;
}

// Load the specified ELF executable header.
int util_elf_read_eheader(const char* executable, GElf_Ehdr* ehdr)
{
	int status = 0;
	Elf* e = NULL;
	
	int fd = open(executable, O_RDONLY);
	if (fd < 0)
	{
		fprintf(stderr, "Cannot open file %s\n", "/proc/self/exe");
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
	
	e = elf_begin(fd, ELF_C_READ, NULL);
	if (!e)
	{
		fprintf(stderr, "elf_begin() failed for %s: %s\n",
			"/proc/self/exe", elf_errmsg(-1));
		status = 1;
		goto finish;
	}

	// Get executable elf program header.
	if (!gelf_getehdr(e, ehdr))
	{
		fprintf(stderr, "gelf_getehdr() failed: %s\n",
			elf_errmsg(-1));
		status = 1;
		goto finish;
	}

finish:

	if (e)	elf_end(e);
	if (fd >= 0) close(fd);
	return status;
}

