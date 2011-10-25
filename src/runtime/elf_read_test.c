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
#include <stdio.h>
#include <string.h>
#include <unistd.h>

int main(int argc, char **argv)
{
	if (argc != 3)
	{
		fprintf(stdout, "Purpose: read value of the specified symbol from entire ELF image\n");
		fprintf(stdout, "Usage: %s <filename> <symname>\n", argv[0]);
		return 0;
	}
	
	char* filename = argv[1];
	char* symname = argv[2];
	size_t szsymname = strlen(symname);

	GElf_Shdr shdr;
	Elf_Scn* scn = NULL;
	Elf_Data* symbols = NULL;
	
	int status = 0;
	
	int fd = open(filename, O_RDONLY);
	if (fd < 0)
	{
		fprintf(stderr, "Cannot open file %s\n", filename);
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
		fprintf(stderr, "elf_begin() failed for %s: %s\n",
			filename, elf_errmsg(-1));
		status = 1;
		goto finish;
	}

	while ((scn = elf_nextscn(e, scn)) != NULL)
	{
		if (!gelf_getshdr(scn, &shdr))
		{
			fprintf(stderr, "gelf_getshdr() failed for %s: %s\n",
				filename, elf_errmsg(-1));
			status = 1;
			goto finish;
		}

		if (shdr.sh_type == SHT_SYMTAB)
		{
			symbols = elf_getdata(scn, NULL);
			if (!symbols)
			{
				fprintf(stderr, "elf_getdata() failed for %s: %s\n",
					filename, elf_errmsg(-1));
				status = 1;
				goto finish;
			}
			break;
		}
	}
	
	if (!scn)
	{
		fprintf(stderr, "Cannot find valid sections in %s\n",
			filename);
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
		if (!strncmp(name, symname, szsymname))
		{
			printf("Found symbol %s in %s:\n", name, filename);
			fwrite((char*)symbol.st_value, symbol.st_size, 1, stdout);
			printf("\n");
			found = 1;
		}
	}
	if (!found)
	{
		printf("Cannot find symbol %s in %s\n",
			symname, filename);
		status = 1;
	}

finish:

	if (e) elf_end(e);
	if (fd >= 0) close(fd);

	return status;
}

