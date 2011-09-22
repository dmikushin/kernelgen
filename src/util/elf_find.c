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
#include <regex.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>

// Search for names with the given pattern in the specified
// ELF image and gather them in the output array.
int util_elf_find(const int fd, const char* symname_pattern,
	char*** symnames, int* count)
{
	if (!symnames)
	{
		fprintf(stderr, "Invalid symbols names array\n");
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

	// Compile regular expression out of the specified
	// string pattern.
	regex_t regex;
	if (regcomp(&regex, symname_pattern, REG_EXTENDED | REG_NOSUB))
	{
		fprintf(stderr, "Invalid regular expression: %s\n",
			symname_pattern);
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

	// In symbols table find the names of symbols
	// matching the specified regular expression.
	// Count them on the first pass and store list on
	// the second.
	*count = 0;
	for (int isymbol = 0; isymbol < nsymbols; isymbol++)
	{
		GElf_Sym symbol;
		gelf_getsym(symbols, isymbol, &symbol);
		char* name = elf_strptr(
			e, shdr.sh_link, symbol.st_name);
		if (!regexec(&regex, name, (size_t) 0, NULL, 0)) (*count)++;
	}
	*symnames = (char**)malloc(sizeof(char*) * *count);
	for (int isymbol = 0, jsymbol = 0; isymbol < nsymbols; isymbol++)
	{
		GElf_Sym symbol;
		gelf_getsym(symbols, isymbol, &symbol);
		char* name = elf_strptr(
			e, shdr.sh_link, symbol.st_name);
		if (!regexec(&regex, name, (size_t) 0, NULL, 0))
		{
			char** symname = *symnames + jsymbol; 
			*symname = (char*)malloc(strlen(name) + 1);
			strcpy(*symname, name);
			jsymbol++;
		}
	}

finish:

	if (e) elf_end(e);

	// Release compiled regex.
	regfree(&regex);

	return status;
}

