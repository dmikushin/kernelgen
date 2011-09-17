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

static int elf_write_symtab(Elf* e, Elf_Sym* sym, int count)
{
	Elf_Scn* scn = elf_newscn(e);
	if (!scn)
	{
		fprintf(stderr, "elf_newscn() failed: %s\n",
			elf_errmsg(-1));
		return 1;
	}
	
	GElf_Shdr *shdr, shdr_buf;
	shdr = gelf_getshdr(scn, &shdr_buf);
	if (!shdr)
	{
		fprintf(stderr, "gelf_getshdr() failed: %s\n",
			elf_errmsg(-1));
		return 1;
	}

	shdr->sh_name = 9;
	shdr->sh_type = SHT_SYMTAB;
	shdr->sh_entsize = sizeof(Elf_Sym);
	shdr->sh_size = shdr->sh_entsize * count;
	shdr->sh_link = 2;

	Elf_Data* data = elf_newdata(scn);
	if (!data)
	{
		fprintf(stderr, "elf_newdata() failed: %s\n",
			elf_errmsg(-1));
		return 1;
	}

	data->d_align = 1;
	data->d_off = 0LL;
	data->d_buf = sym;
	data->d_type = ELF_T_BYTE;
	data->d_size = shdr->sh_entsize * count;
	data->d_version = EV_CURRENT;
	
	if (!gelf_update_shdr(scn, shdr))
	{
		fprintf(stderr, "gelf_update_shdr() failed: %s\n",
			elf_errmsg (-1));
		return 1;
	}
	
	return 0;
}

