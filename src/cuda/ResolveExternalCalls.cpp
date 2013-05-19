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

#include <Cuda.h>
#include <KernelGen.h>
#include <Runtime.h>

#include <cstring>
#include <fcntl.h>
#include <gelf.h>
#include <iostream>
#include <libasfermi.h>
#include <map>
#include <unistd.h>
#include <vector>

#define CUBIN_FUNC_RELOC_TYPE 5

using namespace std;

static map<string, unsigned int> loadEffectiveLayout;

// Get the specified kernel code from its ELF image.
static void GetKernelCode(
		string kernel_name, vector<char>& mcubin, vector<char>& kernel_code)
{
	kernel_name = ".text." + kernel_name;

	Elf *e = NULL;
	try
	{
		// Setup ELF version.
		if (elf_version(EV_CURRENT) == EV_NONE)
			THROW("Cannot initialize ELF library: " << elf_errmsg(-1));

		// First, load input ELF.
		if ((e = elf_memory(&mcubin[0], mcubin.size())) == 0)
			THROW("elf_memory() failed for \"" << kernel_name << "\": " << elf_errmsg(-1));

		// Get sections names section index.
		size_t shstrndx;
		if (elf_getshdrstrndx(e, &shstrndx))
			THROW("elf_getshdrstrndx() failed for " << kernel_name << ": " << elf_errmsg(-1));

		// Find the target kernel section and get its data size.
		Elf_Scn* scn = elf_nextscn(e, NULL);
		for (int i = 1 ; scn != NULL; scn = elf_nextscn(e, scn), i++)
		{
			// Get section header.
			GElf_Shdr shdr;
			if (!gelf_getshdr(scn, &shdr))
				THROW("gelf_getshdr() failed for " << kernel_name << ": " << elf_errmsg(-1));

			// Get name.
			char* cname = NULL;
			if ((cname = elf_strptr(e, shstrndx, shdr.sh_name)) == NULL)
				THROW("Cannot get the name of section " << i << " of " << kernel_name);
			string name = cname;

			if (name != kernel_name) continue;

			// Get section data.
			Elf_Data* data = elf_getdata(scn, NULL);
			if (!data)
				THROW("Expected section " << name << " to contain data in " << kernel_name);

			kernel_code.resize(data->d_size);
			memcpy(&kernel_code[0], data->d_buf, data->d_size);

			elf_end(e);

			return;
		}

		THROW("Kernel " << kernel_name << " not found in CUBIN");
	}
	catch (...)
	{
		if (e)
			elf_end(e);
		throw;
	}
}

// Replace kernel code in ELF image with the specified content.
static void SetKernelCode(
		string kernel_name, const char* cubin, vector<char>& kernel_code)
{
	kernel_name = ".text." + kernel_name;

	int fd = -1;
	Elf *e = NULL;
	try
	{
		// Setup ELF version.
		if (elf_version(EV_CURRENT) == EV_NONE)
			THROW("Cannot initialize ELF library: " << elf_errmsg(-1));

		// First, load input ELF file and output ELF file.
		if ((fd = open(cubin, O_RDWR)) < 0)
			THROW("Cannot open file \"" << cubin << "\"");
		if ((e = elf_begin(fd, ELF_C_RDWR, e)) == 0)
			THROW("elf_begin() failed for \"" << cubin << "\": " << elf_errmsg(-1));

		// Mark the ELF to have managed layout.
		if (elf_flagelf(e, ELF_C_SET, ELF_F_LAYOUT) == 0)
			THROW("elf_flagelf() failed for \"" << cubin << "\": " << elf_errmsg(-1));

		// Get sections names section index.
		size_t shstrndx;
		if (elf_getshdrstrndx(e, &shstrndx))
			THROW("elf_getshdrstrndx() failed for " << kernel_name << ": " << elf_errmsg(-1));

		// Find the target kernel section and get its data size.
		Elf_Scn* scn = elf_nextscn(e, NULL);
		for (int i = 1 ; scn != NULL; scn = elf_nextscn(e, scn), i++)
		{
			// Get section header.
			GElf_Shdr shdr;
			if (!gelf_getshdr(scn, &shdr))
				THROW("gelf_getshdr() failed for " << kernel_name << ": " << elf_errmsg(-1));

			// Get name.
			char* cname = NULL;
			if ((cname = elf_strptr(e, shstrndx, shdr.sh_name)) == NULL)
				THROW("Cannot get the name of section " << i << " of " << kernel_name);
			string name = cname;

			if (name != kernel_name) continue;

			// Get section data.
			Elf_Data* data = elf_getdata(scn, NULL);
			if (!data)
				THROW("Expected section " << name << " to contain data in " << kernel_name);

			// Replace data buffer with provided new content.
			memcpy(data->d_buf, &kernel_code[0], kernel_code.size());

			// Mark data section for update.
			if (elf_flagdata(data, ELF_C_SET, ELF_F_DIRTY) == 0)
				THROW("elf_flagdata() failed for \"" << cubin << "\": " << elf_errmsg(-1));

			// Update ELF.
			if (elf_update(e, ELF_C_WRITE) == -1)
				THROW("elf_update() failed for \"" << cubin << "\": " << elf_errmsg(-1));

			elf_end(e);
			close(fd);

			return;
		}

		THROW("Kernel " << kernel_name << " not found in CUBIN");
	}
	catch (...)
	{
		if (e)
			elf_end(e);
		if (fd >= 0)
			close(fd);
		throw;
	}
}

// Get kernel relocations table from the ELF file.
static void GetKernelRelocations(string kernel_name, vector<char>& mcubin,
	map<string, unsigned int>& kernel_relocations)
{
	kernel_name = ".rel.text." + kernel_name;

	int fd = -1;
	Elf* e = NULL;
	try
	{
		// Setup ELF version.
		if (elf_version(EV_CURRENT) == EV_NONE)
			THROW("Cannot initialize ELF library: " << elf_errmsg(-1));

		// First, load input ELF.
		if ((e = elf_memory(&mcubin[0], mcubin.size())) == 0)
			THROW("elf_memory() failed for \"" << kernel_name << "\": " << elf_errmsg(-1));

		// Get sections names section index.
		size_t shstrndx;
		if (elf_getshdrstrndx(e, &shstrndx))
			THROW("elf_getshdrstrndx() failed for " << kernel_name << ": " << elf_errmsg(-1));

		// First, locate and handle the symbol table.
		Elf_Scn* scn = elf_nextscn(e, NULL);
		int strndx;
		Elf_Data* symtab_data = NULL;
		for (int i = 1; scn != NULL; scn = elf_nextscn(e, scn), i++)
		{
			// Get section header.
			GElf_Shdr shdr;
			if (!gelf_getshdr(scn, &shdr))
				THROW("gelf_getshdr() failed for " << kernel_name << ": " <<  elf_errmsg(-1));

			// If section is not a symbol table:
			if (shdr.sh_type != SHT_SYMTAB) continue;

			// Load symbols.
			symtab_data = elf_getdata(scn, NULL);
			if (!symtab_data)
				THROW("Expected .symtab data section for " << kernel_name);
			strndx = shdr.sh_link;
		}

		// Find relocation section corresponding to the specified kernel.
		scn = elf_nextscn(e, NULL);
		for (int i = 1 ; scn != NULL; scn = elf_nextscn(e, scn), i++)
		{
			// Get section header.
			GElf_Shdr shdr;
			if (!gelf_getshdr(scn, &shdr))
				THROW("gelf_getshdr() failed for " << kernel_name << ": " << elf_errmsg(-1));

			if (shdr.sh_type != SHT_REL) continue;

			// Get name.
			char* cname = NULL;
			if ((cname = elf_strptr(e, shstrndx, shdr.sh_name)) == NULL)
				THROW("Cannot get the name of section " << i << " of " << kernel_name);
			string name = cname;

			if (name != kernel_name) continue;

			if (shdr.sh_size && !shdr.sh_entsize)
				THROW("Cannot get the number of symbols for " << kernel_name);

			// Get section data.
			Elf_Data* data = elf_getdata(scn, NULL);
			if (!data)
				THROW("Expected section " << name << " to contain data in " << kernel_name);

			// Load relocations.
			int nrelocs = 0;
			if (shdr.sh_entsize)
				nrelocs = shdr.sh_size / shdr.sh_entsize;
			for (int k = 0; k < nrelocs; k++)
			{
				GElf_Rel rel;
				if (!gelf_getrel(data, k, &rel))
					THROW("gelf_getrel() failed for " << kernel_name << ": " << elf_errmsg(-1));

				// TODO 64-bit ELF class support only, for now.
				int isym = ELF64_R_SYM(rel.r_info);
				int itype = ELF64_R_TYPE(rel.r_info);

				if (itype != CUBIN_FUNC_RELOC_TYPE) continue;

				// Find symbol name by its index.
				GElf_Sym sym;
				if (!gelf_getsym(symtab_data, isym, &sym))
					THROW("gelf_getsym() failed for " << kernel_name << ": " << elf_errmsg(-1));
				char* name = elf_strptr(e, strndx, sym.st_name);
				if (!name)
					THROW("Cannot get the name of " << i << "-th symbol for " << kernel_name <<
						": " << elf_errmsg(-1));

				if (kernel_relocations.find(name) != kernel_relocations.end()) continue;

				kernel_relocations[name] = rel.r_offset;
			}

			elf_end(e);
			close(fd);
			e = NULL;

			return;
		}
	}
	catch (...)
	{
		if (e)
			elf_end(e);
		if (fd >= 0)
			close(fd);
		throw;
	}
}

// Check if loop kernel contains unresolved calls and resolve them
// using the load-effective layout obtained from the main kernel.
void kernelgen::bind::cuda::CUBIN::ResolveExternalCalls(
		const char* cubin_dst, const char* ckernel_name_dst,
		const char* cubin_src, const char* ckernel_name_src,
		unsigned int kernel_lepc_diff)
{
	// Load CUBIN into memory.
	vector<char> mcubin;
	{
		std::ifstream tmp_stream(cubin_dst);
		tmp_stream.seekg(0, std::ios::end);
		mcubin.resize(tmp_stream.tellg());
		tmp_stream.seekg(0, std::ios::beg);
		mcubin.assign((std::istreambuf_iterator<char>(tmp_stream)),
			std::istreambuf_iterator<char>());
		tmp_stream.close();
	}

	string kernel_name = ckernel_name_dst;

	// Get destination kernel relocations.
	map<string, unsigned int> kernel_relocations;
	GetKernelRelocations(kernel_name, mcubin, kernel_relocations);

	// If kernel has no relocations, there is definitely nothing to do
	// any further.
	if (!kernel_relocations.size()) return;

	// If destination kernel contains JCAL 0x0 instructions, first load the
	// load-effective layout, if not previously loaded.
	if (!loadEffectiveLayout.size())
		CUBIN::GetLoadEffectiveLayout(cubin_src, ckernel_name_src,
				kernel_lepc_diff, loadEffectiveLayout);

	// Load destination kernel binary.
	vector<char> kernel_code;
	GetKernelCode(kernel_name, mcubin, kernel_code);

	// Replace JCAL 0x0 instructions pointed by relocations with
	// JCAL to actual functions addresses from load effective layout table.
	for (map<string, unsigned int>::iterator i = kernel_relocations.begin(),
			ie = kernel_relocations.end(); i != ie; i++)
	{
		string external_call = i->first;
		unsigned int offset = i->second;

		// Get address of function corresponding to the relocation.
		map<string, unsigned int>::iterator entry = loadEffectiveLayout.find(external_call);
		if (entry == loadEffectiveLayout.end())
			THROW("Cannot find load-effective layout for call to " << external_call);
		unsigned int address = entry->second;

		// Codegen JCAL address instruction.
		stringstream sourcestr;
		sourcestr << "!Kernel dummy\n";
		sourcestr << "JCAL 0x" << hex << address << "\n";
		sourcestr << "!EndKernel\n";
		string source = sourcestr.str();
		vector<char> vsource;
		vsource.resize(source.size() + 1);
		memcpy(&source[0], source.c_str(), source.size() + 1);
		size_t szbinary;
		char* binary = asfermi_encode_opcodes(&source[0], 30, &szbinary);
		if (szbinary != 8)
			THROW("Expected 1 opcode, but got szbinary = " << szbinary <<
				" instead");

		// Replace the corresponding instruction in kernel code
		// with the new one.
		memcpy(&kernel_code[offset], binary, szbinary);
		free(binary);

		VERBOSE(Verbose::Loader << "Resolved external call to " << external_call <<
				" in " << kernel_name << "\n" << Verbose::Default);
	}

	// Replace original destination code with the modified version.
	SetKernelCode(kernel_name, cubin_dst, kernel_code);
}
