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
#include "KernelGen.h"

#include <cstring>
#include <fcntl.h>
#include <gelf.h>
#include <iostream>
#include <libasfermi.h>
#include <vector>

using namespace std;

// Insert commands to perform LEPC reporting.
void kernelgen::bind::cuda::CUBIN::InsertLEPCReporter(const char* cubin, const char* ckernel_name)
{
	string kernel_name = ".text." + (string)ckernel_name;

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
			THROW("elf_getshdrstrndx() failed for " << cubin << ": " << elf_errmsg(-1));

		// Find the target kernel section and modify its data.
		bool found = 0;
		Elf_Scn* scn = elf_nextscn(e, NULL);
		for (int i = 1 ; scn != NULL; scn = elf_nextscn(e, scn), i++)
		{
			// Get section header.
			GElf_Shdr shdr;
			if (!gelf_getshdr(scn, &shdr))
				THROW("gelf_getshdr() failed for " << cubin << ": " << elf_errmsg(-1));

			// Get name.
			char* cname = NULL;
			if ((cname = elf_strptr(e, shstrndx, shdr.sh_name)) == NULL)
				THROW("Cannot get the name of section " << i << " of " << cubin);
			string name = cname;

			if (name != kernel_name) continue;

			// Get section data.
			Elf_Data* data = elf_getdata(scn, NULL);
			if (!data)
				THROW("Expected section " << name << " to contain data in " << cubin);

			// Get a binary representation for a command being searched.
			uint64_t search[6];
			{
				char source[] = "!Kernel dummy\nMEMBAR.CTA;\n!EndKernel\n";
				size_t szbinary;
				char* binary = asfermi_encode_opcodes(source, 30, &szbinary);
				if (szbinary != 8)
					THROW("Expected 1 opcode for MEMBAR.CTA, but got szbinary = " <<
							szbinary << " instead");
				for (int i = 0; i < 6; i++)
					memcpy(&search[i], binary, 8);
				free(binary);
			}

			// Get a binary representation for commands to replace the
			// found entry.
			uint64_t replacement[6];
			{
				char source[] =
					"!Kernel dummy\n"
					"LEPC R2;\n"
					"MOV R4, c [0x0] [0x148];\n"
					"MOV R5, c [0x0] [0x14c];\n"
					"ST.E.64 [R4], R2;\n"
					"NOP;\n"
					"NOP;\n"
					"!EndKernel\n";
				size_t szbinary;
				char* binary = asfermi_encode_opcodes(source, 30, &szbinary);
				if (szbinary != 6 * 8)
					THROW("Expected 6 opcodes, but got szbinary = " << szbinary <<
						" instead");
				memcpy(&replacement, binary, szbinary);
				free(binary);
			}

			// Find a sequence of 6 commands being searched.
			for (int k = 0, ke = data->d_size - 5 * 8; k != ke; k += 8)
				if (!memcmp((char*)data->d_buf + k, (char*)search, 6 * 8))
					memcpy((char*)data->d_buf + k, (char*)replacement, 6 * 8);

			// Mark data section for update.
			if (elf_flagdata(data, ELF_C_SET, ELF_F_DIRTY) == 0)
				THROW("elf_flagdata() failed for \"" << cubin << "\": " << elf_errmsg(-1));

			// Update ELF.
			if (elf_update(e, ELF_C_WRITE) == -1)
				THROW("elf_update() failed for \"" << cubin << "\": " << elf_errmsg(-1));

			elf_end(e);
			close(fd);

			found = 1;
			break;
		}
		if (!found)
			THROW("Kernel " << kernel_name << " not found in CUBIN " << cubin);
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
