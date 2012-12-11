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
#include "Runtime.h"

#include <cstring>
#include <fcntl.h>
#include <gelf.h>
#include <iostream>
#include <libasfermi.h>
#include <vector>

using namespace kernelgen::runtime;
using namespace std;

// Insert commands to perform LEPC reporting.
unsigned int kernelgen::bind::cuda::CUBIN::InsertLEPCReporter(
		const char* cubin, const char* ckernel_name)
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
			vector<uint64_t> search;
			if (cuda_context->getSubarchMajor() == 2)
			{
				// Expecting 12 instructions in resulting binary.
				int szbinaryExpected = 12;
				search.resize(szbinaryExpected);
				szbinaryExpected *= 8;

				char source[] =
						"!Kernel dummy\n"
						"MEMBAR.CTA;\n"
						"BPT.DRAIN 0x0;\n"
						"MEMBAR.CTA;\n"
						"BPT.DRAIN 0x0;\n"
						"MEMBAR.CTA;\n"
						"BPT.DRAIN 0x0;\n"
						"MEMBAR.CTA;\n"
						"BPT.DRAIN 0x0;\n"
						"MEMBAR.CTA;\n"
						"BPT.DRAIN 0x0;\n"
						"MEMBAR.CTA;\n"
						"BPT.DRAIN 0x0;\n"
						"!EndKernel\n";

				size_t szbinary;
				char* binary = asfermi_encode_opcodes(source,
						cuda_context->getSubarchMajor() * 10 +
						cuda_context->getSubarchMinor(), &szbinary);
				if (szbinary != szbinaryExpected)
					THROW("Unexpected CUBIN size: have " << szbinary << ", expected " <<
							szbinaryExpected);
				memcpy(&search[0], binary, szbinary);
				free(binary);
			}
			else if (cuda_context->getSubarchMajor() == 3)
			{
				// Expecting 6 instructions in resulting binary.
				int szbinaryExpected = 6;
				search.resize(szbinaryExpected);
				szbinaryExpected *= 8;

				char source[] =
						"!Kernel dummy\n"
						"MEMBAR.CTA;\n"
						"MEMBAR.CTA;\n"
						"MEMBAR.CTA;\n"
						"MEMBAR.CTA;\n"
						"MEMBAR.CTA;\n"
						"MEMBAR.CTA;\n"
						"!EndKernel\n";

				size_t szbinary;
				char* binary = asfermi_encode_opcodes(source,
						cuda_context->getSubarchMajor() * 10 +
						cuda_context->getSubarchMinor(), &szbinary);
				if (szbinary != szbinaryExpected)
					THROW("Unexpected CUBIN size: have " << szbinary << ", expected " <<
							szbinaryExpected);
				memcpy(&search[0], binary, szbinary);
				free(binary);
			}
			else
				THROW("KernelGen dyloader is not tested with targets >= sm_3x");

			// Get a binary representation for commands to replace the
			// found entry.
			vector<uint64_t> replacement;
			if (cuda_context->getSubarchMajor() == 2)
			{
				// Expecting 12 instructions in resulting binary.
				int szbinaryExpected = 12;
				replacement.resize(szbinaryExpected);
				szbinaryExpected *= 8;

				char source[] =
						"!Kernel dummy\n"
						"LEPC R2;\n"
						"MOV R4, c [0x0] [0x28];\n"
						"MOV R5, c [0x0] [0x2c];\n"
						"ST.E.64 [R4], R2;\n"
						"NOP;\n"
						"NOP;\n"
						"NOP;\n"
						"NOP;\n"
						"NOP;\n"
						"NOP;\n"
						"NOP;\n"
						"NOP;\n"
						"!EndKernel\n";

				size_t szbinary;
				char* binary = asfermi_encode_opcodes(source,
						cuda_context->getSubarchMajor() * 10 +
						cuda_context->getSubarchMinor(), &szbinary);
				if (szbinary != szbinaryExpected)
					THROW("Unexpected CUBIN size: have " << szbinary << ", expected " <<
							szbinaryExpected);
				memcpy(&replacement[0], binary, szbinary);
				free(binary);
			}
			else if (cuda_context->getSubarchMajor() == 3)
			{
				// Expecting 6 instructions in resulting binary.
				int szbinaryExpected = 6;
				replacement.resize(szbinaryExpected);
				szbinaryExpected *= 8;

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
				char* binary = asfermi_encode_opcodes(source,
						cuda_context->getSubarchMajor() * 10 +
						cuda_context->getSubarchMinor(), &szbinary);
				if (szbinary != szbinaryExpected)
					THROW("Unexpected CUBIN size: have " << szbinary << ", expected " <<
							szbinaryExpected);
				memcpy(&replacement[0], binary, szbinary);
				free(binary);
			}
			else
				THROW("KernelGen dyloader is not tested with targets >= sm_3x");

			// Find a first occurrence of commands sequence being searched and replace it.
			unsigned int lepc_offset = (unsigned int)-1;
			for (int k = 0, ke = data->d_size - (search.size() - 1) * 8; k != ke; k += 8)
				if (!memcmp((char*)data->d_buf + k, (char*)&search[0], search.size() * 8))
				{
					memcpy((char*)data->d_buf + k, (char*)&replacement[0], search.size() * 8);
					lepc_offset = k;
					break;
				}
			if (lepc_offset == (unsigned int)-1)
				THROW("Cannot find the control code sequence to be replaced by LEPC");

			// Mark data section for update.
			if (elf_flagdata(data, ELF_C_SET, ELF_F_DIRTY) == 0)
				THROW("elf_flagdata() failed for \"" << cubin << "\": " << elf_errmsg(-1));

			// Update ELF.
			if (elf_update(e, ELF_C_WRITE) == -1)
				THROW("elf_update() failed for \"" << cubin << "\": " << elf_errmsg(-1));

			elf_end(e);
			close(fd);

			return lepc_offset;
		}

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
