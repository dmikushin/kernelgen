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

#include "io.h"
#include "util.h"
#include "runtime.h"

#include "cuda_dyloader.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <elf.h>
#include <fcntl.h>
#include <fstream>
#include <gelf.h>
#include <link.h>
#include <sstream>
#include <vector>

using namespace kernelgen;
using namespace kernelgen::bind::cuda;
using namespace kernelgen::runtime;
using namespace util::io;
using namespace std;

static bool debug = false;

// Align cubin global data to the specified boundary.
static void cubin_align_data(const char* cubin, size_t align, list<string>* names)
{
	int fd = -1;
	Elf* e = NULL;
	try
	{
		//
		// 1) First, load the ELF file.
		//
                if ((fd = open(cubin, O_RDWR)) < 0)
                {
			fprintf(stderr, "Cannot open file %s\n", cubin);
			throw;
                }

                if ((e = elf_begin(fd, ELF_C_RDWR, e)) == 0)
                {
			fprintf(stderr, "Cannot read ELF image from %s\n", cubin);
			throw;
                }

		//
		// 2) Find ELF section containing the symbol table and
		// load its data. Meanwhile, search for the global and initialized
		// global sections, count the total number of sections.
		//
		size_t shstrndx;
		if (elf_getshdrstrndx(e, &shstrndx))
		{
			fprintf(stderr, "elf_getshdrstrndx() failed for %s: %s\n",
				cubin, elf_errmsg(-1));
			throw;
		}
		Elf_Data* symbols = NULL;
		int nsymbols = 0, nsections = 0, link = 0;
		Elf_Scn* scn = elf_nextscn(e, NULL);
		Elf_Scn *sglobal, *sglobal_init = NULL;
		GElf_Shdr shglobal, shglobal_init;
		int iglobal = -1, iglobal_init = -1;
		for (int i = 1; scn != NULL; scn = elf_nextscn(e, scn), i++, nsections++)
		{
			GElf_Shdr shdr;
		
			if (!gelf_getshdr(scn, &shdr))
			{
				fprintf(stderr, "gelf_getshdr() failed for %s: %s\n",
					cubin, elf_errmsg(-1));
				throw;
			}

			if (shdr.sh_type == SHT_SYMTAB)
			{
				symbols = elf_getdata(scn, NULL);
				if (!symbols)
				{
					fprintf(stderr, "elf_getdata() failed for %s: %s\n",
						cubin, elf_errmsg(-1));
					throw;
				}
				if (shdr.sh_entsize)
					nsymbols = shdr.sh_size / shdr.sh_entsize;
				link = shdr.sh_link;
			}

			char* name = NULL;
			if ((name = elf_strptr(e, shstrndx, shdr.sh_name)) == NULL)
				throw;

			if (!strcmp(name, ".nv.global"))
			{
				iglobal = i;
				sglobal = scn;
				shglobal = shdr;
			}
			if (!strcmp(name, ".nv.global.init"))
			{
				iglobal_init = i;
				sglobal_init = scn;
				shglobal_init = shdr;
			}
		}
	
		if (!symbols)
		{
			fprintf(stderr, "Cannot find symbols table in %s\n", cubin);
			throw;
		}
		if (iglobal == -1)
		{
			fprintf(stderr, "Cannot find the global symbols section in %s\n", cubin);
			throw;
		}
		if (iglobal_init == -1)
		{
			fprintf(stderr, "Cannot find the initialized global symbols section in %s\n", cubin);
			throw;
		}
		
		//
		// 3) Count what size is needed to keep initialized
		// global symbols, if they were aligned. Also update the sizes
		// and offsets for individual initialized symbols.
		//
                int szglobal_new = 0, szglobal_init_new = 0;
                for (int isymbol = 0; isymbol < nsymbols; isymbol++)
                {
                        GElf_Sym symbol;
                        gelf_getsym(symbols, isymbol, &symbol);

                        if (symbol.st_shndx == iglobal)
                        {
                                symbol.st_value = szglobal_new;
                                if (symbol.st_size % align)
                                        symbol.st_size += align - symbol.st_size % align;
                                szglobal_new += symbol.st_size;

				if (!gelf_update_sym(symbols, isymbol, &symbol))
				{
					fprintf(stderr, "gelf_update_sym() failed for %s: %s\n",
						cubin, elf_errmsg(-1));
					throw;
				}
                        }
                        else if (symbol.st_shndx == iglobal_init)
                        {
				size_t szaligned = symbol.st_size;
                                if (szaligned % align)
                                        szaligned += align - szaligned % align;
                                szglobal_init_new += szaligned;
                        }
                }
                
                //
                // 4) Add new data for aligned globals section.
                //
                vector<char> vglobal_new;
		vglobal_new.resize(szglobal_new);

		Elf_Data* data = elf_getdata(sglobal, NULL);
		if (!data)
		{
			fprintf(stderr, "elf_newdata() failed: %s\n",
				elf_errmsg(-1));
			throw;
		}
	
		data->d_buf = (char*)&vglobal_new[0];
		data->d_size = szglobal_new;

		if (!gelf_update_shdr(sglobal, &shglobal))
		{
			fprintf(stderr, "gelf_update_shdr() failed: %s\n",
				elf_errmsg (-1));
			throw;
		}
		
		//
		// 5) Add new data for aligned initialized globals section.
		//
		vector<char> vglobal_init_new;
		vglobal_init_new.resize(szglobal_init_new);

		data = elf_getdata(sglobal_init, NULL);
		if (!data)
		{
			fprintf(stderr, "elf_newdata() failed: %s\n",
				elf_errmsg(-1));
			throw;
		}
		char* dglobal_init = (char*)data->d_buf;
		
		char* pglobal_init = (char*)dglobal_init;
		char* pglobal_init_new = (char*)&vglobal_init_new[0];
		memset(pglobal_init_new, 0, szglobal_init_new);
		szglobal_init_new = 0;
                for (int isymbol = 0; isymbol < nsymbols; isymbol++)
                {
                        GElf_Sym symbol;
                        gelf_getsym(symbols, isymbol, &symbol);

                        if (symbol.st_shndx == iglobal_init)
                        {
				memcpy(pglobal_init_new, pglobal_init, symbol.st_size);
				pglobal_init += symbol.st_size;

				symbol.st_value = szglobal_init_new;
				if (symbol.st_size % align)
					symbol.st_size += align - symbol.st_size % align;
				szglobal_init_new += symbol.st_size;
				pglobal_init_new += symbol.st_size;

				if (!gelf_update_sym(symbols, isymbol, &symbol))
				{
					fprintf(stderr, "gelf_update_sym() failed for %s: %s\n",
						cubin, elf_errmsg(-1));
					throw;
				}
			}
                }	

		data->d_buf = (char*)&vglobal_init_new[0];
		data->d_size = szglobal_init_new;

		if (!gelf_update_shdr(sglobal_init, &shglobal_init))
		{
			fprintf(stderr, "gelf_update_shdr() failed: %s\n",
				elf_errmsg (-1));
			throw;
		}

		//
		// 6) Commit changes into the underlying ELF binary.
		//
                if (elf_update(e, ELF_C_WRITE) == -1)
                {
                        fprintf(stderr, "Cannot update the ELF image %s\n", cubin);
                        throw;
                }
	
                elf_end(e);
                close(fd);
                e = NULL;

	}
	catch (...)
	{
		if (e) elf_end(e);
		if (fd >= 0) close(fd);
                throw;
        }
}

kernel_func_t kernelgen::runtime::codegen(int runmode, string source, string name, CUstream stream)
{
	// TODO: codegen LLVM IR into PTX or host, depending on the runmode.

	// Load PTX into string.
	string ptx;
	
	if (verbose & KERNELGEN_VERBOSE_SOURCES) cout << ptx;

	// Compile PTX code in temporary file to CUBIN.
	cfiledesc tmp3 = cfiledesc::mktemp("/tmp/");
	{
		string ptxas = "ptxas";
		std::list<string> ptxas_args;
		if (verbose) ptxas_args.push_back("-v");
		ptxas_args.push_back("-arch=sm_20");
		ptxas_args.push_back("-m64");
		//ptxas_args.push_back(tmp2.getFilename());
		ptxas_args.push_back("-o");
		ptxas_args.push_back(tmp3.getFilename());
		if (debug)
		{
			ptxas_args.push_back("-g");
			ptxas_args.push_back("--dont-merge-basicblocks");
			ptxas_args.push_back("--return-at-end");
		}
		if (verbose)
		{
			cout << ptxas;
                        for (std::list<string>::iterator it = ptxas_args.begin();
                                it != ptxas_args.end(); it++)
                                cout << " " << *it;
                        cout << endl;
                }
                execute(ptxas, ptxas_args, "", NULL, NULL);
	}

	// Align cubin global data to the virtual memory page boundary.
	std::list<string> names;
	if (name == "__kernelgen_main")
		cubin_align_data(tmp3.getFilename().c_str(), 4096, &names);

	// Dump Fermi assembly from CUBIN.
	if (verbose & KERNELGEN_VERBOSE_ISA)
	{
		string cuobjdump = "cuobjdump";
		std::list<string> cuobjdump_args;
		cuobjdump_args.push_back("-sass");
		cuobjdump_args.push_back(tmp3.getFilename());
		execute(cuobjdump, cuobjdump_args, "", NULL, NULL);
	}

	// Load CUBIN into string.
	string cubin;
	{
		std::ifstream tmp_stream(tmp3.getFilename().c_str());
		tmp_stream.seekg(0, std::ios::end);
		cubin.reserve(tmp_stream.tellg());
		tmp_stream.seekg(0, std::ios::beg);

		cubin.assign((std::istreambuf_iterator<char>(tmp_stream)),
			std::istreambuf_iterator<char>());
		tmp_stream.close();
	}

	CUfunction kernel_func = NULL;
	if (name == "__kernelgen_main")
	{
		// Load CUBIN from string into module.
		CUmodule module;
		int err = cuModuleLoad(&module, tmp3.getFilename().c_str());
		if (err)
			THROW("Error in cuModuleLoadData " << err);

		err = cuModuleGetFunction(&kernel_func, module, name.c_str());
		if (err)
			THROW("Error in cuModuleGetFunction " << err);
		
		// Check data objects are aligned
		for (list<string>::iterator i = names.begin(), e = names.end(); i != e; i++)
		{
			const char* name = i->c_str();
			void* ptr; size_t size = 0;
			err = cuModuleGetGlobal(&ptr, &size, module, name);
			printf("%s\t%p\t%04zu\n", name, ptr, size);
		}
	}
	else
	{
		// Load kernel function from the binary opcodes.
		CUresult err = cudyLoadCubin((CUDYfunction*)&kernel_func,
			cuda_context->loader, (char*)tmp3.getFilename().c_str(),
			name.c_str(), stream);
		if (err)
			THROW("Error in cudyLoadCubin " << err);
	}
		
	if (verbose)
		cout << "Loaded '" << name << "' at: " << kernel_func << endl;
	
	return (kernel_func_t)kernel_func;
}

