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

#include "bind.h"
#include "io.h"
#include "util.h"
#include "runtime.h"

#include "cuda_dyloader.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <elf.h>
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

bool debug = false;

// Align cubin global data to the specified boundary.
static void align(const char* cubin, size_t align, list<string>* names)
{
	vector<char> container, new_container;
	char *image = NULL, *new_image = NULL;
	try
	{	
		stringstream stream(stringstream::in | stringstream::out |
			stringstream::binary);
		ifstream f(cubin, ios::in | ios::binary);
		stream << f.rdbuf();
		f.close();
		string str = stream.str();
		container.resize(str.size() + 1);
		image = (char*)&container[0];
		memcpy(image, str.c_str(), str.size() + 1);
	}
	catch (...)
	{
		fprintf(stderr, "Error reading data from %s\n", cubin);
		throw;
	}

	Elf* e = NULL;
	try
	{
		if (strncmp(image, ELFMAG, 4))
		{
			fprintf(stderr, "Cannot read ELF image from %s\n", cubin);
			throw;
		}

		e = elf_memory(image, container.size());
		size_t shstrndx;
		if (elf_getshdrstrndx(e, &shstrndx))
		{
			fprintf(stderr, "elf_getshdrstrndx() failed for %s: %s\n",
				cubin, elf_errmsg(-1));
			throw;
		}

		Elf_Data* symbols = NULL;
		int nsymbols = 0, link = 0;
		Elf_Scn* scn = elf_nextscn(e, NULL);
		int iglobal = -1, iglobal_init = -1;
		GElf_Shdr last_shdr, hglobal_init;
		for (int i = 1; scn != NULL; scn = elf_nextscn(e, scn), i++)
		{
			if (!gelf_getshdr(scn, &last_shdr))
			{
				fprintf(stderr, "gelf_getshdr() failed for %s: %s\n",
					cubin, elf_errmsg(-1));
				throw;
			}

			if (last_shdr.sh_type == SHT_SYMTAB)
			{
				symbols = elf_getdata(scn, NULL);
				if (!symbols)
				{
					fprintf(stderr, "elf_getdata() failed for %s: %s\n",
						cubin, elf_errmsg(-1));
					throw;
				}
				if (last_shdr.sh_entsize)
					nsymbols = last_shdr.sh_size / last_shdr.sh_entsize;
				link = last_shdr.sh_link;
			}

			char* name = NULL;
			if ((name = elf_strptr(e, shstrndx, last_shdr.sh_name)) == NULL)
				throw;

			if (!strcmp(name, ".nv.global"))
				iglobal = i;
			if (!strcmp(name, ".nv.global.init"))
			{
				iglobal_init = i;
				hglobal_init = last_shdr;
			}
		}
	
		if (!symbols)
		{
			fprintf(stderr, "Cannot find symbols table in %s\n", cubin);
			throw;
		}

		// Update offsets & sizes for global objects,
		// separately for globals and initialized globals.
		size_t oglobal = 0, oglobal_init = 0;
		list<GElf_Sym> symbols_init;
		for (int isymbol = 0; isymbol < nsymbols; isymbol++)
		{
			GElf_Sym symbol;
			gelf_getsym(symbols, isymbol, &symbol);

			if ((GELF_ST_TYPE(symbol.st_info) == STT_OBJECT) &&
				(GELF_ST_BIND(symbol.st_info) == STB_GLOBAL))
			{
				char* name = elf_strptr(
					e, link, symbol.st_name);
				char* value = (char*)symbol.st_value;
				size_t size = (size_t)symbol.st_size;

				if (symbol.st_shndx == iglobal)
				{
					symbol.st_value = oglobal;
					if (symbol.st_size % align)
						symbol.st_size += 4096 - symbol.st_size % 4096;
					oglobal += symbol.st_size;
				}
				if (symbol.st_shndx == iglobal_init)
				{
					symbols_init.push_back(symbol);
					symbol.st_value = oglobal_init;
					if (symbol.st_size % align)
						symbol.st_size += 4096 - symbol.st_size % 4096;
					oglobal_init += symbol.st_size;
				}

				if (!gelf_update_sym(symbols, isymbol, &symbol))
				{
					fprintf(stderr, "gelf_update_sym() failed for %s: %s\n",
						cubin, elf_errmsg(-1));
					throw;
				}

				value = (char*)symbol.st_value;
				size = (size_t)symbol.st_size;
			}
		}
		elf_end(e);

		// Clone ELF image into new location, inserting
		// new global init section with different size.
		// Write new global init symbols.
		new_container.resize(container.size() + oglobal_init);
		new_image = &new_container[0];
		memcpy(new_image, image, last_shdr.sh_offset + last_shdr.sh_size);
		char* data = image + hglobal_init.sh_offset;
		char* new_data = new_image + last_shdr.sh_offset + last_shdr.sh_size;
		int last_offset = 0;
		for (list<GElf_Sym>::iterator i = symbols_init.begin(), ie = symbols_init.end(); i != ie; i++)
		{
			GElf_Sym symbol = *i;
			memcpy(new_data, data + symbol.st_value, symbol.st_size);
			new_data += 4096;
		}
		memcpy(new_image + last_shdr.sh_offset + last_shdr.sh_size + oglobal_init,
			image + last_shdr.sh_offset + last_shdr.sh_size,
			container.size() - last_shdr.sh_offset - last_shdr.sh_size);

		e = elf_memory(new_image, new_container.size());
		if (elf_getshdrstrndx(e, &shstrndx))
		{
			fprintf(stderr, "elf_getshdrstrndx() failed for %s: %s\n",
				cubin, elf_errmsg(-1));
			throw;
		}

		// Adjust the program header offset.
		GElf_Ehdr ehdr;
		if (!gelf_getehdr(e, &ehdr))
		{
			fprintf(stderr, "gelf_getehdr() failed: %s\n",
				elf_errmsg(-1));
			throw;
		}
		ehdr.e_phoff += oglobal_init;
		if (!gelf_update_ehdr(e, &ehdr))
		{
			fprintf(stderr, "gelf_update_ehdr() failed: %s\n",
				elf_errmsg (-1));
			throw;
		}

		// Update the global init section size and offset.
		scn = elf_nextscn(e, NULL);
		for (int i = 1; i < iglobal_init; i++)
			scn = elf_nextscn(e, scn);
		GElf_Shdr shdr;
		if (!gelf_getshdr(scn, &shdr))
		{
			fprintf(stderr, "gelf_getshdr() failed for %s: %s\n",
				cubin, elf_errmsg(-1));
			throw;
		}
		shdr.sh_offset = last_shdr.sh_offset + last_shdr.sh_size;
		shdr.sh_size = oglobal_init;
		if (!gelf_update_shdr(scn, &shdr))
		{
			fprintf(stderr, "gelf_update_shdr() failed for %s: %s\n",
				cubin, elf_errmsg(-1));
			throw;
		}
	}
	catch (...)
	{
		elf_end(e);
		throw;
	}
	elf_end(e);

	try
	{	
		ofstream f(cubin, ios::out | ios::binary);
		f.write(new_image, new_container.size() - 1);
		f.close();
	}
	catch (...)
	{
		fprintf(stderr, "Error writing data to %s\n", cubin);
		throw;
	}
}

kernel_func_t kernelgen::runtime::nvopencc(string source, string name, CUstream stream)
{
	// Dump generated kernel object to first temporary file.
	cfiledesc tmp1 = cfiledesc::mktemp("/tmp/");
	{
		fstream tmp_stream;
		tmp_stream.open(tmp1.getFilename().c_str(),
			fstream::binary | fstream::out | fstream::trunc);
		tmp_stream << source;
		tmp_stream.close();
	}

	// Compile CUDA code in temporary file to PTX.
	cfiledesc tmp2 = cfiledesc::mktemp("/tmp/");
	{
		string nvopencc = "nvopencc";
		std::list<string> nvopencc_args;
		nvopencc_args.push_back("-D__CUDA_DEVICE_FUNC__");
		nvopencc_args.push_back("-U_FORTIFY_SOURCE");
		nvopencc_args.push_back("-I/opt/kernelgen/include");
		nvopencc_args.push_back("-include");
		nvopencc_args.push_back("kernelgen_runtime.h");
		nvopencc_args.push_back("-x");
		nvopencc_args.push_back("c");
		nvopencc_args.push_back("-TARG:compute_20");
		nvopencc_args.push_back("-m64");
		if (debug)
		{
			nvopencc_args.push_back("-g");
			nvopencc_args.push_back("-O0");
			nvopencc_args.push_back("-OPT:ftz=0");
			nvopencc_args.push_back("-CG:ftz=0");
		}
		else
		{
			nvopencc_args.push_back("-OPT:ftz=1");
			nvopencc_args.push_back("-CG:ftz=1");
		}
		nvopencc_args.push_back("-CG:prec_div=0");
		nvopencc_args.push_back("-CG:prec_sqrt=0");

		// Since the main kernel may need to pass its data to other
		// kernels or host, all symbols must be globally available.
		if (name == "__kernelgen_main")
			nvopencc_args.push_back("-CG:auto_as_static=0");

		// Disable load/store vectorization (very buggy).
		//nvopencc_args.push_back("-CG:vector_loadstore=0");

		nvopencc_args.push_back(tmp1.getFilename());
		nvopencc_args.push_back("-o");
		nvopencc_args.push_back(tmp2.getFilename());
		if (verbose)
		{
			cout << nvopencc;
                        for (std::list<string>::iterator it = nvopencc_args.begin();
                                it != nvopencc_args.end(); it++)
                                cout << " " << *it;
                        cout << endl;
                }
                execute(nvopencc, nvopencc_args, "", NULL, NULL);
	}

	// Load PTX into string.
	string ptx;
	{
		std::ifstream tmp_stream(tmp2.getFilename().c_str());
		tmp_stream.seekg(0, std::ios::end);
		ptx.reserve(tmp_stream.tellg());
		tmp_stream.seekg(0, std::ios::beg);

		ptx.assign((std::istreambuf_iterator<char>(tmp_stream)),
			std::istreambuf_iterator<char>());
		tmp_stream.close();
	}
	
	if (verbose & KERNELGEN_VERBOSE_SOURCES) cout << ptx;

	// Compile PTX code in temporary file to CUBIN.
	cfiledesc tmp3 = cfiledesc::mktemp("/tmp/");
	{
		string ptxas = "ptxas";
		std::list<string> ptxas_args;
		if (verbose) ptxas_args.push_back("-v");
		ptxas_args.push_back("-arch=sm_20");
		ptxas_args.push_back("-m64");
		ptxas_args.push_back(tmp2.getFilename());
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
		align(tmp3.getFilename().c_str(), 4096, &names);

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

	void* kernel_func = NULL;
	if (name == "__kernelgen_main")
	{
		// Load CUBIN from string into module.
		void* module;
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

