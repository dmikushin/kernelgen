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

#include "llvm/Module.h"
#include "llvm/PassManager.h"
#include "llvm/Target/TargetData.h"
#include "llvm/Target/TargetOptions.h"
#include "llvm/Support/FormattedStream.h"
#include "llvm/Support/Host.h"
#include "llvm/Support/raw_os_ostream.h"
#include "llvm/Support/TargetRegistry.h"
#include "llvm/Support/TargetSelect.h"

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
using namespace llvm;
using namespace util::io;
using namespace std;

static bool debug = false;

// Target machines for runmodes.
auto_ptr<TargetMachine> kernelgen::targets[KERNELGEN_RUNMODE_COUNT];

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
                vector<char>::size_type szglobal_new = 0, szglobal_init_new = 0;
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
				memcpy(pglobal_init_new, pglobal_init + symbol.st_value, symbol.st_size);

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

// Compile C source to x86 binary or PTX assembly,
// using the corresponding LLVM backends.
kernel_func_t kernelgen::runtime::codegen(int runmode, kernel_t* kernel, Module* m)
{
	// Codegen LLVM IR into PTX or host, depending on the runmode.
	string name = kernel->name;
	switch (runmode) {

		case KERNELGEN_RUNMODE_NATIVE :	{

			// Create target machine for NATIVE target and get its target data.
			if (!targets[KERNELGEN_RUNMODE_NATIVE].get()) {
				InitializeAllTargets();
				InitializeAllTargetMCs();
				InitializeAllAsmPrinters();
				InitializeAllAsmParsers();

				Triple triple;
				triple.setTriple(sys::getDefaultTargetTriple());
				string err;
				TargetOptions options;
				const Target* target = TargetRegistry::lookupTarget(triple.getTriple(), err);
				if (!target)
					THROW("Error auto-selecting target for module '" << err << "'." << endl <<
						"Please use the -march option to explicitly pick a target.");
				targets[KERNELGEN_RUNMODE_NATIVE].reset(target->createTargetMachine(
					triple.getTriple(), "", "", options, Reloc::PIC_, CodeModel::Default));
				if (!targets[KERNELGEN_RUNMODE_NATIVE].get())
					THROW("Could not allocate target machine");

				// Override default to generate verbose assembly.
				targets[KERNELGEN_RUNMODE_NATIVE].get()->setAsmVerbosityDefault(true);
			}

			// Setup output stream.
			string bin_string;
			raw_string_ostream bin_stream(bin_string);
			formatted_raw_ostream bin_raw_stream(bin_stream);

			// Ask the target to add backend passes as necessary.
			PassManager manager;
			const TargetData* tdata =
				targets[KERNELGEN_RUNMODE_NATIVE].get()->getTargetData();
			manager.add(new TargetData(*tdata));
			if (targets[KERNELGEN_RUNMODE_NATIVE].get()->addPassesToEmitFile(manager, bin_raw_stream,
				TargetMachine::CGFT_ObjectFile, CodeGenOpt::Aggressive))
				THROW("Target does not support generation of this file type");
			manager.run(*m);

			// Flush the resulting object binary to the
			// underlying string.
			bin_raw_stream.flush();

			// Dump generated kernel object to first temporary file.
			cfiledesc tmp1 = cfiledesc::mktemp("/tmp/");
			{
				fstream tmp_stream;
				tmp_stream.open(tmp1.getFilename().c_str(),
					fstream::binary | fstream::out | fstream::trunc);
				tmp_stream << bin_string;
				tmp_stream.close();
			}

			// Link first and second objects together into third one.
			cfiledesc tmp2 = cfiledesc::mktemp("/tmp/");
			{
				string linker = "ld";
				std::list<string> linker_args;
				linker_args.push_back("-shared");
				linker_args.push_back("-o");
				linker_args.push_back(tmp2.getFilename());
				linker_args.push_back(tmp1.getFilename());
				if (verbose) {
					cout << linker;
					for (std::list<string>::iterator it = linker_args.begin();
						it != linker_args.end(); it++)
						cout << " " << *it;
					cout << endl;
				}
				execute(linker, linker_args, "", NULL, NULL);
			}

			// Load linked image and extract kernel entry point.
			void* handle = dlopen(tmp2.getFilename().c_str(),
				RTLD_NOW | RTLD_GLOBAL | RTLD_DEEPBIND);

			if (!handle)
				THROW("Cannot dlopen " << dlerror());

			kernel_func_t kernel_func = (kernel_func_t)dlsym(handle, name.c_str());
			if (!kernel_func)
				THROW("Cannot dlsym " << dlerror());

			if (verbose)
				cout << "Loaded '" << name << "' at: " << (void*)kernel_func << endl;

			return kernel_func;
	        }
	
		case KERNELGEN_RUNMODE_CUDA : {

			int device;
			CUresult err = cuDeviceGet(&device, 0);
			if (err)
				THROW("Error in cuDeviceGet " << err);

			int major = 2, minor = 0;
			err = cuDeviceComputeCapability(&major, &minor, device);
			if (err)
				THROW("Cannot get the CUDA device compute capability" << err);

			// Create target machine for CUDA target and get its target data.
			if (!targets[KERNELGEN_RUNMODE_CUDA].get()) {
				InitializeAllTargets();
				InitializeAllTargetMCs();
				InitializeAllAsmPrinters();
				InitializeAllAsmParsers();

				const Target* target = NULL;
				Triple triple(m->getTargetTriple());
				if (triple.getTriple().empty())
					triple.setTriple(sys::getDefaultTargetTriple());
				for (TargetRegistry::iterator it = TargetRegistry::begin(),
					ie = TargetRegistry::end(); it != ie; ++it) {
					if (!strcmp(it->getName(), "nvptx64")) {
						target = &*it;
						break;
					}
				}

				if (!target)
					THROW("LLVM is built without NVPTX Backend support");

				stringstream sarch;
				sarch << "sm_" << (major * 10 + minor);
				targets[KERNELGEN_RUNMODE_CUDA].reset(target->createTargetMachine(
					triple.getTriple(), sarch.str(), "", TargetOptions(),
						Reloc::PIC_, CodeModel::Default, CodeGenOpt::Aggressive));
				if (!targets[KERNELGEN_RUNMODE_CUDA].get())
					THROW("Could not allocate target machine");

				// Override default to generate verbose assembly.
				targets[KERNELGEN_RUNMODE_CUDA].get()->setAsmVerbosityDefault(true);
			}

        	        // Setup output stream.
                	string ptx_string;
	                raw_string_ostream ptx_stream(ptx_string);
        	        formatted_raw_ostream ptx_raw_stream(ptx_stream);

			// Ask the target to add backend passes as necessary.
			PassManager manager;
			const TargetData* tdata =
				targets[KERNELGEN_RUNMODE_CUDA].get()->getTargetData();
			manager.add(new TargetData(*tdata));
			if (targets[KERNELGEN_RUNMODE_CUDA].get()->addPassesToEmitFile(manager, ptx_raw_stream,
				TargetMachine::CGFT_AssemblyFile, CodeGenOpt::Aggressive))
				THROW("Target does not support generation of this file type");
			manager.run(*m);

			// Flush the resulting object binary to the
			// underlying string.
			ptx_raw_stream.flush();

			if (verbose & KERNELGEN_VERBOSE_SOURCES) cout << ptx_string;

			// Dump generated kernel object to first temporary file.
			cfiledesc tmp2 = cfiledesc::mktemp("/tmp/");
			{
				fstream tmp_stream;
				tmp_stream.open(tmp2.getFilename().c_str(),
					fstream::binary | fstream::out | fstream::trunc);
				tmp_stream << ptx_string;
				tmp_stream.close();
			}

			// Compile PTX code in temporary file to CUBIN.
			cfiledesc tmp3 = cfiledesc::mktemp("/tmp/");
			{
				string ptxas = "ptxas";
				std::list<string> ptxas_args;
				if (verbose) ptxas_args.push_back("-v");
                                stringstream sarch;
                                sarch << "-arch=sm_" << (major * 10 + minor);
                                ptxas_args.push_back(sarch.str().c_str());
				ptxas_args.push_back("-m64");
				ptxas_args.push_back(tmp2.getFilename());
				ptxas_args.push_back("-o");
				ptxas_args.push_back(tmp3.getFilename());
				if (name != "__kernelgen_main")
				{
					typedef struct
					{
						int maxThreadsPerBlock;
						int maxThreadsDim[3];
						int maxGridSize[3];
						int sharedMemPerBlock;
						int totalConstantMemory;
						int SIMDWidth;
						int memPitch;
						int regsPerBlock;
						int clockRate;
						int textureAlign;
					} CUdevprop;
			
					CUdevprop props;			
					err = cuDeviceGetProperties((void*)&props, device);
					if (err)
						THROW("Error in cuDeviceGetProperties " << err);

					dim3 blockDim = kernel->target[runmode].blockDim;
					int maxregcount = props.regsPerBlock / (blockDim.x * blockDim.y * blockDim.z) - 4;
					if ((major == 3) && (minor >= 5))
					{
						if (maxregcount > 128) maxregcount = 128;
					}
					else
					{
						if (maxregcount > 63) maxregcount = 63;
					}
					ptxas_args.push_back("--maxrregcount");
					std::ostringstream smaxregcount;
					smaxregcount << maxregcount;
					ptxas_args.push_back(smaxregcount.str().c_str());

					// The -g option by some reason is needed even w/o debug.
					// Otherwise some tests are failing.
					//ptxas_args.push_back("-g");
					//ptxas_args.push_back("--cloning=yes");
				}

				ptxas_args.push_back("--cloning=no");
				if (::debug)
				{
					ptxas_args.push_back("-g");
					ptxas_args.push_back("--return-at-end");
					ptxas_args.push_back("--dont-merge-basicblocks");
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
			}
			else
			{
				// FIXME: The following ugly fix mimics the backend asm printer
				// mangler behavior. We should instead get names from the real
				// mangler, but currently it is unclear how to instantiate it,
				// since it needs MCContext, which is not available here.
				string dot = "2E_";
				for (size_t index = name.find(".", 0);
					index = name.find(".", index); index++)
				{
					if (index == string::npos) break;
					name.replace(index, 1, "_");
					name.insert(index + 1, dot);
				}

				// Load kernel function from the binary opcodes.
				CUstream stream = kernel->target[runmode].monitor_kernel_stream;
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
	}
}

