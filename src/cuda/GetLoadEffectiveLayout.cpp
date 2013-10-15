/*
 * KernelGen - the LLVM-based compiler with GPU kernels generation over C
 * backend.
 *
 * Copyright (c) 2011 Dmitry Mikushin
 *
 * This software is provided 'as-is', without any express or implied warranty.
 * In no event will the authors be held liable for any damages arising
 * from the use of this software.
 * Permission is granted to anyone to use this software for any purpose,
 * including commercial applications, and to alter it and redistribute it
 * freely,
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
#include <fstream>
#include <iostream>
#include <libasfermi.h>
#include <libelf/gelf.h>
#include <map>
#include <stdlib.h>
#include <string>
#include <unistd.h>
#include <vector>

using namespace kernelgen;
using namespace kernelgen::runtime;
using namespace kernelgen::bind::cuda;
using namespace std;

#define CUBIN_FUNC_RELOC_TYPE 5

// Get kernel size as it is recorded in ELF.
static size_t GetKernelSize(string kernel_name, vector<char> &mcubin) {
  kernel_name = ".text." + kernel_name;

  Elf *e = NULL;
  try {
    // Setup ELF version.
    if (elf_version(EV_CURRENT) == EV_NONE)
      THROW("Cannot initialize ELF library: " << elf_errmsg(-1));

    // First, load input ELF.
    if ((e = elf_memory(&mcubin[0], mcubin.size())) == 0)
      THROW("elf_memory() failed for \"" << kernel_name
                                         << "\": " << elf_errmsg(-1));

    // Get sections names section index.
    size_t shstrndx;
    if (elf_getshdrstrndx(e, &shstrndx))
      THROW("elf_getshdrstrndx() failed for " << kernel_name << ": "
                                              << elf_errmsg(-1));

    // Find the target kernel section and get its size.
    Elf_Scn *scn = elf_nextscn(e, NULL);
    for (int i = 1; scn != NULL; scn = elf_nextscn(e, scn), i++) {
      // Get section header.
      GElf_Shdr shdr;
      if (!gelf_getshdr(scn, &shdr))
        THROW("gelf_getshdr() failed for " << kernel_name << ": "
                                           << elf_errmsg(-1));

      // Get name.
      char *cname = NULL;
      if ((cname = elf_strptr(e, shstrndx, shdr.sh_name)) == NULL)
        THROW("Cannot get the name of section " << i << " of " << kernel_name);
      string name = cname;

      if (name != kernel_name)
        continue;

      elf_end(e);
      return shdr.sh_size;
    }

    THROW("Kernel " << kernel_name << " not found in CUBIN");
  }
  catch (...) {
    if (e)
      elf_end(e);
    throw;
  }

  return 0;
}

// Get kernel size as it is recorded in ELF.
static unsigned short GetKernelRegcount(string kernel_name, vector<char> &mcubin) {
  kernel_name = ".text." + kernel_name;

  Elf *e = NULL;
  try {
    // Setup ELF version.
    if (elf_version(EV_CURRENT) == EV_NONE)
      THROW("Cannot initialize ELF library: " << elf_errmsg(-1));

    // First, load input ELF.
    if ((e = elf_memory(&mcubin[0], mcubin.size())) == 0)
      THROW("elf_memory() failed for \"" << kernel_name
                                         << "\": " << elf_errmsg(-1));

    // Get sections names section index.
    size_t shstrndx;
    if (elf_getshdrstrndx(e, &shstrndx))
      THROW("elf_getshdrstrndx() failed for " << kernel_name << ": "
                                              << elf_errmsg(-1));

    // Find the target kernel section and get its size.
    Elf_Scn *scn = elf_nextscn(e, NULL);
    for (int i = 1; scn != NULL; scn = elf_nextscn(e, scn), i++) {
      // Get section header.
      GElf_Shdr shdr;
      if (!gelf_getshdr(scn, &shdr))
        THROW("gelf_getshdr() failed for " << kernel_name << ": "
                                           << elf_errmsg(-1));

      // Get name.
      char *cname = NULL;
      if ((cname = elf_strptr(e, shstrndx, shdr.sh_name)) == NULL)
        THROW("Cannot get the name of section " << i << " of " << kernel_name);
      string name = cname;

      if (name != kernel_name)
        continue;
      
      unsigned short regcount = shdr.sh_info >> 24;

      elf_end(e);
      return regcount;
    }

    THROW("Kernel " << kernel_name << " not found in CUBIN");
  }
  catch (...) {
    if (e)
      elf_end(e);
    throw;
  }

  return 0;
}

// Get kernel code as it is loaded on GPU with substituted relocations.
static void GetKernelLoadEffectiveCode(string kernel_name, size_t kernel_size,
                                       unsigned int kernel_lepc,
                                       vector<char> &kernel_code) {
  // Convert LEPC to GPU memory address.
  uint64_t address = 0;
  ((uint32_t *)&address)[0] = kernel_lepc;
  ((uint32_t *)&address)[1] = 0x1;

  stringstream hexaddress;
  hexaddress << hex << address;
  VERBOSE(Verbose::Loader << "GPU address for LEPC = 0x" << hexaddress.str()
                          << "\n" << Verbose::Default);

  // Copy code from GPU memory to host.
  void *buffer = NULL;
  CU_SAFE_CALL(cuMemAlloc((CUdeviceptr *)&buffer, kernel_size));

  struct {
    unsigned int x, y, z;
  } gridDim, blockDim;
  gridDim.x = 1;
  gridDim.y = 1;
  gridDim.z = 1;
  blockDim.x = 1;
  blockDim.y = 1;
  blockDim.z = 1;
  size_t szshmem = 0;

  void *kernel_func_args[] = {(void *)&buffer, (void *)&address,
                              (void *)&kernel_size };
  int err = cuLaunchKernel(cuda_context->kernelgen_memcpy, gridDim.x, gridDim.y,
                           gridDim.z, blockDim.x, blockDim.y, blockDim.z,
                           szshmem, cuda_context->getSecondaryStream(),
                           kernel_func_args, NULL);
  if (err)
    THROW("Error in cuLaunchKernel " << err);
  CU_SAFE_CALL(cuStreamSynchronize(cuda_context->getSecondaryStream()));
  kernel_code.resize(kernel_size);
  CU_SAFE_CALL(
      cuMemcpyDtoHAsync(&kernel_code[0], (CUdeviceptr) buffer, kernel_size,
                        cuda_context->getSecondaryStream()));
  CU_SAFE_CALL(cuStreamSynchronize(cuda_context->getSecondaryStream()));
  //CU_SAFE_CALL(cuMemFree((CUdeviceptr)buffer));
}

// Get kernel relocations table from the ELF file.
static void
GetKernelRelocations(string kernel_name, vector<char> &mcubin,
                     map<string, unsigned int> &kernel_relocations) {
  kernel_name = ".rel.text." + kernel_name;

  int fd = -1;
  Elf *e = NULL;
  try {
    // Setup ELF version.
    if (elf_version(EV_CURRENT) == EV_NONE)
      THROW("Cannot initialize ELF library: " << elf_errmsg(-1));

    // First, load input ELF.
    if ((e = elf_memory(&mcubin[0], mcubin.size())) == 0)
      THROW("elf_memory() failed for \"" << kernel_name
                                         << "\": " << elf_errmsg(-1));

    // Get sections names section index.
    size_t shstrndx;
    if (elf_getshdrstrndx(e, &shstrndx))
      THROW("elf_getshdrstrndx() failed for " << kernel_name << ": "
                                              << elf_errmsg(-1));

    // First, locate and handle the symbol table.
    Elf_Scn *scn = elf_nextscn(e, NULL);
    int strndx;
    Elf_Data *symtab_data = NULL;
    for (int i = 1; scn != NULL; scn = elf_nextscn(e, scn), i++) {
      // Get section header.
      GElf_Shdr shdr;
      if (!gelf_getshdr(scn, &shdr))
        THROW("gelf_getshdr() failed for " << kernel_name << ": "
                                           << elf_errmsg(-1));

      // If section is not a symbol table:
      if (shdr.sh_type != SHT_SYMTAB)
        continue;

      // Load symbols.
      symtab_data = elf_getdata(scn, NULL);
      if (!symtab_data)
        THROW("Expected .symtab data section for " << kernel_name);
      strndx = shdr.sh_link;
    }

    // Find relocation section corresponding to the specified kernel.
    scn = elf_nextscn(e, NULL);
    for (int i = 1; scn != NULL; scn = elf_nextscn(e, scn), i++) {
      // Get section header.
      GElf_Shdr shdr;
      if (!gelf_getshdr(scn, &shdr))
        THROW("gelf_getshdr() failed for " << kernel_name << ": "
                                           << elf_errmsg(-1));

      if (shdr.sh_type != SHT_REL)
        continue;

      // Get name.
      char *cname = NULL;
      if ((cname = elf_strptr(e, shstrndx, shdr.sh_name)) == NULL)
        THROW("Cannot get the name of section " << i << " of " << kernel_name);
      string name = cname;

      if (name != kernel_name)
        continue;

      if (shdr.sh_size && !shdr.sh_entsize)
        THROW("Cannot get the number of symbols for " << kernel_name);

      // Get section data.
      Elf_Data *data = elf_getdata(scn, NULL);
      if (!data)
        THROW("Expected section " << name << " to contain data in "
                                  << kernel_name);

      // Load relocations.
      int nrelocs = 0;
      if (shdr.sh_entsize)
        nrelocs = shdr.sh_size / shdr.sh_entsize;
      for (int k = 0; k < nrelocs; k++) {
        GElf_Rel rel;
        if (!gelf_getrel(data, k, &rel))
          THROW("gelf_getrel() failed for " << kernel_name << ": "
                                            << elf_errmsg(-1));

        // TODO 64-bit ELF class support only, for now.
        int isym = ELF64_R_SYM(rel.r_info);
        int itype = ELF64_R_TYPE(rel.r_info);

        if (itype != CUBIN_FUNC_RELOC_TYPE)
          continue;

        // Find symbol name by its index.
        GElf_Sym sym;
        if (!gelf_getsym(symtab_data, isym, &sym))
          THROW("gelf_getsym() failed for " << kernel_name << ": "
                                            << elf_errmsg(-1));
        char *name = elf_strptr(e, strndx, sym.st_name);
        if (!name)
          THROW("Cannot get the name of " << i << "-th symbol for "
                                          << kernel_name << ": "
                                          << elf_errmsg(-1));

        if (kernel_relocations.find(name) != kernel_relocations.end())
          continue;

        kernel_relocations[name] = rel.r_offset;
      }

      elf_end(e);
      close(fd);
      e = NULL;

      return;
    }
  }
  catch (...) {
    if (e)
      elf_end(e);
    if (fd >= 0)
      close(fd);
    throw;
  }
}

static unsigned int GetKernelJcalTarget(uint64_t jcal_cmd) {
  return (jcal_cmd - 0x1000000000010007ULL) >> 26;
}

// Discover kernels load-effective layout.
static void GetKernelsLoadEffectiveLayout(map<string, pair<unsigned int, unsigned short> > &layout,
                                          string kernel_name,
                                          unsigned int kernel_lepc,
                                          vector<char> &mcubin) {
  VERBOSE(Verbose::Loader << "kernel " << kernel_name << "\n"
                          << Verbose::Default);

  // Get the size of current kernel.
  size_t kernel_size = GetKernelSize(kernel_name, mcubin);

  VERBOSE(Verbose::Loader << "kernel_size = " << kernel_size << "\n"
                          << Verbose::Default);

  // Get the code of the kernel as it is loaded on GPU.
  vector<char> kernel_code;
  GetKernelLoadEffectiveCode(kernel_name, kernel_size, kernel_lepc,
                             kernel_code);

  VERBOSE(Verbose::Loader << kernel_name << " code: "
                          << "\n" << Verbose::Default);
  for (int i = 0; i < kernel_size; i += 8) {
    stringstream opcode;
    opcode << hex << *(uint64_t *)&kernel_code[i];
    VERBOSE(Verbose::Loader << "0x" << opcode.str() << "\n"
                            << Verbose::Default);
  }
  VERBOSE(Verbose::Loader << "\n" << Verbose::Default);

  // Get relocations in kernel code.
  map<string, unsigned int> kernel_relocations;
  GetKernelRelocations(kernel_name, mcubin, kernel_relocations);

  if (kernel_relocations.size()) {
    VERBOSE(Verbose::Loader << kernel_name << " relocations: "
                            << "\n" << Verbose::Default);
    for (map<string, unsigned int>::iterator i = kernel_relocations.begin(),
                                             ie = kernel_relocations.end();
         i != ie; i++) {
      kernel_name = i->first;
      unsigned int offset = i->second;

      stringstream hexoffset;
      hexoffset << hex << offset;
      VERBOSE(Verbose::Loader << "0x" << hexoffset.str() << " " << kernel_name
                              << "\n" << Verbose::Default);
    }
    VERBOSE(Verbose::Loader << "\n" << Verbose::Default);
  }

  for (map<string, unsigned int>::iterator i = kernel_relocations.begin(),
                                           ie = kernel_relocations.end();
       i != ie; i++) {
    kernel_name = i->first;
    unsigned int offset = i->second;

    // Check if entire kernel is already discovered.
    if (layout.find(kernel_name) != layout.end())
      continue;

    uint64_t jcal_cmd = *(uint64_t *)(&kernel_code[0] + offset);
    kernel_lepc = GetKernelJcalTarget(jcal_cmd);

    // Get the regcount of the current kernel.
    unsigned short regcount = GetKernelRegcount(kernel_name, mcubin);

    // Add the current kernel to index.
    layout[kernel_name] = pair<unsigned int, unsigned short>(kernel_lepc, regcount);

    // Discover kernels that might be called from entire kernel.
    GetKernelsLoadEffectiveLayout(layout, kernel_name, kernel_lepc, mcubin);
  }
}

// Get CUBIN Load-effective layout - the runtime address ranges of kernels code,
// as they are loaded into GPU memory.
void kernelgen::bind::cuda::CUBIN::GetLoadEffectiveLayout(
    const char *cubin, const char *ckernel_name, unsigned int kernel_lepc_diff,
    map<string, pair<unsigned int, unsigned short> > &layout) {

  string kernel_name = ckernel_name;

  // Read LEPC.
  unsigned int kernel_lepc = cuda_context->getLEPC();
  kernel_lepc -= kernel_lepc_diff;
  stringstream hexlepc;
  hexlepc << hex << kernel_lepc;
  VERBOSE(Verbose::Loader << "LEPC = 0x" << hexlepc << "\n"
                          << Verbose::Default);

  // Load CUBIN into memory.
  vector<char> mcubin;
  {
    std::ifstream tmp_stream(cubin);
    tmp_stream.seekg(0, std::ios::end);
    mcubin.resize(tmp_stream.tellg());
    tmp_stream.seekg(0, std::ios::beg);
    mcubin.assign((std::istreambuf_iterator<char>(tmp_stream)),
                  std::istreambuf_iterator<char>());
    tmp_stream.close();
  }

  // Get the regcount of the current kernel.
  unsigned short regcount = GetKernelRegcount(kernel_name, mcubin);

  // Add the current kernel to index.
  layout[kernel_name] = pair<unsigned int, unsigned short>(kernel_lepc, regcount);

  GetKernelsLoadEffectiveLayout(layout, kernel_name, kernel_lepc, mcubin);

  for (map<string, pair<unsigned int, unsigned short> >::iterator i = layout.begin(),
                                           ie = layout.end();
       i != ie; i++) {
    stringstream hexaddress;
    hexaddress << hex << i->second.first;
    VERBOSE(Verbose::Loader << i->first << " -> 0x" << hexaddress.str() << "\n"
                            << Verbose::Default);
  }
}
