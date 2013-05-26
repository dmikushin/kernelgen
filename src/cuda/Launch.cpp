//===- Memory.cpp - Device memory pool for NVIDIA GPUs --------------------===//
//
//     KernelGen -- A prototype of LLVM-based auto-parallelizing Fortran/C
//        compiler for NVIDIA GPUs, targeting numerical modeling code.
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains functions to support the dynamic memory heap in GPU
// global memory. The rationale is to replace builtin malloc/free calls, since
// they are incompatible with concurrent kernels execution.
//
//===----------------------------------------------------------------------===//

#include "Runtime.h"

#include <dlfcn.h>
#include <fcntl.h>
#include <libelf/gelf.h>
#include <string.h>
#include <unistd.h>

using namespace kernelgen;
using namespace kernelgen::bind::cuda;
using namespace kernelgen::runtime;
using namespace std;

using namespace std;

namespace {

struct uint3 {
  unsigned int x, y, z;
};

class Cubin {
  char *start;
  size_t size;

  bool materialized;

  vector<string> kernelsNames;

  int virt_cc, real_cc;

  CUmodule module;

public:

  Cubin() : start(NULL), size(0), materialized(false) {}

  Cubin(char *start, size_t size)
      : start(start), size(size), materialized(false) {}

  int getVirtualCC() const { return virt_cc; }

  int getRealCC() const { return real_cc; }

  const vector<string> &getKernelsNames() {
    if (!materialized)
      Materialize();
    return kernelsNames;
  }

  // Launch the specificed kernel from the entire CUBIN.
  int LaunchKernel(const char *kernelName, unsigned int gridDimX,
                   unsigned int gridDimY, unsigned int gridDimZ,
                   unsigned int blockDimX, unsigned int blockDimY,
                   unsigned int blockDimZ, unsigned int sharedMemBytes,
                   CUstream hStream, void *args, size_t szargs) {
    if (!materialized)
      Materialize();

    CUfunction kernel_func = NULL;

    // Load kernel function from the binary opcodes.
    // TODO: do it only once.
    CU_SAFE_CALL(cudyLoadCubinData((CUDYfunction *)&kernel_func,
                                   cuda_context->loader, kernelName, start,
                                   size, cuda_context->getSecondaryStream()));

    // Launch kernel function.
    // TODO: take care of szargs
    CU_SAFE_CALL(
        cudyLaunch((CUDYfunction) kernel_func, gridDimX, gridDimY, gridDimZ,
                   blockDimX, blockDimY, blockDimZ, sharedMemBytes, args,
                   cuda_context->getSecondaryStream(), NULL));

    return 0;
  }

  void Materialize() {
    if (!start) {
      THROW("CUBIN is not initialized properly");
    }

    // Find all kernels defined by the current ELF image.
    Elf *e = NULL;
    try {
      // Setup ELF version.
      if (elf_version(EV_CURRENT) == EV_NONE) {
        THROW("Cannot initialize ELF library: " << elf_errmsg(-1));
      }

      // First, load input ELF.
      if ((e = elf_memory(start, size)) == 0) {
        THROW("elf_memory() failed for " << (void *)start << ": "
                                         << elf_errmsg(-1));
      }

      // Load executable header.
      GElf_Ehdr ehdr;
      if (!gelf_getehdr(e, &ehdr)) {
        THROW("gelf_getehdr() failed for " << (void *)start << ": "
                                           << elf_errmsg(-1));
      }

      // Determine CUBIN compute capability.
      virt_cc = (ehdr.e_flags & 0xff0000) >> 16;
      real_cc = ehdr.e_flags & 0x0000ff;

      VERBOSE("Cubin " << (void *)start << " is compute_" << virt_cc << "/sm_"
                       << real_cc << "\n");

      // Get sections names section index.
      size_t shstrndx;
      if (elf_getshdrstrndx(e, &shstrndx)) {
        THROW("elf_getshdrstrndx() failed for " << (void *)start << ": "
                                                << elf_errmsg(-1));
      }

      // Locate the symbol table.
      Elf_Scn *scn = elf_nextscn(e, NULL);
      for (int i = 1; scn != NULL; scn = elf_nextscn(e, scn), i++) {
        // Get section header.
        GElf_Shdr shdr;
        if (!gelf_getshdr(scn, &shdr)) {
          THROW("gelf_getshdr() failed for " << (void *)start << ": "
                                             << elf_errmsg(-1));
        }

        // If section is not a symbol table:
        if (shdr.sh_type != SHT_SYMTAB)
          continue;

        // Load symbols.
        Elf_Data *data = elf_getdata(scn, NULL);
        if (!data) {
          THROW("Expected .nv_fatbin data section for " << start);
        }
        if (shdr.sh_size && !shdr.sh_entsize) {
          THROW("Cannot get the number of symbols for " << start);
        }
        int nsymbols = 0;
        if (shdr.sh_size)
          nsymbols = shdr.sh_size / shdr.sh_entsize;
        int strndx = shdr.sh_link;
        for (int i = 0; i < nsymbols; i++) {
          GElf_Sym sym;
          if (!gelf_getsym(data, i, &sym)) {
            THROW("gelf_getsym() failed for " << (void *)start << ": "
                                              << elf_errmsg(-1));
          }
          char *name = elf_strptr(e, strndx, sym.st_name);
          if (!name) {
            THROW("Cannot get the name of " << i << "-th symbol for "
                                            << (void *)start << ": "
                                            << elf_errmsg(-1));
          }
          if (!strncmp(name, ".text.", strlen(".text."))) {
            string kernelName = name + strlen(".text.");
            kernelsNames.push_back(kernelName);
            VERBOSE("Found kernel " << kernelName.c_str() << "\n");
          }
        }

        elf_end(e);
        break;
      }
      materialized = true;
    }
    catch (...) {
      if (e)
        elf_end(e);
      throw;
    }
  }
};

class CubinsIndex : public map<string, map<char, map<char, Cubin *> > > {
  vector<Cubin> cubins;

public:

  // Index the specified CUBINs array in entire collection.
  void Merge(const vector<Cubin> &cubins) {
    size_t offset = this->cubins.size();
    this->cubins.resize(offset + cubins.size());
    for (int i = 0, e = cubins.size(); i != e; i++) {
      const Cubin &cubin = cubins[i];
      this->cubins[offset + i] = cubin;
      const vector<string> &kernelsNames =
          this->cubins[offset + i].getKernelsNames();
      for (int ii = 0, ee = kernelsNames.size(); ii != ee; ii++) {
        const string kernelName = kernelsNames[ii];
        char real_cc = (char) cubin.getRealCC();
        char virt_cc = (char) cubin.getVirtualCC();
        this->operator[](kernelName)[real_cc][virt_cc] =
            &this->cubins[offset + i];
      }
    }
  }

  // Merge entire CUBINs index with another one.
  void Merge(const CubinsIndex &other) { Merge(other.cubins); }

  // Launch the most suitable kernel with the specified name
  // from	the entire kernels index.
  int LaunchKernel(const char *kernelName, unsigned int gridDimX,
                   unsigned int gridDimY, unsigned int gridDimZ,
                   unsigned int blockDimX, unsigned int blockDimY,
                   unsigned int blockDimZ, unsigned int sharedMemBytes,
                   CUstream hStream, void *args, size_t szargs) {
    // Determine the current device compute capability (CC).
    CUdevice device;
    int major, minor;
    CU_SAFE_CALL(cuCtxGetDevice(&device));
    CU_SAFE_CALL(cuDeviceComputeCapability(&major, &minor, device));

    // Find CUBIN with the same real CC and the closest virtual CC
    // to the current device CC.
    int real_cc = major * 10 + minor;
    int virt_cc = major * 10;
    map<char, Cubin *> cubins = this->operator[](kernelName)[real_cc];
    if (!cubins.size()) {
      THROW("Cannot find CUBINs containing \""
            << kernelName << "\" kernel for sm_" << real_cc);
    }
    Cubin *cubin = NULL;
    {
      // First, check if we have a CUBIN, which CC exactly matches
      // the required one.
      map<char, Cubin *>::iterator f = cubins.find(virt_cc);
      if (f != cubins.end()) {
        cubin = f->second;
      } else {
        // If no exact match, find the closest.
        int max_virt_cc = 0;
        for (map<char, Cubin *>::iterator i = cubins.begin(), e = cubins.end();
             i != e; i++) {
          if ((i->first <= virt_cc) && (i->first > max_virt_cc))
            cubin = i->second;
        }
      }
    }
    if (!cubin) {
      THROW("Invalid CUBIN containing \"" << kernelName << "\" for compute_"
                                          << virt_cc << "/sm_" << real_cc);
    }
    return cubin->LaunchKernel(kernelName, gridDimX, gridDimY, gridDimZ,
                               blockDimX, blockDimY, blockDimZ, sharedMemBytes,
                               hStream, args, szargs);
  }
};

class Fatbin {
  CubinsIndex cubinsIndex;

  vector<char> binary;

  const char *filename;

  bool materialized;

public:

  const CubinsIndex &getCubinsIndex() {
    if (!materialized)
      Materialize();

    return cubinsIndex;
  }

  Fatbin() : materialized(false) {}

  Fatbin(const char *filename) : filename(filename), materialized(false) {}

  void Materialize() {
    if (!filename) {
      THROW("Fatbin is not initialized properly");
    }

    int fd = -1;
    Elf *e = NULL;
    bool found = false;
    char *data;
    size_t size = 0;
    try {
      // Setup ELF version.
      if (elf_version(EV_CURRENT) == EV_NONE) {
        THROW("Cannot initialize ELF library: " << elf_errmsg(-1));
      }

      // First, load input ELF.
      if ((fd = open(filename, O_RDONLY)) < 0) {
        THROW("Cannot open file " << filename);
      }
      if ((e = elf_begin(fd, ELF_C_READ, e)) == 0) {
        THROW("Cannot read ELF image from " << filename << ": "
                                            << elf_errmsg(-1));
      }

      // Get sections names section index.
      size_t shstrndx;
      if (elf_getshdrstrndx(e, &shstrndx)) {
        THROW("elf_getshdrstrndx() failed for " << filename << ": "
                                                << elf_errmsg(-1));
      }

      // Locate the fatbin section.
      Elf_Scn *scn = elf_nextscn(e, NULL);
      for (int i = 1; scn != NULL; scn = elf_nextscn(e, scn), i++) {
        // Get section header.
        GElf_Shdr shdr;
        if (!gelf_getshdr(scn, &shdr)) {
          THROW("gelf_getshdr() failed for " << filename << ": "
                                             << elf_errmsg(-1));
        }

        char *name = elf_strptr(e, shstrndx, shdr.sh_name);
        if (strcmp(name, ".nv_fatbin"))
          continue;

        Elf_Data *scn_data = elf_getdata(scn, NULL);
        if (!scn_data) {
          THROW("Expected .nv_fatbin data section for " << filename);
        }

        // Copy fatbin binary data to the local vector.
        data = (char *)scn_data->d_buf;
        size = scn_data->d_size;
        binary.assign(data, data + size);
        data = &binary[0];

        found = true;
        break;

      }
      if (!found) {
        THROW("Cannot find .nv_fatbin section in " << filename);
      }
      elf_end(e);
      close(fd);
    }
    catch (...) {
      if (e)
        elf_end(e);
      if (fd != -1)
        close(fd);
      throw;
    }

    // Count ELF images inside cubin.
    size_t ncubins = 0;
    for (int offset = 0; offset < size - 2; offset++) {
      if (strncmp((char *)data + offset, ELFMAG, SELFMAG))
        continue;

      ncubins++;
    }

    // Find all ELF images inside fatbin.
    vector<Cubin> cubins;
    cubins.resize(ncubins);
    for (int offset = 0, icubin = 0; offset < size - 2; offset++) {
      if (strncmp((char *)data + offset, ELFMAG, SELFMAG))
        continue;

      cubins[icubin++] = Cubin((char *)data + offset, size - offset);
      VERBOSE("Found cubin @ " << filename << "/.nv_fatbin ~ "
                               << (void *)((char *)data + offset) << "\n");
    }
    cubinsIndex.Merge(cubins);

    materialized = true;
  }
};

struct KernelConfig {
  vector<char> arguments;
  dim3 gridDim, blockDim;
  size_t sharedMem;
};

static KernelConfig currKernelConfig;

static map<const char *, string> kernels;

extern "C" {

/*void** __cudaRegisterFatBinary(void* fatCubinHandle)
{
	return NULL;
}*/

void __cudaRegisterFunction(void **fatCubinHandle, const char *hostFun,
                            char *deviceFun, const char *deviceName,
                            int thread_limit, uint3 *tid, uint3 *bid,
                            dim3 *bDim, dim3 *gDim, int *wSize) {
  kernels[hostFun] = deviceName;

  typedef int(*__cudaRegisterFunction_t)(void **, const char *, char *,
                                         const char *, int, uint3 *, uint3 *,
                                         dim3 *, dim3 *, int *);
  static __cudaRegisterFunction_t __cudaRegisterFunction_;
  static int __cudaRegisterFunction_init = 0;
  if (!__cudaRegisterFunction_init) {
    void *handle = dlopen("libcudart.so", RTLD_LAZY);
    __cudaRegisterFunction_ =
        (__cudaRegisterFunction_t) dlsym(handle, "__cudaRegisterFunction");
    __cudaRegisterFunction_init = 1;
  }
  __cudaRegisterFunction_(fatCubinHandle, hostFun, deviceFun, deviceName,
                          thread_limit, tid, bid, bDim, gDim, wSize);

  VERBOSE("Registered kernel function " << (void *)hostFun << " -> "
                                        << deviceName << "\n");
}

/*void __cudaRegisterVar(void** fatCubinHandle, char* hostVar,
	char* deviceAddress, const char* deviceName,
	int ext, int size, int constant, int global)
{
}*/

/*void __cudaRegisterTexture(void** fatCubinHandle, const void* hostVar,
	const void** deviceAddress, const char* deviceName,
	int dim, int norm, int ext)
{
}*/

int cudaFuncSetCacheConfig(void *func, int config) {
  // TODO: Have uberkerns for every possible setting of cache config?
  return 0;
}

int cudaConfigureCall(dim3 gridDim, dim3 blockDim, size_t sharedMem,
                      void *stream) {
  // We ignore stream setting, because KernelGen will launch
  // kernel in its own secondary stream.
  currKernelConfig.gridDim = gridDim;
  currKernelConfig.blockDim = blockDim;
  currKernelConfig.sharedMem = sharedMem;

  return 0;
}

int cudaSetupArgument(const void *arg, size_t size, size_t offset) {
  size_t currSize = currKernelConfig.arguments.size();
  if (currSize < offset + size)
    currKernelConfig.arguments.resize(offset + size);
  memcpy(&currKernelConfig.arguments[0] + offset, arg, size);

  return 0;
}

int cudaLaunch(char *entry) {
  // Determine the executable/dynamic filename fatCubin entry
  // belongs to.
  Dl_info info;
  dladdr((void *)entry, &info);
  if (!info.dli_fname) {
    THROW("Cannot determine which file fatbin " << entry << " belongs to");
  }
  const char *filename = info.dli_fname;

  static map<string, Fatbin> fatbins;
  static CubinsIndex cubinsIndex;

  // Parse the fatbin and include found kernels
  // to the global index of known kernels, if not already done.
  if (fatbins.find(filename) == fatbins.end()) {
    Fatbin &fatbin = fatbins[filename];
    fatbin = Fatbin(filename);
    cubinsIndex.Merge(fatbin.getCubinsIndex());
  }

  // Find the requested kernel in index.
  string kernelName = kernels[entry];
  if (kernelName == "") {
    THROW("Cannot find kernel name for address " << entry << " in \""
                                                 << filename << "\"");
  }
  VERBOSE("Launching kernel \"" << kernelName.c_str() << "\" @ "
                                << (void *)entry << "\n");

  size_t szarguments = currKernelConfig.arguments.size();
  return cubinsIndex.LaunchKernel(
      kernelName.c_str(), currKernelConfig.gridDim.x,
      currKernelConfig.gridDim.y, currKernelConfig.gridDim.z,
      currKernelConfig.blockDim.x, currKernelConfig.blockDim.y,
      currKernelConfig.blockDim.z, currKernelConfig.sharedMem,
      cuda_context->getSecondaryStream(), &currKernelConfig.arguments[0],
      szarguments);
}

int cudaDeviceSynchronize() {
  CU_SAFE_CALL(cuStreamSynchronize(cuda_context->getSecondaryStream()));
  return 0;
}

}

}
