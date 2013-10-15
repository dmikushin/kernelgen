/*
 * Copyright (c) 2012 by Dmitry Mikushin
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 * 
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 */

#include "llvm/Support/Program.h"

#include "Cuda.h"
#include "cuda_dyloader.h"
#include "libasfermi.h"
#include "loader.h"
#include "KernelGen.h"
#include "Runtime.h"

#include <cstring>
#include <elf.h>
#include <fstream>
#include <gelf.h>
#include <iomanip>
#include <iostream>
#include <libelf.h>
#include <link.h>
#include <list>
#include <malloc.h>
#include <memory>
#include <string>
#include <sstream>
#include <vector>

using namespace kernelgen::bind::cuda;
using namespace kernelgen::runtime;
using namespace kernelgen::utils;
using namespace llvm::sys;
using namespace std;

// The maximum number of registers per thread.
#define MAX_REGCOUNT 63

// The register footprint of uberkern loader code itself.
// Engine still should be able to run dynamic kernels with
// smaller footprints, but when loader code is running, this
// number is a must.
#define LOADER_REGCOUNT 7

// Ad extra offset between the end of uberkerel loader code
// and the first dynamic kernel code
#define BASE_EXTRA_OFFSET 1024

// An extra offset between loaded dynamic kernels codes to
// force no caching/prefetching.
#define EXTRA_OFFSET 512

using namespace kernelgen;

struct CUDYloader_t;
 
struct CUDYfunction_t {
  unsigned int szbinary;
  vector<char> binary;

  short regcount;

  CUDYloader_t *loader;

  unsigned int offset;

  // Read CUBIN function from file.
  CUDYfunction_t(CUDYloader_t *loader, const char *name, char *cubin, int regcount)
      : loader(loader), regcount(regcount) {
    // Read cubin from file
    stringstream stream(stringstream::in | stringstream::out |
                        stringstream::binary);
    ifstream f(cubin, ios::in | ios::binary);
    stream << f.rdbuf();
    f.close();
    string content = stream.str();
    if (strncmp(content.c_str(), ELFMAG, SELFMAG)) {
      THROW("Expected ELF magic in the beginning of CUBIN ELF\n");
    }
    Initialize(name, content.c_str(), content.length());
  }

  // Read CUBIN function from memory buffer.
  CUDYfunction_t(CUDYloader_t *loader, const char *name, char *cubin,
                 size_t size)
      : loader(loader), regcount(-1) {
    // Search for ELF magic in cubin. If found, then content
    // is supplied, otherwise - filename.
    stringstream stream(stringstream::in | stringstream::out |
                        stringstream::binary);
    if (strncmp(cubin, ELFMAG, SELFMAG)) {
      THROW("Expected ELF magic in the beginning of CUBIN ELF\n");
    }
    Initialize(name, cubin, size);
  }

  void Initialize(const char *name, const char *content, size_t size) {
    // Build kernel name as it should appear in cubin ELF.
    stringstream namestream;
    namestream << ".text." << name;
    string elfname = namestream.str();

    // Extract kernel details: regcount, opcodes and their size.
    // Method: walk thorough the cubin using ELF tools and dump
    // details of the first kernel found (section starting with
    // ".text.").
    Elf *e = NULL;
    try {
      e = elf_memory((char *)content, size);
      size_t shstrndx;
      if (elf_getshdrstrndx(e, &shstrndx)) {
        THROW("Cannot get the CUBIN/ELF strings section header index",
              CUDA_ERROR_INVALID_SOURCE);
      }
      Elf_Scn *scn = NULL;
      while ((scn = elf_nextscn(e, scn)) != NULL) {
        // Get section name.
        GElf_Shdr shdr;
        if (gelf_getshdr(scn, &shdr) != &shdr) {
          THROW("Cannot load the CUBIN/ELF section header",
                CUDA_ERROR_INVALID_SOURCE);
        }
        char *name = NULL;
        if ((name = elf_strptr(e, shstrndx, shdr.sh_name)) == NULL) {
          THROW("Cannot load the CUBIN/ELF section name",
                CUDA_ERROR_INVALID_SOURCE);
        }

        if (elfname != name)
          continue;

        // Extract regcount out of 24 bits of section info.
        if (regcount == -1)
          regcount = shdr.sh_info >> 24;
        else
          regcount = max(regcount, (short)(shdr.sh_info >> 24));

        // Extract binary opcodes and size.
        szbinary = shdr.sh_size;
        binary.resize(szbinary);
        memcpy(&binary[0], content + shdr.sh_offset, szbinary);

        // For asynchronous data transfers to work, need to
        // pin memory for binary content.
        CUresult cuerr = cuMemHostRegister(&binary[0], szbinary, 0);
        if (cuerr != CUDA_SUCCESS) {
          if (cuerr != CUDA_ERROR_HOST_MEMORY_ALREADY_REGISTERED)
            THROW("Cannot pin memory for CUBIN binary content", cuerr);

          // We are fine, if memory is already registered.
        }

        break;
      }
    }
    catch (CUresult cuerr) {
      if (e)
        elf_end(e);
      throw cuerr;
    }
    elf_end(e);

    if (regcount == -1)
      THROW("Cannot find kernel " << name << " in " << content,
            CUDA_ERROR_INVALID_SOURCE);

    VERBOSE(name << ": regcount = " << regcount << ", size = " << szbinary
                 << "\n");
  }

  CUDYfunction_t(CUDYloader_t *loader, char *opcodes, size_t nopcodes,
                 int regcount)
      : loader(loader), szbinary(8 * nopcodes), regcount(regcount) {
    // Copy binary.
    binary.resize(szbinary);
    memcpy(&binary[0], opcodes, szbinary);

    // For asynchronous data transfers to work, need to
    // pin memory for binary content.
    CUresult cuerr = cuMemHostRegister(&binary[0], szbinary, 0);
    if (cuerr != CUDA_SUCCESS) {
      if (cuerr != CUDA_ERROR_HOST_MEMORY_ALREADY_REGISTERED)
        THROW("Cannot pin memory for CUBIN binary content", cuerr);

      // We are fine, if memory is already registered.
    }
  }

  ~CUDYfunction_t() {
    // Unpin pinned memory for binary.
    CUresult cuerr = cuMemHostUnregister(&binary[0]);
    if (cuerr != CUDA_SUCCESS) {
      if (cuerr != CUDA_ERROR_HOST_MEMORY_NOT_REGISTERED)
        THROW("Cannot unpin memory for CUBIN binary content", cuerr);

      // We are fine, if memory is already unregistered.
    }

    // Unpin pinned memory for offset.
    cuerr = cuMemHostUnregister(&offset);
    if (cuerr != CUDA_SUCCESS) {
      if (cuerr != CUDA_ERROR_HOST_MEMORY_NOT_REGISTERED)
        THROW("Cannot unpin memory for the dynamic kernel pool offset", cuerr);

      // We are fine, if memory is already unregistered.
    }
  }
};

struct CUDYloader_t {
  int offset;
  int capacity;

  // Defines the name of host cubin, if
  string host_cubin;

  CUmodule module;
  CUfunction loader, entry[MAX_REGCOUNT + 1];
  
  uint32_t lepc;
  
  struct UberkernArgs {
    void* userPtr1;
    void* userPtr2;

    // Dynamic kernel binary size.
    uint32_t szbinary;

    // Dynamic kernel binary to load (if not yet loaded).
    CUdeviceptr *binary;
  
    uint32_t command;
    uint32_t address;
    
    UberkernArgs(int capacity) : 
      userPtr1(NULL), userPtr2(NULL), szbinary(0), binary(NULL), command(0), address(0) {

      // Note command value is initialized with ZERO, so on the first
      // launch uberkern will simply report LEPC and exit.
      
      // Allocate space for dynamic kernel binary.
      CUresult curesult = cuMemAlloc((void**)&binary, capacity);
      if (curesult != CUDA_SUCCESS)
        THROW("Cannot allocate the uberkern memory buffer " << curesult,
              curesult);
    }
    
    ~UberkernArgs() {
      if (binary) {
        CUresult curesult = cuMemFree(binary);
        if (curesult != CUDA_SUCCESS)
          THROW("Cannot free the uberkern memory buffer " << curesult,
                curesult);
      }
    }
  }
  __attribute__((packed, aligned(4)));
      
  UberkernArgs uberkernArgs;
  
  list<CUDYfunction_t *> functions;

  CUDYloader_t(int capacity, string host_cubin = "")
      : offset(BASE_EXTRA_OFFSET), capacity(BASE_EXTRA_OFFSET + capacity * 8),
        host_cubin(host_cubin), uberkernArgs(BASE_EXTRA_OFFSET + capacity * 8) {
        
    // The number of items in uberkern loader template code (defined in loader.h).
    int ntokens = sizeof(uberkern) / sizeof(const char *);

    // Select the arguments starting offset: differs between Fermi and Kepler.
    uint16_t offset = 0;
    if (cuda_context->getSubarchMajor() == 2) {
      offset = 0x20;
    }
    else if (cuda_context->getSubarchMajor() == 3) {
      offset = 0x140;
    }
    else
      THROW("KernelGen dyloader is not tested with targets >= sm_3x");

    stringstream stream;
    {
      string szbinary; {
        stringstream stream;
        uint16_t value = offset + offsetof(UberkernArgs, szbinary);
        stream << "0x" << hex << value << dec;
        szbinary = stream.str();
      }
      string binarylo; {
        stringstream stream;
        uint16_t value = offset + offsetof(UberkernArgs, binary);
        stream << "0x" << hex << value << dec;
        binarylo = stream.str();
      }
      string binaryhi; {
        stringstream stream;
        uint16_t value = offset + offsetof(UberkernArgs, binary) + 4;
        stream << "0x" << hex << value << dec;
        binaryhi = stream.str();
      }
      string command; {
        stringstream stream;
        uint16_t value = offset + offsetof(UberkernArgs, command);
        stream << "0x" << hex << value << dec;
        command = stream.str();
      }
      string address; {
        stringstream stream;
        uint16_t value = offset + offsetof(UberkernArgs, address);
        stream << "0x" << hex << value << dec;
        address = stream.str();
      }

      stream << setfill('0');
      for (int i = 0, ilines = 0; i < ntokens; i++) {
        string line = uberkern[i];

        if (line[0] == '!') {
          stream << "\t\t" << line << endl;
          continue;
        }

        // Replace uberkernel parameters placeholders with actual addresses.
        for (size_t index = line.find("$SZBINARY", 0);
             index = line.find("$SZBINARY", index); index++) {
          if (index == string::npos)
            break;
          line.replace(index, string("$SZBINARY").length(), szbinary);
        }
        for (size_t index = line.find("$BINARYHI", 0);
             index = line.find("$BINARYHI", index); index++) {
          if (index == string::npos)
            break;
          line.replace(index, string("$BINARYHI").length(), binaryhi);
        }
        for (size_t index = line.find("$BINARYLO", 0);
             index = line.find("$BINARYLO", index); index++) {
          if (index == string::npos)
            break;
          line.replace(index, string("$BINARYLO").length(), binarylo);
        }
        for (size_t index = line.find("$COMMAND", 0);
             index = line.find("$COMMAND", index); index++) {
          if (index == string::npos)
            break;
          line.replace(index, string("$COMMAND").length(), command);
        }
        for (size_t index = line.find("$ADDRESS", 0);
             index = line.find("$ADDRESS", index); index++) {
          if (index == string::npos)
            break;
          line.replace(index, string("$ADDRESS").length(), address);
        }

        // Output the specified number of NOPs in place of $BUF.
        if (line == "$BUF") {
          for (unsigned int j = 0; j < capacity; j++) {
            stream << "/* 0x" << setw(4) << ilines * 8 << " */\tNOP;" << endl;
            ilines++;
          }
          continue;
        }

        stream << "/* 0x" << setw(4) << ilines * 8 << " */\t" << line << ";"
               << endl;
        ilines++;
      }
    }

    // Output additional entry points with all possible
    // reg counts.
    for (unsigned int regcount = 0; regcount < MAX_REGCOUNT + 1; regcount++) {
      stream << "\t\t!Kernel uberkern" << regcount << endl;
      stream << "\t\t!RegCount " << regcount << endl;
      stream << "\t\t!Param 256 1" << endl;
      //stream << "\t\t!Shared 256" << endl;
      //stream << "/* 0x0000 */\tMOV R0, c[0x0][0x" << hex << (offset +
      //          offsetof(UberkernArgs, command)) << dec << "];" << endl;
      stream << "/* 0x0000 */\tJMP c[0x0][0x" << hex << (offset +
                offsetof(UberkernArgs, command)) << dec << "];" << endl;
      stream << "\t\t!EndKernel" << endl;
    }

    // Duplicate source array.
    string source = stream.str();
    std::vector<char> vsource(source.size() + 1);
    char *csource = (char *)&vsource[0];
    csource[source.size()] = '\0';
    memcpy(csource, source.c_str(), source.size());

    char *cubin = NULL;
    bool moduleLoaded = false;

    try {
      // Emit cubin for the current device architecture.
      size_t size;
      cubin =
          asfermi_encode_cubin(csource, cuda_context->getSubarchMajor() * 10 +
                                            cuda_context->getSubarchMinor(),
                               0, &size);
      if (!cubin)
        THROW("Cannot encode the uberkern into cubin",
              CUDA_ERROR_INVALID_SOURCE);

      if (host_cubin != "") {
        // Save the dyloader cubin to disk.
        TempFile file1 = Temp::getFile("%%%%%%%%.cubin");
        file1.download(cubin, size);

        // Merge dyloader cubin with host cubin.
        TempFile file2 = Temp::getFile("%%%%%%%%.cubin");
        VERBOSE("Merge: " << host_cubin << " " << file1.getName() << " "
                          << file2.getName().c_str() << "\n");
        CUBIN::Merge(host_cubin.c_str(), file1.getName().c_str(),
                     file2.getName().c_str());

        // Align main kernel cubin global data to the virtual memory
        // page boundary.
        CUBIN::AlignData(file2.getName().c_str(), 4096);

        // Replace the dyloader cubin with the resulting cubin.
        free(cubin);
        file2.upload(&cubin, &size);
      }

      // Load binary containing uberkernel to deivce memory.
      CUresult curesult = cuModuleLoadData(&module, cubin);
      if (curesult != CUDA_SUCCESS)
        THROW("Cannot load uberkern module " << curesult, curesult);
      moduleLoaded = true;

      // Load uberkern loader entry point from module.
      curesult = cuModuleGetFunction(&loader, module, "uberkern");
      if (curesult != CUDA_SUCCESS)
        THROW("Cannot load uberkern loader function " << curesult, curesult);

      // Load uberkernel entry points from module.
      for (int i = 0; i < MAX_REGCOUNT + 1; i++) {
        stringstream stream;
        stream << "uberkern" << i;
        string name = stream.str();
        curesult = cuModuleGetFunction(&entry[i], module, name.c_str());
        if (curesult != CUDA_SUCCESS)
          THROW("Cannot load uberkern entry function " << curesult, curesult);
      }

      // Launch uberkernel to fill the LEPC.
      // Note we are always sending 256 Bytes, regardless
      // the actual size of arguments.
      char args[256];      
      memcpy(args, &uberkernArgs, sizeof(UberkernArgs));
      size_t szargs = 256;
      void *params[] = { CU_LAUNCH_PARAM_BUFFER_POINTER, args,
                         CU_LAUNCH_PARAM_BUFFER_SIZE, &szargs,
                         CU_LAUNCH_PARAM_END };
      curesult = cuLaunchKernel(loader, 1, 1, 1, 1, 1, 1, 0, 0, NULL, params);
      if (curesult != CUDA_SUCCESS)
        THROW("Cannot launch the uberkern loader " << curesult, curesult);

      // Synchronize kernel.
      curesult = cuCtxSynchronize();
      if (curesult != CUDA_SUCCESS)
        THROW("Cannot synchronize the uberkern loader " << curesult, curesult);

      // Read the LEPC.
      lepc = 0;
      curesult = cuMemcpyDtoH(&lepc, uberkernArgs.binary, sizeof(int));
      if (curesult != CUDA_SUCCESS)
        THROW("Cannot read the uberkern LEPC value " << curesult, curesult);

      stringstream xlepc;
      xlepc << hex << lepc;
      VERBOSE("LEPC = 0x" << xlepc.str() << "\n");
    }
    catch (CUresult cuerr) {
      if (cubin)
        free(cubin);
      if (moduleLoaded) {
        CUresult curesult = cuModuleUnload(module);
        if (curesult != CUDA_SUCCESS)
          THROW("Cannot unload the uberkern module " << curesult, curesult);
      }
      THROW("Error in dynamic loader", cuerr);
    }
    free(cubin);
  }

  ~CUDYloader_t() {
    // Dispose functions.
    for (list<CUDYfunction_t *>::iterator i = functions.begin(),
                                          ie = functions.end();
         i != ie; i++)
      delete *i;

    CUresult curesult = cuModuleUnload(module);
    if (curesult != CUDA_SUCCESS)
      THROW("Cannot unload the uberkern module " << curesult, curesult);

  }

  CUresult Load(CUDYfunction_t *function, CUstream stream) {
    // Check the dynamic pool has enough free space to
    // incorporate the specified dynamic kernel body.
    if (offset + function->szbinary > capacity) {
      THROW("Insufficient space in the uberkern memory pool",
            CUDA_ERROR_OUT_OF_MEMORY);
    }

    // Set dynamic kernel binary size.
    uberkernArgs.szbinary = function->szbinary;

    // Initialize command value with ONE, so on the next
    // launch uberkern will load dynamic kernel code and exit.
    uberkernArgs.command = 1;

    // Fill the dynamic kernel code BRA target address.
    uberkernArgs.address = offset;

    // Load dynamic kernel binary.
    CUresult curesult = cuMemcpyHtoDAsync((CUdeviceptr) uberkernArgs.binary,
                                 &function->binary[0],
                                 function->szbinary, stream);
    if (curesult != CUDA_SUCCESS)
      THROW("Cannot load the dynamic kernel binary " << curesult, curesult);

    // Synchronize stream.
    curesult = cuStreamSynchronize(stream);
    if (curesult != CUDA_SUCCESS)
      THROW("Cannot synchronize after the dynamic kernel binary loading "
                << curesult,
            curesult);

    // Launch uberkernel to load the dynamic kernel code.
    // Note we are always sending 256 Bytes, regardless
    // the actual size of arguments.
    char args[256];
    memcpy(args, &uberkernArgs, sizeof(UberkernArgs));
    size_t szargs = 256;
    void *params[] = { CU_LAUNCH_PARAM_BUFFER_POINTER, args,
                       CU_LAUNCH_PARAM_BUFFER_SIZE, &szargs,
                       CU_LAUNCH_PARAM_END };
    curesult =
        cuLaunchKernel(loader, 1, 1, 1, 1, 1, 1, 0, stream, NULL, params);
    if (curesult != CUDA_SUCCESS)
      THROW("Cannot launch the uberkern loader " << curesult, curesult);

    // Synchronize stream.
    curesult = cuStreamSynchronize(stream);
    if (curesult != CUDA_SUCCESS)
      THROW("Cannot synchronize the uberkern loader " << curesult, curesult);

    // Store function body offset.
    function->offset = offset;

    // Increment pool offset by the size of kernel binary.
    offset += function->szbinary + EXTRA_OFFSET;

    // Track function for disposal.
    functions.push_back(function);

    return CUDA_SUCCESS;
  }

  CUresult Launch(CUDYfunction_t *function, unsigned int gx, unsigned int gy,
                  unsigned int gz, unsigned int bx, unsigned int by,
                  unsigned int bz, size_t szshmem, void *data, CUstream stream,
                  float *time) {
    // Initialize command value with LEPC, so on the next
    // launch uberkern will load dynamic kernel code and exit.
    // XXX: 0x128 is #BRA of uberkernel loader code - the value
    // may change if loader code gets changed.
    uberkernArgs.command = lepc + 0x128;

    // Fill the dynamic kernel code BRA target address.
    uberkernArgs.address = function->offset;

    CUevent start, stop;
    if (time) {
      CUresult curesult = cuEventCreate(&start, 0);
      if (curesult != CUDA_SUCCESS)
        THROW("Cannot create timer start event " << curesult, curesult);
      curesult = cuEventCreate(&stop, 0);
      if (curesult != CUDA_SUCCESS)
        THROW("Cannot create timer stop event " << curesult, curesult);
      curesult = cuEventRecord(start, stream);
      if (curesult != CUDA_SUCCESS)
        THROW("Cannot record the timer start event " << curesult, curesult);
    }

    // Launch device function.
    // Note we are always sending 256 Bytes, regardless
    // the actual size of arguments.
    uberkernArgs.userPtr1 = ((void**)data)[0];
    uberkernArgs.userPtr2 = ((void**)data)[1];
    char args[256];    
    memcpy(args, &uberkernArgs, sizeof(UberkernArgs));
    size_t szargs = 256;
    void *config[] = { CU_LAUNCH_PARAM_BUFFER_POINTER, args,
                       CU_LAUNCH_PARAM_BUFFER_SIZE, &szargs,
                       CU_LAUNCH_PARAM_END };
    CUresult curesult = cuLaunchKernel(entry[function->regcount],
                                       gx, gy, gz, bx, by, bz,
                                       szshmem, stream, NULL, config);
    if (curesult != CUDA_SUCCESS)
      THROW("Cannot launch the dynamic kernel " << curesult, curesult);

    if (time) {
      curesult = cuEventRecord(stop, stream);
      if (curesult != CUDA_SUCCESS)
        THROW("Cannot record the timer stop event " << curesult, curesult);
      curesult = cuEventSynchronize(stop);
      if (curesult != CUDA_SUCCESS)
        THROW("Cannot synchronize the dynamic kernel " << curesult, curesult);
      curesult = cuEventElapsedTime(time, start, stop);
      if (curesult != CUDA_SUCCESS)
        THROW("Cannot get the timer elapsed time " << curesult, curesult);
      *time *= 1e-3;
    }

    return CUDA_SUCCESS;
  }
};

// Initialize a new instance of CUDA dynamic loader with the
// specified capacity (in 8-byte instructions) in GPU memory
// and optionally attach it to the specified hosting cubin image.
CUresult cudyInit(CUDYloader *loader, int capacity, string host_cubin) {
  *loader = new CUDYloader_t(capacity, host_cubin);
  return CUDA_SUCCESS;
}

// Load kernel function with the specified name from cubin file
// into dynamic loader context.
CUresult cudyLoadCubin(CUDYfunction *function, CUDYloader loader,
                       const char *name, char *cubin, CUstream stream,
                       int regcount) {
  // Create function.
  *function = new CUDYfunction_t(loader, name, cubin, regcount);
  return loader->Load(*function, stream);
}

// Load kernel function with the specified name from memory buffer
// into dynamic loader context.
CUresult cudyLoadCubinData(CUDYfunction *function, CUDYloader loader,
                           const char *name, char *cubin, size_t size,
                           CUstream stream) {
  // Create function.
  *function = new CUDYfunction_t(loader, name, cubin, size);
  return loader->Load(*function, stream);
}

// Load kernel function from the specified assembly opcodes
// into dynamic loader context.
CUresult cudyLoadOpcodes(CUDYfunction *function, CUDYloader loader,
                         char *opcodes, size_t nopcodes, CUstream stream,
                         int regcount) {
  // Create function.
  *function = new CUDYfunction_t(loader, opcodes, nopcodes, regcount);
  return loader->Load(*function, stream);
}

// Launch kernel function through the dynamic loader.
CUresult cudyLaunch(CUDYfunction function, unsigned int gx, unsigned int gy,
                    unsigned int gz, unsigned int bx, unsigned int by,
                    unsigned int bz, size_t szshmem, void *args,
                    CUstream stream, float *time) {
  return function->loader
      ->Launch(function, gx, gy, gz, bx, by, bz, szshmem, args, stream, time);
}

// Dispose the specified CUDA dynamic loader instance.
CUresult cudyDispose(CUDYloader loader) {
  delete loader;
  return CUDA_SUCCESS;
}
