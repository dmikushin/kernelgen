//===- HostCall.cpp - API for invoking host functions from GPU code -------===//
//
//     KernelGen -- A prototype of LLVM-based auto-parallelizing Fortran/C
//        compiler for NVIDIA GPUs, targeting numerical modeling code.
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements API for invoking host functions from GPU code.
//
//===----------------------------------------------------------------------===//

#include "Runtime.h"
#include "Util.h"

#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/DataLayout.h"

#include <dlfcn.h>
#include <errno.h>
#include <ffi.h>
#include <signal.h>
#include <sys/mman.h>
#include <sys/types.h>
#include <unistd.h>

using namespace kernelgen;
using namespace kernelgen::bind::cuda;
using namespace kernelgen::runtime;
using namespace llvm;
using namespace std;

static ffi_type *ffiTypeFor(Type *Ty) {
  switch (Ty->getTypeID()) {
  case Type::VoidTyID:
    return &ffi_type_void;
  case Type::IntegerTyID:
    switch (cast<IntegerType>(Ty)->getBitWidth()) {
    case 8:
      return &ffi_type_sint8;
    case 16:
      return &ffi_type_sint16;
    case 32:
      return &ffi_type_sint32;
    case 64:
      return &ffi_type_sint64;
    }
  case Type::FloatTyID:
    return &ffi_type_float;
  case Type::DoubleTyID:
    return &ffi_type_double;
  case Type::PointerTyID:
    return &ffi_type_pointer;
  default:
    // TODO: Support other types such as StructTyID, ArrayTyID, OpaqueTyID, etc.
    THROW("Type could not be mapped for use with libffi.");
  }
  return NULL;
}

struct mmap_t {
  char *addr, *buffer;
  size_t size, align;
};

static list<struct mmap_t> mmappings;

static Kernel *activeKernel;
static StructType *activeStructTy;

static unsigned NumArgs;
static std::vector<ffi_type *> args;
static SmallVector<void *, 16> values;
static struct sigaction sa_new, sa_old;
static void *params;

static long szpage = -1;

// SIGSEGV signal handler to catch accesses to GPU memory.
static void sighandler(int code, siginfo_t *siginfo, void *ucontext) {
  // Check if address is valid on GPU.
  void *addr = siginfo->si_addr;

  void *base;
  size_t size;
  int err = cuMemGetAddressRange(&base, &size, addr);
  if (err)
    THROW("Not a GPU memory: " << addr);

  size_t align = (size_t) base % szpage;

  // Copy device memory to host buffer.
  char* buffer = new char[size];
  // Not used, because the corresponding cuMemHostUnregister hangs.
  // CU_SAFE_CALL(cuMemHostRegister(buffer, size, 0));
  CU_SAFE_CALL(cuMemcpyDtoHAsync(buffer, base, size,
                                 cuda_context->getSecondaryStream()));

  void *map = mmap((char *)base - align, size + align, PROT_READ | PROT_WRITE,
                   MAP_PRIVATE | MAP_ANONYMOUS | MAP_FIXED, -1, 0);
  if (map == (void *)-1)
    THROW("Cannot map memory onto " << base << " + " << size);

  mmap_t mmap;
  mmap.buffer = buffer;
  mmap.addr = (char*)map;
  mmap.size = size;
  mmap.align = align;
  mmappings.push_back(mmap);

  VERBOSE(Verbose::DataIO << "Mapped memory " << map << "(" << base << " - "
                          << align << ") + " << size << "\n"
                          << Verbose::Default);

  CU_SAFE_CALL(cuStreamSynchronize(cuda_context->getSecondaryStream()));
  memcpy(base, buffer, size);
}

typedef void (*func_t)();

void kernelgen_hostcall(Kernel *kernel, FunctionType *FTy, StructType *StructTy,
                        void *params_) {
  if (activeKernel)
    THROW("There is already an unfinished hostcall in progress");

  params = params_;

  // Compile native kernel, if there is source code and not already compiled.
  KernelFunc *func = &kernel->target[KERNELGEN_RUNMODE_NATIVE].binary;
  if (!*func)
    *func = Compile(KERNELGEN_RUNMODE_NATIVE, kernel);

  Dl_info info;
  if (settings.getVerboseMode() & Verbose::Hostcall) {
    if (dladdr((void *)*func, &info))
      VERBOSE("Host function call " << info.dli_sname << "\n");
    else
      VERBOSE("Host kernel call " << (void *)*func << "\n");
  }

  DataLayout *DL = new DataLayout(kernel->module);

  if (szpage == -1)
    szpage = sysconf(_SC_PAGESIZE);

  activeKernel = kernel;
  activeStructTy = StructTy;

  // Skip first two fields, that are FunctionType and
  // StructureType itself, respectively.
  unsigned ArgBytes = 0;
  NumArgs = StructTy->getNumElements() - 2;

  // Also skip the last field, if return value is not void.
  // In this case the last field would be the return value
  // buffer itself that is neither function argument, nor
  // return value buffer pointer.
  // So, without first two fileds and without last field
  // NumArgs would only cover pointers to function arguments
  // and a pointer to return value buffer.
  Type *RetTy = FTy->getReturnType();
  if (!RetTy->isVoidTy())
    NumArgs--;

  args.resize(NumArgs);
  for (int i = 0; i < NumArgs; i++) {
    Type *ArgTy = StructTy->getElementType(i + 2);
    args[i] = ffiTypeFor(ArgTy);
    ArgBytes += DL->getTypeStoreSize(ArgTy);
  }

  const StructLayout *layout = DL->getStructLayout(StructTy);
  SmallVector<uint8_t, 128> ArgData;
  ArgData.resize(ArgBytes);
  uint8_t *ArgDataPtr = ArgData.data();
  values.resize(NumArgs);
  for (int i = 0; i < NumArgs; i++) {
    Type *ArgTy = StructTy->getElementType(i + 2);
    int offset = layout->getElementOffset(i + 2);
    size_t size = DL->getTypeStoreSize(ArgTy);
    void **address = (void **)((char *)params + offset);
    memcpy(ArgDataPtr, address, size);
    values[i] = ArgDataPtr;
    ArgDataPtr += size;
    if (ArgTy->isPointerTy()) {
      // If pointer corresponds to device memory,
      // use the host memory instead:
      // figure out the allocated device memory range
      // and shadow it with host memory mapping.
      void *base;
      size_t size;
      int err = cuMemGetAddressRange(&base, &size, *address);
      if (err == CUDA_ERROR_NOT_FOUND) {
          VERBOSE(Verbose::DataIO
                  << "Directly accessible address, no mapping: " << "("
                  << *address << ") + " << size << "\n" << Verbose::Default);
      }
      else if (!err) {
        size_t align = (size_t) base % szpage;

        list<struct mmap_t>::iterator mmapping = mmappings.begin();
        for (; mmapping != mmappings.end(); mmapping++) {
          if (mmapping->addr == (char *)base - align)
            break;
        }

        if (mmapping == mmappings.end()) {
          // Copy device memory to host buffer.
          char* buffer = new char[size];
          // Not used, because the corresponding cuMemHostUnregister hangs.
          // CU_SAFE_CALL(cuMemHostRegister(buffer, size, 0));
          CU_SAFE_CALL(cuMemcpyDtoHAsync(buffer, base, size,
                                         cuda_context->getSecondaryStream()));

          // Map host memory with the same address and size
          // device memory has.
          void *map =
              mmap((char *)base - align, size + align, PROT_READ | PROT_WRITE,
                   MAP_PRIVATE | MAP_ANONYMOUS | MAP_FIXED, -1, 0);
          if (map == (void *)-1)
            THROW("Cannot map host memory onto " << base << " + " << size);

          // Track the mapped memory in list of mappings,
          // to synchronize them after the hostcall finishes.
          struct mmap_t mmapping;
          mmapping.buffer = buffer;
          mmapping.addr = (char*)map;
          mmapping.size = size;
          mmapping.align = align;
          mmappings.push_back(mmapping);

          VERBOSE(Verbose::DataIO
                  << "Mapped memory " << (void *)((char *)base - align) << "("
                  << base << " - " << align << ") + " << size << "\n"
                  << Verbose::Default);
        } else {
          // Remap the existing mapping, if it is required
          // to be larger.
          if (size + align > mmapping->size + mmapping->align) {
            // Copy device memory to host buffer.
            char* buffer = new char[size];
            // Not used, because the corresponding cuMemHostUnregister hangs.
            // CU_SAFE_CALL(cuMemHostRegister(buffer, size, 0));
            CU_SAFE_CALL(cuMemcpyDtoHAsync(buffer, base, size,
                                           cuda_context->getSecondaryStream()));

            void *remap =
                mremap((char *)base - align, mmapping->size + mmapping->align,
                       size + align, 0);
            if (remap == (void *)- 1)
              THROW("Cannot map host memory onto " << base << " + " << size);

            // Store new size & align.
            mmapping->buffer = buffer;
            mmapping->size = size;
            mmapping->align = align;

            VERBOSE(Verbose::DataIO
                    << "Mapped memory " << (void *)((char *)base - align) << "("
                    << base << " - " << align << ") + " << size << "\n"
                    << Verbose::Default);
          }
        }
      } else {
        THROW("Error in cuMemGetAddressRange: " << err);
      }
    }
  }

  ffi_type *rtype = NULL;
  rtype = ffiTypeFor(RetTy);
  if (!RetTy->isVoidTy())
    NumArgs--;

  ffi_cif cif;
  if (ffi_prep_cif(&cif, FFI_DEFAULT_ABI, NumArgs, rtype, &args[0]) != FFI_OK)
    THROW("Error in ffi_prep_cif");

  // Set address of return value buffer. Also mmapped
  // transparently in the same way as pointer arguments.
  void *ret = NULL;
  if (!RetTy->isVoidTy()) {
    NumArgs++;
    Type *ArgTy = StructTy->getElementType(StructTy->getNumElements() - 2);
    int offset = layout->getElementOffset(StructTy->getNumElements() - 2);
    ret = *(void **)((char *)params + offset);
  }

  // Register SIGSEGV signal handler to catch
  // accesses to GPU memory and remember the original handler.
  sa_new.sa_handler = SIG_DFL;
  sa_new.sa_sigaction = sighandler;
  sigemptyset(&sa_new.sa_mask);
  sa_new.sa_flags = SA_SIGINFO;
  if (sigaction(SIGSEGV, &sa_new, &sa_old) == -1)
    THROW("Error in sigaction " << errno);

  VERBOSE(Verbose::Hostcall << "Starting hostcall to " << (void *)*func << "\n"
                            << Verbose::Default);

  // Synchronize pending mmapped data transfers.
  CU_SAFE_CALL(cuStreamSynchronize(cuda_context->getSecondaryStream()));

  // Copy memory from host buffers to host mapped memory.
  for (list<struct mmap_t>::iterator i = mmappings.begin(), e = mmappings.end();
       i != e; i++) {
    // TODO: transfer hangs when size is not a multiplier of some power of two.
    // Currently the guess is 16. So, do we need to pad all arrays to 16?..
    struct mmap_t mmap = *i;
    char* buffer = mmap.buffer;
    size_t size = mmap.size;
    void* base = mmap.addr + mmap.align;
    memcpy(base, buffer, size);
  }

  ffi_call(&cif, (func_t) * func, ret, values.data());

  VERBOSE(Verbose::Hostcall << "Finishing hostcall to " << (void *)*func << "\n"
                            << Verbose::Default);

  // Unregister SIGSEGV signal handler and restore the
  // original handler.
  if (sigaction(SIGSEGV, &sa_old, &sa_new) == -1)
    THROW("Error in sigaction " << errno);

  kernelgen_hostcall_memsync();

  activeKernel = NULL;

  VERBOSE(Verbose::Hostcall << "Finished hostcall handler\n"
                            << Verbose::Default);
}

void kernelgen_hostcall_memsync() {
  // Don't do anything, of no active kernel.
  // Means somebody just launched us to ensure all mmaped GPU
  // data is synchronized.
  if (!activeKernel)
    return;

  Kernel *kernel = activeKernel;
  StructType *StructTy = activeStructTy;

  DataLayout *DL = new DataLayout(kernel->module);

  const StructLayout *layout = DL->getStructLayout(StructTy);

  // Refresh arguments and return value pointer.
  for (int i = 0; i < NumArgs; i++) {
    Type *ArgTy = StructTy->getElementType(i + 2);
    int offset = layout->getElementOffset(i + 2);
    size_t size = DL->getTypeStoreSize(ArgTy);
    void *address = (void *)((char *)params + offset);
    memcpy(address, values[i], size);
  }

  // Copy data back from host-mapped memory to device.
  for (list<struct mmap_t>::iterator i = mmappings.begin(), e = mmappings.end();
       i != e; i++) {
    struct mmap_t mmap = *i;
    size_t size = mmap.size;

    // Copy modified host-mapped memory to buffer.
    memcpy(mmap.buffer, (char *)mmap.addr + mmap.align, size);
    
    // Unmap the host memory mapping.
    int err = munmap(mmap.addr, mmap.size + mmap.align);
    if (err == -1)
      THROW("Cannot unmap memory from " << mmap.addr << " + "
                                        << mmap.size + mmap.align);

    // TODO: transfer hangs when size is not a multiplier of some power of two.
    // Currently the guess is 16. So, do we need to pad all arrays to 16?..
    if (size % 16)
      size -= mmap.size % 16;

    CU_SAFE_CALL(cuMemcpyHtoDAsync((char *)mmap.addr + mmap.align,
                                   mmap.buffer, size,
                                   cuda_context->getSecondaryStream()));
    VERBOSE(Verbose::DataIO
            << "mmap.addr = " << (void*)mmap.addr << ", mmap.align = " << mmap.align
            << ", mmap.size = " << mmap.size << " (" << size << ")\n"
            << Verbose::Default);
  }

  // Synchronize and unmap previously mapped host memory.
  CU_SAFE_CALL(cuStreamSynchronize(cuda_context->getSecondaryStream()));
  for (list<struct mmap_t>::iterator i = mmappings.begin(), e = mmappings.end();
       i != e; i++) {
    struct mmap_t mmap = *i;
    // Not used, because cuMemHostUnregister hangs.
    // CU_SAFE_CALL(cuMemHostUnregister(buffer));
    delete[] mmap.buffer;
  }

  mmappings.clear();
}
