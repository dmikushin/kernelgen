//===- Entry.cpp - KernelGen application entry point ----------------------===//
//
//     KernelGen -- A prototype of LLVM-based auto-parallelizing Fortran/C
//        compiler for NVIDIA GPUs, targeting numerical modeling code.
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements KernelGen application entry point.
//
//===----------------------------------------------------------------------===//

#include "llvm/IR/Constants.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/PassManager.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Bitcode/ReaderWriter.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Program.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/ToolOutputFile.h"

#include "KernelGen.h"

#include "Elf.h"
#include "Util.h"
#include "Runtime.h"
#include "kernelgen_interop.h"

#include <dlfcn.h>
#include <ffi.h>
#include <fstream>
#include <iostream>
#include <cstdlib>

//#include "TrackedPassManager.h"

// Regular main entry.
extern "C" int __regular_main(int argc, char *argv[]);

using namespace kernelgen;
using namespace kernelgen::runtime;
using namespace kernelgen::bind::cuda;
using namespace llvm;
using namespace llvm::sys;
using namespace llvm::sys::fs;
using namespace std;
using namespace util::elf;

// The pool of already loaded kernels.
// After kernel is loaded, we pin it here
// for futher references.
std::map<string, Kernel *> kernelgen::kernels;

// The array contains addresses of globalVatiables
uint64_t *kernelgen::AddressesOfGVars;
int kernelgen::NumOfGVars;

// order of globals in which they were stored in addressesOfGlobalVariables
std::map<llvm::StringRef, uint64_t> kernelgen::orderOfGlobals;

// CUDA runtime context.
// TODO: sort out how to turn it into auto_ptr.
kernelgen::bind::cuda::context *kernelgen::runtime::cuda_context = NULL;

// Base address of GPU dynamic memory pool.
kernelgen_memory_t *kernelgen::runtime::memory_pool;

// Monitoring module and kernel (applicable for some targets).
Module *kernelgen::runtime::monitor_module = NULL;
KernelFunc kernelgen::runtime::monitor_kernel;

// Runtime module (applicable for some targets).
Module *kernelgen::runtime::runtime_module = NULL;

// CUDA device functions module.
Module *kernelgen::runtime::cuda_module = NULL;

// KernelGen plugins.
std::vector<kernelgen_after_ptx_t> kernelgen::runtime::pluginsAfterPTX;
std::vector<kernelgen_after_cubin_t> kernelgen::runtime::pluginsAfterCUBIN;

void load_kernel(Kernel *kernel);

namespace llvm {
  ModulePass *createNVVMReflectPass();
}

static Module* load_module(string Path)
{
  string err;
  std::ifstream tmp_stream(Path.c_str());
  if (!tmp_stream.is_open())
    THROW("Cannot load KernelGen module " << Path << ": " << err);
  tmp_stream.seekg(0, std::ios::end);
  string source = "";
  source.reserve(tmp_stream.tellg());
  tmp_stream.seekg(0, std::ios::beg);

  source.assign(std::istreambuf_iterator<char>(tmp_stream),
                std::istreambuf_iterator<char>());
  tmp_stream.close();

  MemoryBuffer *buffer = MemoryBuffer::getMemBuffer(source);
  LLVMContext &context = getGlobalContext();
  Module* module = ParseBitcodeFile(buffer, context, &err);
  if (!module)
    THROW("Cannot load KernelGen module " << Path << ": " << err);
  return module;
}

int main(int argc, char *argv[], char *envp[]) {
  //tracker = new PassTracker("codegen", NULL, NULL);

  LLVMContext &context = getGlobalContext();

  // Retrieve the regular main entry function prototype out of
  // the internal table.
  Function *regular_main = NULL;
  celf e(argv[0], "");
  //celf e("/proc/self/exe", "");
  cregex regex("^__kernelgen_main$", REG_EXTENDED | REG_NOSUB);
  vector<csymbol *> symbols = e.getSymtab()->find(regex);
  if (!symbols.size()) {
    THROW("Cannot find the __kernelgen_main symbol. Is application compiled "
          "with KernelGen?");
  } else {
    // Load the regular main function prototype.
    // It is needed to correctly select the argument list.
    MemoryBuffer *buffer = MemoryBuffer::getMemBuffer(
        StringRef(symbols[0]->getData(), symbols[0]->getSize()), "", false);
    string err;
    Module *m = getLazyBitcodeModule(buffer, context, &err);
    if (!m)
      THROW(err);
    regular_main = m->getFunction("__kernelgen_regular_main");
    if (!regular_main)
      THROW("Cannot find the __kernelgen_regular_main function");
  }

  FunctionType *mainFuncTy = regular_main->getFunctionType();
  Type *mainRetTy = regular_main->getReturnType();

  // Structure defining aggregate form of the main entry
  // parameters. Return value is also packed, since
  // in CUDA and OpenCL kernels must return void.
  // Also structure aggregates the callback record containing
  // parameters of host-device communication state and the
  // head of memory pool.
  // Note the prototype of main function may vary, and in
  // structure we define the maximum possible parameter list.
  struct main_args_t {
    uint64_t *addressesOfGlobalVariables;
    kernelgen_callback_t *callback;
    kernelgen_memory_t *memory;
    int argc;
    char **argv;
    char **envp;
    int ret;
  };

  if (RUNMODE != KERNELGEN_RUNMODE_UNDEF) {
    // Build kernels index.
    VERBOSE("Building kernels index ...\n");
    cregex regex("^__kernelgen_.*$", REG_EXTENDED | REG_NOSUB);
    vector<csymbol *> symbols = e.getSymtab()->find(regex);
    for (vector<csymbol *>::iterator i = symbols.begin(), ie = symbols.end();
         i != ie; i++) {
      csymbol *symbol = *i;
      const string &name = symbol->getName();
      Kernel *kernel = new Kernel();
      kernel->name = name;
      kernel->source = string(symbol->getData(), symbol->getSize());
      kernel->loaded = false;

      // Initially, all targets are supported.
      for (int ii = 0; ii < KERNELGEN_RUNMODE_COUNT; ii++) {
        kernel->target[ii].supported = true;
        kernel->target[ii].binary = NULL;
      }

      kernels[name] = kernel;
      VERBOSE(name << "\n");
    }
    VERBOSE("\n");

    // Check whether the internal table contains a main entry.
    Kernel *kernel = kernels["__kernelgen_main"];
    if (!kernel) {
      THROW("Cannot find the __kernelgen_main symbol");
    }

    // Walk through kernel index and replace
    // all names with kernel structure addresses
    // for each kernelgen_launch call.
    //SMDiagnostic diag;
    for (map<string, Kernel *>::iterator i = kernels.begin(), e = kernels.end();
         i != e; i++) {
      Kernel *kernel = (*i).second;

      if (!kernel)
        THROW("Invalid kernel item");

#ifdef KERNELGEN_LOAD_KERNELS_LAZILY
      if (kernel->name != "__kernelgen_main")
        continue;
#endif

      load_kernel(kernel);
    }

    kernel = kernels["__kernelgen_main"];
    assert(kernel->module && "main module must be loaded");
    assert(sizeof(void *) == sizeof(uint64_t));

    NamedMDNode *orderOfGlobalsMD =
        kernel->module->getNamedMetadata("OrderOfGlobals");
    assert(orderOfGlobalsMD);
    NumOfGVars = orderOfGlobalsMD->getNumOperands();
    AddressesOfGVars = NULL;
    for (int i = 0; i < NumOfGVars; i++) {
      MDNode *mdNode = orderOfGlobalsMD->getOperand(i);
      assert(mdNode->getNumOperands() == 2);
      assert(isa<MDString>(*(mdNode->getOperand(0))) &&
             isa<ConstantInt>(*(mdNode->getOperand(1))));
      StringRef name = cast<MDString>(mdNode->getOperand(0))->getString();
      uint64_t index = cast<ConstantInt>(mdNode->getOperand(1))->getZExtValue();
      orderOfGlobals[name] = index;
    }

    // Load arguments, depending on the target runmode
    // and invoke the entry point kernel.
    switch (RUNMODE) {
    case KERNELGEN_RUNMODE_NATIVE: {
      AddressesOfGVars = (uint64_t *)(calloc(NumOfGVars, sizeof(void *)));
      main_args_t args;
      args.addressesOfGlobalVariables = AddressesOfGVars;
      args.argc = argc;
      args.argv = argv;
      args.envp = envp;
      kernelgen_launch(kernel, sizeof(main_args_t), sizeof(int),
                       (CallbackData *)&args);
      return args.ret;
    }
    case KERNELGEN_RUNMODE_CUDA: {
      // Initialize dynamic kernels loader.
      kernelgen::runtime::cuda_context =
          kernelgen::bind::cuda::context::init(8192);

      const string& kernelgenSimplePath = FindProgramByName("kernelgen-simple");
      if (kernelgenSimplePath.empty())
        THROW("Cannot locate kernelgen binaries folder, is it included into "
              "$PATH ?");
      const string& kernelgenPath = kernelgenSimplePath.substr(0,
        kernelgenSimplePath.find_last_of("/\\"));

      // Load LLVM IR for kernel monitor, if not yet loaded.
      if (!monitor_module)
        monitor_module = load_module(kernelgenPath + "/../include/cuda/monitor.bc");

      // Load LLVM IR for KernelGen runtime functions, if not yet loaded.
      if (!runtime_module)
        runtime_module = load_module(kernelgenPath + "/../include/cuda/runtime.bc");

      // Load LLVM IR for CUDA math functions, if not yet loaded.
      if (!cuda_module) {
      	if (cuda_context->getSubarchMajor() == 2)
          cuda_module = load_module(kernelgenPath + "/../include/cuda/math.sm_20.bc");
        else if (cuda_context->getSubarchMajor() == 3) {
          if (cuda_context->getSubarchMinor() == 0)
            cuda_module = load_module(kernelgenPath + "/../include/cuda/math.sm_30.bc");
          else if (cuda_context->getSubarchMinor() == 5)
            cuda_module = load_module(kernelgenPath + "/../include/cuda/math.sm_35.bc");
        }

        if (!cuda_module)
          THROW("Device compute capability \"" << cuda_context->getSubarch() <<
            "\" is not supported by available CUDA device functions modules");

	// Replace __nvvm_reflect calls in CUDA functions.
        PassManager manager;
        manager.add(new DataLayout(cuda_module));
        manager.add(createNVVMReflectPass());
        manager.run(*cuda_module);

        // Mark all module functions as device functions.
        for (Module::iterator F = cuda_module->begin(), FE = cuda_module->end();
             F != FE; F++)
          F->setCallingConv(CallingConv::PTX_Device);
      }

      // Initialize plugins.
      char *plugins = getenv("kernelgen_plugins");
      if (plugins) {
        stringstream ss(plugins);
        string plugin;
        while (std::getline(ss, plugin, ':')) {
          void *handle = dlopen(plugin.c_str(), RTLD_NOW);
          if (!handle)
            THROW("Cannot dlopen " << plugin << " " << dlerror());
          kernelgen_after_ptx_t afterPTX =
              (kernelgen_after_ptx_t) dlsym(handle, "kernelgen_after_ptx");
          if (afterPTX)
            pluginsAfterPTX.push_back(afterPTX);
          kernelgen_after_cubin_t afterCUBIN =
              (kernelgen_after_cubin_t) dlsym(handle, "kernelgen_after_cubin");
          if (afterCUBIN)
            pluginsAfterCUBIN.push_back(afterCUBIN);
        }
      }

      // Initialize callback structure.
      // Initial lock state is "locked". It will be dropped by
      // special GPU monitor kernel, upon its launch.
      kernelgen_callback_t callback;
      callback.lock = 1;
      callback.state = KERNELGEN_STATE_INACTIVE;
      callback.kernel = NULL;
      callback.data = NULL;
      callback.szdata = sizeof(main_args_t);
      callback.szdatai = sizeof(int);
      kernelgen_callback_t *callback_dev = NULL;
      CU_SAFE_CALL(
          cuMemAlloc((void **)&callback_dev, sizeof(kernelgen_callback_t)));
      CU_SAFE_CALL(
          cuMemcpyHtoD(callback_dev, &callback, sizeof(kernelgen_callback_t)));
      kernel->target[RUNMODE].callback = callback_dev;

      // Setup device dynamic memory heap.
      int szheap = 16 * 1024 * 1024;
      char *cszheap = getenv("kernelgen_szheap");
      if (cszheap)
        szheap = atoi(cszheap);
      memory_pool = InitMemoryPool(szheap);

      // Duplicate argv into device memory.
      // Note in addition to actiual arguments we must pass pass
      // argv[argc] = NULL.
      char **argv_dev = NULL;
      if (mainFuncTy->getNumParams() > 1) {
        {
          size_t size = sizeof(char *) * (argc + 1);
          for (int i = 0; i < argc; i++)
            size += strlen(argv[i]) + 1;
          CU_SAFE_CALL(cuMemAlloc((void **)&argv_dev, size));
        }
        for (int i = 0, offset = 0; i < argc; i++) {
          char *arg = (char *)(argv_dev + argc + 1) + offset;
          CU_SAFE_CALL(cuMemcpyHtoD(argv_dev + i, &arg, sizeof(char *)));
          size_t length = strlen(argv[i]) + 1;
          CU_SAFE_CALL(cuMemcpyHtoD(arg, argv[i], length));
          offset += length;
        }
        CU_SAFE_CALL(cuMemsetD8(argv_dev + argc, 0, sizeof(char *)));
      }

      // Duplicate envp into device memory.
      // Note in addition to actual arguments we must pass
      // envp[argc] = NULL.
      char **envp_dev = NULL;
      if (mainFuncTy->getNumParams() > 2) {
        int envc = 0;
        {
          while (envp[envc])
            envc++;
          size_t size = sizeof(char *) * (envc + 1);
          for (int i = 0; i < envc; i++)
            size += strlen(envp[i]) + 1;
          CU_SAFE_CALL(cuMemAlloc((void **)&envp_dev, size));
        }
        for (int i = 0, offset = 0; i < envc; i++) {
          char *env = (char *)(envp_dev + envc + 1) + offset;
          CU_SAFE_CALL(cuMemcpyHtoD(envp_dev + i, &env, sizeof(char *)));
          size_t length = strlen(envp[i]) + 1;
          CU_SAFE_CALL(cuMemcpyHtoD(env, envp[i], length));
          offset += length;
        }
        CU_SAFE_CALL(cuMemsetD8(envp_dev + envc, 0, sizeof(char *)));
      }

      // Allocate page-locked memory for globals addresses.
      CU_SAFE_CALL(cuMemAllocHost((void **)&AddressesOfGVars,
                                  NumOfGVars * sizeof(void *)));
      CU_SAFE_CALL(cuMemsetD8((CUdeviceptr) AddressesOfGVars, 0,
                              NumOfGVars * sizeof(void *)));

      // Setup argerator structure and fill it with the main
      // entry arguments.
      main_args_t args_host;
      args_host.addressesOfGlobalVariables = AddressesOfGVars;
      args_host.argc = argc;
      args_host.argv = argv_dev;
      args_host.callback = callback_dev;
      args_host.memory = memory_pool;
      main_args_t *args_dev = NULL;
      CU_SAFE_CALL(cuMemAlloc((void **)&args_dev, sizeof(main_args_t)));
      CU_SAFE_CALL(cuMemcpyHtoD(args_dev, &args_host, sizeof(main_args_t)));
      kernelgen_launch(kernel, sizeof(main_args_t), sizeof(int),
                       (CallbackData *)args_dev);

      // Store back to host the return value, if present.
      int ret = EXIT_SUCCESS;
      if (!mainRetTy->isVoidTy()) {
        CU_SAFE_CALL(cuMemcpyDtoH(&ret, &args_dev->ret, sizeof(int)));
      }

      // Release device memory buffers.
      if (argv_dev) {
        CU_SAFE_CALL(cuMemFree(argv_dev));
      }
      if (envp_dev) {
        CU_SAFE_CALL(cuMemFree(envp_dev));
      }
      CU_SAFE_CALL(cuMemFree(callback_dev));
      CU_SAFE_CALL(cuMemFree(args_dev));

      delete kernelgen::runtime::cuda_context;

      return ret;
    }
    case KERNELGEN_RUNMODE_OPENCL: { THROW("Unsupported runmode" << RUNMODE); }
    default:
      THROW("Unknown runmode " << RUNMODE);
    }
  }

  // Chain to entry point of the regular binary.
  // Since the main entry prototype may vary, it is
  // invoked over FFI using the previously discovered
  // actual argument list.

  ffi_type *rtype = &ffi_type_sint32;
  if (mainRetTy->isVoidTy())
    rtype = &ffi_type_void;

  ffi_type *argsTy[] = { &ffi_type_sint32, &ffi_type_pointer,
                         &ffi_type_pointer };

  ffi_cif cif;
  if (ffi_prep_cif(&cif, FFI_DEFAULT_ABI, mainFuncTy->getNumParams(), rtype,
                   argsTy) !=
      FFI_OK)
    THROW("Error in ffi_prep_cif");

  main_args_t args;
  args.argc = argc;
  args.argv = argv;
  args.envp = envp;

  void *pargs[] = { &args.argc, &args.argv, &args.envp };

  int ret;
  typedef void(*func_t)();
  func_t func = (func_t) __regular_main;
  ffi_call(&cif, func, &ret, (void **)&pargs);

  return ret;
}
