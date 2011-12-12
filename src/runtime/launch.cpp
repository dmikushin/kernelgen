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
#include "runtime.h"
#include "util.h"

#include <ffi.h>
#include <mhash.h>

#include "llvm/DerivedTypes.h"
#include "llvm/Function.h"
#include "llvm/ExecutionEngine/GenericValue.h"
#include "llvm/Target/TargetData.h"

using namespace kernelgen;
using namespace kernelgen::bind::cuda;
using namespace kernelgen::runtime;
using namespace llvm;
using namespace std;

static ffi_type* ffiTypeFor(Type *Ty)
{
	switch (Ty->getTypeID())
	{
	case Type::VoidTyID : return &ffi_type_void;
	case Type::IntegerTyID :
		switch (cast<IntegerType>(Ty)->getBitWidth())
		{
		case 8 : return &ffi_type_sint8;
		case 16 : return &ffi_type_sint16;
		case 32 : return &ffi_type_sint32;
		case 64 : return &ffi_type_sint64;
		}
	case Type::FloatTyID : return &ffi_type_float;
	case Type::DoubleTyID : return &ffi_type_double;
	case Type::PointerTyID : return &ffi_type_pointer;
	default : break;
	}
	
	// TODO: Support other types such as StructTyID, ArrayTyID, OpaqueTyID, etc.
	THROW("Type could not be mapped for use with libffi.");
	return NULL;
}

static void* ffiValueFor(
	Type* Ty, const GenericValue &AV, void* ArgDataPtr)
{
	switch (Ty->getTypeID())
	{
	case Type::IntegerTyID :
		switch (cast<IntegerType>(Ty)->getBitWidth())
		{
		case 8 : 
			{
				int8_t* I8Ptr = (int8_t *) ArgDataPtr;
				*I8Ptr = (int8_t) AV.IntVal.getZExtValue();
				return ArgDataPtr;
			}
		case 16 :
			{
				int16_t* I16Ptr = (int16_t *) ArgDataPtr;
				*I16Ptr = (int16_t) AV.IntVal.getZExtValue();
				return ArgDataPtr;
			}
		case 32 :
			{
				int32_t* I32Ptr = (int32_t *) ArgDataPtr;
				*I32Ptr = (int32_t) AV.IntVal.getZExtValue();
				return ArgDataPtr;
			}
		case 64 :
			{
				int64_t* I64Ptr = (int64_t *) ArgDataPtr;
				*I64Ptr = (int64_t) AV.IntVal.getZExtValue();
				return ArgDataPtr;
			}
		}
	case Type::FloatTyID :
	{
		float* FloatPtr = (float *) ArgDataPtr;
		*FloatPtr = AV.FloatVal;
		return ArgDataPtr;
	}
	case Type::DoubleTyID :
	{
		double* DoublePtr = (double *) ArgDataPtr;
		*DoublePtr = AV.DoubleVal;
		return ArgDataPtr;
	}
	case Type::PointerTyID :
	{
		void** PtrPtr = (void **) ArgDataPtr;
		*PtrPtr = GVTOP(AV);
		return ArgDataPtr;
	}
	default: break;
	}

	// TODO: Support other types such as StructTyID, ArrayTyID, OpaqueTyID, etc.
	THROW("Type value could not be mapped for use with libffi.");
	return NULL;
}

typedef void (*func_t)();

static bool ffiInvoke(func_t Fn, Function *F,
                      const std::vector<GenericValue> &ArgVals,
                      const TargetData *TD, GenericValue &Result) {
  ffi_cif cif;
  FunctionType *FTy = F->getFunctionType();
  const unsigned NumArgs = F->arg_size();

  // TODO: We don't have type information about the remaining arguments, because
  // this information is never passed into ExecutionEngine::runFunction().
  if (ArgVals.size() > NumArgs && F->isVarArg()) {
    THROW("Calling external var arg function '" << F->getName().data() << "' is not supported by the Interpreter.");
  }

  unsigned ArgBytes = 0;

  std::vector<ffi_type*> args(NumArgs);
  for (Function::const_arg_iterator A = F->arg_begin(), E = F->arg_end();
       A != E; ++A) {
    const unsigned ArgNo = A->getArgNo();
    Type *ArgTy = FTy->getParamType(ArgNo);
    args[ArgNo] = ffiTypeFor(ArgTy);
    ArgBytes += TD->getTypeStoreSize(ArgTy);
  }

  SmallVector<uint8_t, 128> ArgData;
  ArgData.resize(ArgBytes);
  uint8_t *ArgDataPtr = ArgData.data();
  SmallVector<void*, 16> values(NumArgs);
  for (Function::const_arg_iterator A = F->arg_begin(), E = F->arg_end();
       A != E; ++A) {
    const unsigned ArgNo = A->getArgNo();
    Type *ArgTy = FTy->getParamType(ArgNo);
    values[ArgNo] = ffiValueFor(ArgTy, ArgVals[ArgNo], ArgDataPtr);
    ArgDataPtr += TD->getTypeStoreSize(ArgTy);
  }

  Type *RetTy = FTy->getReturnType();
  ffi_type *rtype = ffiTypeFor(RetTy);

  if (ffi_prep_cif(&cif, FFI_DEFAULT_ABI, NumArgs, rtype, &args[0]) == FFI_OK) {
    SmallVector<uint8_t, 128> ret;
    if (RetTy->getTypeID() != Type::VoidTyID)
      ret.resize(TD->getTypeStoreSize(RetTy));
    ffi_call(&cif, Fn, ret.data(), values.data());
    switch (RetTy->getTypeID()) {
      case Type::IntegerTyID:
        switch (cast<IntegerType>(RetTy)->getBitWidth()) {
          case 8:  Result.IntVal = APInt(8 , *(int8_t *) ret.data()); break;
          case 16: Result.IntVal = APInt(16, *(int16_t*) ret.data()); break;
          case 32: Result.IntVal = APInt(32, *(int32_t*) ret.data()); break;
          case 64: Result.IntVal = APInt(64, *(int64_t*) ret.data()); break;
        }
        break;
      case Type::FloatTyID:   Result.FloatVal   = *(float *) ret.data(); break;
      case Type::DoubleTyID:  Result.DoubleVal  = *(double*) ret.data(); break;
      case Type::PointerTyID: Result.PointerVal = *(void **) ret.data(); break;
      default: break;
    }
    return true;
  }

  return false;
}


// Launch kernel from the specified source code address.
int kernelgen_launch(
	char* entry, unsigned long long szarg, int* arg)
{
	kernel_t* kernel = (kernel_t*)entry;
	
	// Initialize hashing engine.
	MHASH td = mhash_init(MHASH_MD5);
	if (td == MHASH_FAILED)
		THROW("Cannot inilialize mhash");
	
	// Compute hash, depending on the runmode.
	switch (runmode)
	{
		case KERNELGEN_RUNMODE_NATIVE :
		{
			char* content = (char*)arg + sizeof(int64_t);
			mhash(td, content, szarg);
			break;
		}
		case KERNELGEN_RUNMODE_CUDA :
		{
			void* monitor_stream =
				kernel->target[runmode].monitor_kernel_stream;
			char* content;
			int err = cuMemAllocHost((void**)&content, szarg);
			if (err) THROW("Error in cuMemAllocHost " << err);
			cuMemcpyDtoHAsync(content, arg + sizeof(int64_t), szarg, monitor_stream);
			if (err) THROW("Error in cuMemcpyDtoHAsync " << err);
			err = cuStreamSynchronize(monitor_stream);
			if (err) THROW("Error in cuStreamSynchronize " << err);
			mhash(td, content, szarg);
			err = cuMemFreeHost(content);
			if (err) THROW("Error in cuMemFreeHost " << err);
			break;
		}
		case KERNELGEN_RUNMODE_OPENCL :
		{
			THROW("Unsupported runmode" << runmode);
		}
		default :
			THROW("Unknown runmode " << runmode);
	}
	unsigned char hash[16];	
	mhash_deinit(td, hash);
	if (verbose)
	{
		cout << kernel->name << " @ ";
		for (int i = 0 ; i < 16; i++)
			cout << (int)hash[i];
		cout << endl;
	}
	
	// Check if kernel with the specified hash is
	// already compiled.
	string strhash((char*)hash, 16);
	binaries_map_t& binaries =
		kernel->target[runmode].binaries;
	binaries_map_t::iterator
		binary = binaries.find(strhash);
	char* kernel_func = NULL;
	if (binary == binaries.end())
	{
		if (verbose)
			cout << "No prebuilt kernel, compiling..." << endl;
	
		// Compile kernel for the specified target.
		kernel_func = compile(runmode, kernel);
		binaries[strhash] = kernel_func;
	}
	else
		kernel_func = (*binary).second;
	
	// Execute kernel, depending on target.
	switch (runmode)
	{
		case KERNELGEN_RUNMODE_NATIVE :
		{
			kernel_func_t native_kernel_func =
				(kernel_func_t)kernel_func;
			native_kernel_func(arg);
			break;
		}
		case KERNELGEN_RUNMODE_CUDA :
		{
			// If this is the main kernel being lauched,
			// first launch GPU monitor kernel, then launch
			// target kernel. Otherwise - vise versa.
			if (strcmp(kernel->name.c_str(), "__kernelgen_main"))
			{
				// Launch GPU loop kernel.
				{
					struct { unsigned int x, y, z; } gridDim, blockDim;
					gridDim.x = 1; gridDim.y = 1; gridDim.z = 1;
					blockDim.x = 1; blockDim.y = 1; blockDim.z = 1;
					size_t szshmem = 0;
					void* kernel_func_args[] = { (void*)&arg };
					int err = cuLaunchKernel((void*)kernel_func,
						gridDim.x, gridDim.y, gridDim.z,
						blockDim.x, blockDim.y, blockDim.z, szshmem,
						kernel->target[runmode].monitor_kernel_stream,
						kernel_func_args, NULL);
					if (err)
						THROW("Error in cuLaunchKernel " << err);
				}
				
				// Wait for loop kernel completion.
				int err = cuStreamSynchronize(
					kernel->target[runmode].monitor_kernel_stream);
				if (err) THROW("Error in cuStreamSynchronize " << err);
				break;
			}

			// Launch monitor GPU kernel.
			{
				struct { unsigned int x, y, z; } gridDim, blockDim;
				gridDim.x = 1; gridDim.y = 1; gridDim.z = 1;
				blockDim.x = 1; blockDim.y = 1; blockDim.z = 1;
				size_t szshmem = 0;
				void* monitor_kernel_func_args[] =
					{ (void*)&kernel->target[runmode].callback };
				int err = cuLaunchKernel(
					kernel->target[runmode].monitor_kernel_func,
					gridDim.x, gridDim.y, gridDim.z,
					blockDim.x, blockDim.y, blockDim.z, szshmem,
					kernel->target[runmode].monitor_kernel_stream,
					monitor_kernel_func_args, NULL);
				if (err)
					THROW("Error in cuLaunchKernel " << err);
			}
	
			// Launch main GPU kernel.
			{
				struct { unsigned int x, y, z; } gridDim, blockDim;
				gridDim.x = 1; gridDim.y = 1; gridDim.z = 1;
				blockDim.x = 1; blockDim.y = 1; blockDim.z = 1;
				size_t szshmem = 0;
				void* kernel_func_args[] = { (void*)&arg };
				int err = cuLaunchKernel((void*)kernel_func,
					gridDim.x, gridDim.y, gridDim.z,
					blockDim.x, blockDim.y, blockDim.z, szshmem,
					kernel->target[runmode].kernel_stream,
					kernel_func_args, NULL);
				if (err)
					THROW("Error in cuLaunchKernel " << err);
			}

			// Create host-pinned callback structure buffer.
			struct kernelgen_callback_t* callback = NULL;
			int err = cuMemAllocHost((void**)&callback, sizeof(struct kernelgen_callback_t));
			if (err) THROW("Error in cuMemAllocHost " << err);

			while (1)
			{
				// Wait for monitor kernel completion.
				err = cuStreamSynchronize(
					kernel->target[runmode].monitor_kernel_stream);
				if (err) THROW("Error in cuStreamSynchronize " << err);

				// Copy callback structure back to host memory and
				// check the state.
				err = cuMemcpyDtoHAsync(
					callback, kernel->target[runmode].callback,
					sizeof(struct kernelgen_callback_t),
					kernel->target[runmode].monitor_kernel_stream);
				if (err) THROW("Error in cuMemcpyDtoHAsync");
				err = cuStreamSynchronize(
					kernel->target[runmode].monitor_kernel_stream);
				if (err) THROW("Error in cuStreamSynchronize " << err);
				switch (callback->state)
				{
					case KERNELGEN_STATE_INACTIVE :
					{
						if (verbose) 
							cout << "Kernel " << kernel->name <<
								" has finished" << endl;
						break;
					}
					case KERNELGEN_STATE_LOOPCALL :
					{
						if (verbose)
							cout << "Kernel " << kernel->name <<
								" requested loop kernel call " << (void*)callback->name << endl;

						// Launch the loop kernel.
						kernelgen_launch((char*)callback->name, callback->szarg, callback->arg);

						break;
					}
					case KERNELGEN_STATE_HOSTCALL :
					{
						if (verbose)
							cout << "Kernel " << kernel->name <<
								" requested host function call " << (void*)callback->name << endl;
					
						// Copy arguments to the host memory.
						struct
						{
							FunctionType* FunctionTy;
							StructType* StructTy;
							void* params;
						}
						*arg = NULL;
						err = cuMemAllocHost((void**)&arg, callback->szarg);
						if (err) THROW("Error in cuMemAllocHost " << err);
						err = cuMemcpyDtoHAsync(arg, callback->arg, callback->szarg,
							kernel->target[runmode].monitor_kernel_stream);
						if (err) THROW("Error in cuMemcpyDtoHAsync " << err);
						err = cuStreamSynchronize(
							kernel->target[runmode].monitor_kernel_stream);
						if (err) THROW("Error in cuStreamSynchronize " << err);
						
						// Extract types from call inst and translate them to the
						// corresponding types of the FFI.
						FunctionType* FunctionTy = arg->FunctionTy;
						StructType* StructTy = arg->StructTy;
						/*for (unsigned i = 0, e = call->getNumArgOperands(); i != e; i++)
						{
							Type* type = call->getArgOperand(i)->getType();
							type->dump();
						}*/

						err = cuMemFreeHost(arg);
						if (err) THROW("Error in cuMemFreeHost " << err);

						break;
					}
					default :
						THROW("Unknown callback state : " << callback->state);
				}
				
				if (callback->state == KERNELGEN_STATE_INACTIVE) break;

				// Launch monitor GPU kernel.
				{
					struct { unsigned int x, y, z; } gridDim, blockDim;
					gridDim.x = 1; gridDim.y = 1; gridDim.z = 1;
					blockDim.x = 1; blockDim.y = 1; blockDim.z = 1;
					size_t szshmem = 0;
					void* monitor_kernel_func_args[] =
						{ (void*)&kernel->target[runmode].callback };
					int err = cuLaunchKernel(
						kernel->target[runmode].monitor_kernel_func,
						gridDim.x, gridDim.y, gridDim.z,
						blockDim.x, blockDim.y, blockDim.z, szshmem,
						kernel->target[runmode].monitor_kernel_stream,
						monitor_kernel_func_args, NULL);
					if (err)
						THROW("Error in cuLaunchKernel " << err);
				}
			}

			// Finally, sychronize kernel stream.
			err = cuStreamSynchronize(
				kernel->target[runmode].kernel_stream);
			if (err) THROW("Error in cuStreamSynchronize " << err);
			
			err = cuMemFreeHost(callback);
			if (err) THROW("Error in cuMemFreeHost " << err);
			
			break;
		}
		case KERNELGEN_RUNMODE_OPENCL :
		{
			// TODO: Launch kernel using OpenCL API
			THROW("Unsupported runmode" << runmode);
			break;
		}
		default :
			THROW("Unknown runmode " << runmode);
	}

	return 0;
}

