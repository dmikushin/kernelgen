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
	if (!verbose)
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
		default :
			// TODO: Support other types such as StructTyID, ArrayTyID, OpaqueTyID, etc.
			THROW("Type could not be mapped for use with libffi.");
		}
	}

	switch (Ty->getTypeID())
	{
	case Type::VoidTyID :
		cout << "void";
		return &ffi_type_void;
	case Type::IntegerTyID :
		switch (cast<IntegerType>(Ty)->getBitWidth())
		{
		case 8 :
			cout << "ffi_type_sint8";
			return &ffi_type_sint8;
		case 16 :
			cout << "ffi_type_sint16";
			return &ffi_type_sint16;
		case 32 :
			cout << "ffi_type_sint32";
			return &ffi_type_sint32;
		case 64 :
			cout << "ffi_type_sint64"; 
			return &ffi_type_sint64;
		}
	case Type::FloatTyID :
		cout << "ffi_type_float";
		return &ffi_type_float;
	case Type::DoubleTyID :
		cout << "ffi_type_double";
		return &ffi_type_double;
	case Type::PointerTyID :
		cout << "ffi_type_pointer";
		return &ffi_type_pointer;
	default : break;
		// TODO: Support other types such as StructTyID, ArrayTyID, OpaqueTyID, etc.
		THROW("Type could not be mapped for use with libffi.");
	}
	
	return NULL;
}

static void* ffiValueFor(
	kernel_t* kernel, Type* Ty, void* AV, void* ArgDataPtr)
{
	if (!verbose)
	{
		switch (Ty->getTypeID())
		{
		case Type::IntegerTyID :
			switch (cast<IntegerType>(Ty)->getBitWidth())
			{
			case 8 : 
				{
					*((int8_t*)ArgDataPtr) = *((int8_t*)AV);
					return ArgDataPtr;
				}
			case 16 :
				{
					*((int16_t*)ArgDataPtr) = *((int16_t*)AV);
					return ArgDataPtr;
				}
			case 32 :
				{
					*((int32_t*)ArgDataPtr) = *((int32_t*)AV);
					return ArgDataPtr;
				}
			case 64 :
				{
					*((int64_t*)ArgDataPtr) = *((int64_t*)AV);
					return ArgDataPtr;
				}
			}
		case Type::FloatTyID :
		{
			*((float*)ArgDataPtr) = *((float*)AV);
			return ArgDataPtr;
		}
		case Type::DoubleTyID :
		{
			*((double*)ArgDataPtr) = *((double*)AV);
			return ArgDataPtr;
		}
		case Type::PointerTyID :
		{
			// Use host memory instead of device:
			// figure out the allocated device memory range
			// and duplicate it on host.
			void* device = NULL;
			size_t size = 0;
			int err = cuMemGetAddressRange(&device, &size, AV);
			if (err) THROW("Error in cuMemGetAddressRange " << err);
			void* host = NULL;
			err = cuMemAllocHost((void**)&host, size);
			if (err) THROW("Error in cuMemAllocHost " << err);
			err = cuMemcpyDtoHAsync(host, device, size,
				kernel->target[runmode].monitor_kernel_stream);
			if (err) THROW("Error in cuMemcpyDtoHAsync");
			err = cuStreamSynchronize(
				kernel->target[runmode].monitor_kernel_stream);
			if (err) THROW("Error in cuStreamSynchronize " << err);
			*((void**)ArgDataPtr) = host;
			return ArgDataPtr;
		}
		default :
			// TODO: Support other types such as StructTyID, ArrayTyID, OpaqueTyID, etc.
			THROW("Type value could not be mapped for use with libffi.");
		}
	}

	switch (Ty->getTypeID())
	{
	case Type::IntegerTyID :
		switch (cast<IntegerType>(Ty)->getBitWidth())
		{
		case 8 : 
			{
				*((int8_t*)ArgDataPtr) = *((int8_t*)AV);
				cout << *((int8_t*)AV);
				return ArgDataPtr;
			}
		case 16 :
			{
				*((int16_t*)ArgDataPtr) = *((int16_t*)AV);
				cout << *((int16_t*)AV);
				return ArgDataPtr;
			}
		case 32 :
			{
				*((int32_t*)ArgDataPtr) = *((int32_t*)AV);
				cout << *((int32_t*)AV);
				return ArgDataPtr;
			}
		case 64 :
			{
				*((int64_t*)ArgDataPtr) = *((int64_t*)AV);
				cout << *((int64_t*)AV);
				return ArgDataPtr;
			}
		}
	case Type::FloatTyID :
	{
		*((float*)ArgDataPtr) = *((float*)AV);
		cout << *((float*)AV);
		return ArgDataPtr;
	}
	case Type::DoubleTyID :
	{
		*((double*)ArgDataPtr) = *((double*)AV);
		cout << *((double*)AV);
		return ArgDataPtr;
	}
	case Type::PointerTyID :
	{
		// Use host memory instead of device:
		// figure out the allocated device memory range
		// and duplicate it on host.
		void* base = NULL;
		size_t size = 0;
		int err = cuMemGetAddressRange(&base, &size, *((void**)AV));
		if (err) THROW("Error in cuMemGetAddressRange " << err);
		void* host = NULL;
		err = cuMemAllocHost((void**)&host, size);
		if (err) THROW("Error in cuMemAllocHost " << err);
		err = cuMemcpyDtoHAsync(host, base, size,
			kernel->target[runmode].monitor_kernel_stream);
		if (err) THROW("Error in cuMemcpyDtoHAsync " << err);
		err = cuStreamSynchronize(
			kernel->target[runmode].monitor_kernel_stream);
		if (err) THROW("Error in cuStreamSynchronize " << err);
		*((void**)ArgDataPtr) = host + ((ptrdiff_t)*((void**)AV) - (ptrdiff_t)base);
		cout << *((void**)AV) << " -> " << *((void**)ArgDataPtr);
		return ArgDataPtr;
	}
	default :
		// TODO: Support other types such as StructTyID, ArrayTyID, OpaqueTyID, etc.
		THROW("Type value could not be mapped for use with libffi.");
	}

	return NULL;
}

typedef void (*func_t)();

static void ffiInvoke(
	kernel_t* kernel, func_t func, FunctionType* FTy,
	StructType* StructTy, void* params,
	const TargetData* TD)
{
	// Skip first two fields, that are FunctionType and
	// StructureType itself, respectively.
	// Also exclude return arguent in case of non-void function.
	unsigned ArgBytes = 0;
	unsigned NumArgs = StructTy->getNumElements() - 2;
	if (!FTy->getReturnType()->isVoidTy()) NumArgs--;
	std::vector<ffi_type*> args(NumArgs);
	if (!verbose)
	{
		for (int i = 0; i < NumArgs; i++)
		{
			Type* ArgTy = StructTy->getElementType(i + 2);
			args[i] = ffiTypeFor(ArgTy);
			ArgBytes += TD->getTypeStoreSize(ArgTy);
		}
	}
	else
	{
		for (int i = 0; i < NumArgs; i++)
		{
			cout << "arg " << i << " type: ";
			Type* ArgTy = StructTy->getElementType(i + 2);
			args[i] = ffiTypeFor(ArgTy);
			ArgBytes += TD->getTypeStoreSize(ArgTy);
			cout << endl;
		}
	}

	const StructLayout* layout = TD->getStructLayout(StructTy);
	SmallVector<uint8_t, 128> ArgData;
	ArgData.resize(ArgBytes);
	uint8_t *ArgDataPtr = ArgData.data();
	SmallVector<void*, 16> values(NumArgs);
	if (!verbose)
	{
		for (int i = 0; i < NumArgs; i++)
		{
			Type* ArgTy = StructTy->getElementType(i + 2);
			int offset = layout->getElementOffset(i + 2);
			values[i] = ffiValueFor(kernel, ArgTy,
				(void*)((char*)params + offset), ArgDataPtr);
			ArgDataPtr += TD->getTypeStoreSize(ArgTy);
		}
	}
	else
	{
		for (int i = 0; i < NumArgs; i++)
		{
			cout << "arg " << i << " value: ";
			Type* ArgTy = StructTy->getElementType(i + 2);
			int offset = layout->getElementOffset(i + 2);
			values[i] = ffiValueFor(kernel, ArgTy,
				(void*)((char*)params + offset), ArgDataPtr);
			ArgDataPtr += TD->getTypeStoreSize(ArgTy);
			cout << endl;
		}
	}

	Type* RetTy = FTy->getReturnType();
	ffi_type* rtype = NULL;
	if (!verbose)
		rtype = ffiTypeFor(RetTy);
	else
	{
		cout << "ret type: ";
		rtype = ffiTypeFor(RetTy);
		cout << endl;
	}

	ffi_cif cif;
	if (ffi_prep_cif(&cif, FFI_DEFAULT_ABI, NumArgs,
		rtype, &args[0]) != FFI_OK)
		THROW("Error in fi_prep_cif");

	SmallVector<uint8_t, 128> ret;
	if (RetTy->getTypeID() != Type::VoidTyID)
		ret.resize(TD->getTypeStoreSize(RetTy));
	ffi_call(&cif, func, ret.data(), values.data());

	if (!RetTy->isVoidTy())
	{
		Type* ArgTy = StructTy->getElementType(NumArgs - 1);
		int offset = layout->getElementOffset(NumArgs - 1);
		memcpy((void*)((char*)params + offset), ret.data(),
			TD->getTypeStoreSize(ArgTy));
	}

	for (int i = 0; i < NumArgs; i++)
	{
		Type* ArgTy = StructTy->getElementType(i + 2);
		int offset = layout->getElementOffset(i + 2);
		size_t size = TD->getTypeStoreSize(ArgTy);
		if (!ArgTy->isPointerTy())
			memcpy((void*)((char*)params + offset), values[i], size);
		else
		{
			// Move changed data back from host to device memory.
			void* device = *(void**)((char*)params + offset);
			void* base = NULL;
			size_t size = 0;
			int err = cuMemGetAddressRange(&base, &size, device);
			if (err) THROW("Error in cuMemGetAddressRange " << err);
			void* host = values[i] - ((ptrdiff_t)device - (ptrdiff_t)base);
			err = cuMemcpyHtoDAsync(base, host, size,
				kernel->target[runmode].monitor_kernel_stream);
			if (err) THROW("Error in cuMemcpyDtoHAsync " << err);
			err = cuStreamSynchronize(
				kernel->target[runmode].monitor_kernel_stream);
			if (err) THROW("Error in cuStreamSynchronize " << err);
		}
	}
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

			// Create host-pinned callback structure buffer.
			struct kernelgen_callback_t* callback = NULL;
			int err = cuMemAllocHost((void**)&callback, sizeof(struct kernelgen_callback_t));
			if (err) THROW("Error in cuMemAllocHost " << err);

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
						Dl_info info;
						if (verbose)
						{
							if (!dladdr((void*)callback->name, &info))
								THROW("Error in dladdr " << dlerror());
							cout << "Kernel " << kernel->name <<
								" requested host function call " << info.dli_sname << endl;
						}
					
						// Copy arguments to the host memory.
						struct
						{
							FunctionType* FunctionTy;
							StructType* StructTy;
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

						// Perform hostcall using FFI.
						TargetData* TD = new TargetData(kernel->module);					
						ffiInvoke(kernel, (func_t)callback->name, arg->FunctionTy,
							arg->StructTy, (void*)arg, TD);

						//err = cuMemFreeHost(arg);
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

