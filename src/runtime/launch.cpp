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
#include <signal.h>
#include <sys/mman.h>
#include <sys/types.h>

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
	default :
		// TODO: Support other types such as StructTyID, ArrayTyID, OpaqueTyID, etc.
		THROW("Type could not be mapped for use with libffi.");
	}
	return NULL;
}

struct mmap_t
{
	void* addr;
	size_t size, align;
};

static list<struct mmap_t> mmaps;

static kernel_t* active_kernel;

// SIGSEGV signal handler to catch accesses to GPU memory.
static void sighandler(int code, siginfo_t *siginfo, void* ucontext)
{
	// Check if address is valid on GPU.
	void* addr = siginfo->si_addr;

	void* base;
	size_t size;
	int err = cuMemGetAddressRange(&base, &size, addr);
	if (err) THROW("Not a GPU memory: " << addr);

	size_t align = (size_t)base % 4096;
	void* map = mmap((char*)base - align, size + align,
		PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_FIXED,
		-1, 0);
	if (map == (void*)-1)
		THROW("Cannot map memory onto " << base << " + " << size);

	mmap_t mmap;
	mmap.addr = map;
	mmap.size = size;
	mmap.align = align;
	mmaps.push_back(mmap);

	if (verbose)
		cout << "Mapped memory " << map << "(" << base << " - " <<
		align << ") + " << size << endl;

	err = cuMemcpyDtoHAsync(base, base, size,
		active_kernel->target[runmode].monitor_kernel_stream);
	if (err) THROW("Error in cuMemcpyDtoH " << err);
	err = cuStreamSynchronize(
		active_kernel->target[runmode].monitor_kernel_stream);
	if (err) THROW("Error in cuStreamSynchronize " << err);
}

typedef void (*func_t)();

static void ffiInvoke(
	kernel_t* kernel, func_t func, FunctionType* FTy,
	StructType* StructTy, void* params,
	const TargetData* TD)
{
	active_kernel = kernel;

	// Skip first two fields, that are FunctionType and
	// StructureType itself, respectively.
	// Also exclude return arguent in case of non-void function.
	unsigned ArgBytes = 0;
	unsigned NumArgs = StructTy->getNumElements() - 2;
	if (!FTy->getReturnType()->isVoidTy()) NumArgs--;
	std::vector<ffi_type*> args(NumArgs);
	for (int i = 0; i < NumArgs; i++)
	{
		Type* ArgTy = StructTy->getElementType(i + 2);
		args[i] = ffiTypeFor(ArgTy);
		ArgBytes += TD->getTypeStoreSize(ArgTy);
	}

	const StructLayout* layout = TD->getStructLayout(StructTy);
	SmallVector<uint8_t, 128> ArgData;
	ArgData.resize(ArgBytes);
	uint8_t *ArgDataPtr = ArgData.data();
	SmallVector<void*, 16> values(NumArgs);
	for (int i = 0; i < NumArgs; i++)
	{
		Type* ArgTy = StructTy->getElementType(i + 2);
		int offset = layout->getElementOffset(i + 2);
		size_t size = TD->getTypeStoreSize(ArgTy);
		void** address = (void**)((char*)params + offset);
		memcpy(ArgDataPtr, address, size);
		values[i] = ArgDataPtr;
		ArgDataPtr += size;
		if (ArgTy->isPointerTy())
		{
			// If pointer corresponds to device memory,
			// use the host memory instead:
			// figure out the allocated device memory range
			// and shadow it with host memory mapping.
			void* base;
			size_t size;
			int err = cuMemGetAddressRange(&base, &size, *address);
			if (!err)
			{			
				size_t align = (size_t)base % 4096;

				size_t mapped = (size_t)-1;
				for (list<struct mmap_t>::iterator i = mmaps.begin(), e = mmaps.end(); i != e; i++)
				{
					struct mmap_t mmap = *i;
					if (mmap.addr == (char*)base - align)
					{
						mapped = mmap.size + mmap.align;
						break;
					}
				}

				if (mapped == (size_t)-1)
				{
					// Map host memory with the same address and size
					// device memory has.
					void* map = mmap((char*)base - align, size + align,
						PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_FIXED,
						-1, 0);
					if (map == (void*)-1)
						THROW("Cannot map host memory onto " << base << " + " << size);
				

					// Track the mapped memory in list of mappings,
					// to synchronize them after the hostcall finishes.
					struct mmap_t mmap;
					mmap.addr = map;
					mmap.size = size;
					mmap.align = align;								
					mmaps.push_back(mmap);
				}
				else
				{
					// Remap the existing mapping, if it is required
					// to be larger.
					if (size + align > mapped)
					{
						void* map = mremap((char*)base - align, mapped, size + align,
							PROT_READ | PROT_WRITE);
						if (map == (void*)-1)
							THROW("Cannot map host memory onto " << base << " + " << size);
					
						// TODO: store new size in mmaps.
					}
				}

				if (verbose)
					cout << "Mapped memory " << base - align << "(" << base << " - " <<
					align << ") + " << size << endl;

				// Copy device memory to host mapped memory.
				err = cuMemcpyDtoHAsync(base, base, size,
					kernel->target[runmode].monitor_kernel_stream);
				if (err) THROW("Error in cuMemcpyDtoHAsync");
				err = cuStreamSynchronize(
					kernel->target[runmode].monitor_kernel_stream);
				if (err) THROW("Error in cuStreamSynchronize " << err);
			}
		}
	}

	Type* RetTy = FTy->getReturnType();
	ffi_type* rtype = NULL;
	rtype = ffiTypeFor(RetTy);

	ffi_cif cif;
	if (ffi_prep_cif(&cif, FFI_DEFAULT_ABI, NumArgs,
		rtype, &args[0]) != FFI_OK)
		THROW("Error in fi_prep_cif");

	SmallVector<uint8_t, 128> ret;
	if (RetTy->getTypeID() != Type::VoidTyID)
		ret.resize(TD->getTypeStoreSize(RetTy));

	// Register SIGSEGV signal handler to catch
	// accesses to GPU memory and remebmer the original handler.
        struct sigaction sa_new, sa_old;
        sa_new.sa_flags = SA_SIGINFO;
        sigfillset(&sa_new.sa_mask);
        sa_new.sa_sigaction = sighandler;
        sigaction(SIGSEGV, &sa_new, &sa_old);
        
        if (verbose)
        	cout << "Starting hostcall to " << (void*)func << endl;

	ffi_call(&cif, func, ret.data(), values.data());

        if (verbose)
        	cout << "Finishing hostcall to " << (void*)func << endl;
	
	// Unregister SIGSEGV signal handler and resore the
	// original handler.
	memset(&sa_new, 0, sizeof(struct sigaction));
        sigaction(SIGSEGV, &sa_old, 0);

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
		void* address = (void*)((char*)params + offset);
		memcpy(address, values[i], size);
	}

	// Copy data back from host-mapped memory to device.
	for (list<struct mmap_t>::iterator i = mmaps.begin(), e = mmaps.end(); i != e; i++)
	{
		struct mmap_t mmap = *i;
		int err = cuMemcpyHtoDAsync(
			(char*)mmap.addr + mmap.align, (char*)mmap.addr + mmap.align, mmap.size,
			kernel->target[runmode].monitor_kernel_stream);
		if (err) THROW("Error in cuMemcpyHtoDAsync");
	}
	
	// Synchronize and unmap previously mapped host memory.
	int err = cuStreamSynchronize(
		kernel->target[runmode].monitor_kernel_stream);
	if (err) THROW("Error in cuStreamSynchronize " << err);
	for (list<struct mmap_t>::iterator i = mmaps.begin(), e = mmaps.end(); i != e; i++)
	{
		struct mmap_t mmap = *i;
                err = munmap(mmap.addr, mmap.size + mmap.align);
                if (err == -1)
                	THROW("Cannot unmap memory from " << mmap.addr <<
                		" + " << mmap.size + mmap.align);
        }
        
        mmaps.clear();
        
        if (verbose)
        	cout << "Finished hostcall handler" << endl;
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

