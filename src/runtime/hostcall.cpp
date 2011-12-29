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

#include "llvm/DerivedTypes.h"
#include "llvm/Target/TargetData.h"

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

static long szpage = -1;

// SIGSEGV signal handler to catch accesses to GPU memory.
static void sighandler(int code, siginfo_t *siginfo, void* ucontext)
{
	// Check if address is valid on GPU.
	void* addr = siginfo->si_addr;

	void* base;
	size_t size;
	int err = cuMemGetAddressRange(&base, &size, addr);
	if (err) THROW("Not a GPU memory: " << addr);

	size_t align = (size_t)base % szpage;
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
	if (szpage == -1)
		szpage = sysconf(_SC_PAGESIZE);

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
				size_t align = (size_t)base % szpage;

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

void kernelgen_hostcall(kernel_t* kernel,
	unsigned long long szdata, unsigned long long szdatai,
	kernelgen_callback_data_t* data_dev)
{
	// Compile native kernel, if there is source code.
	kernel_func_t kernel_func = compile(KERNELGEN_RUNMODE_NATIVE, kernel);

	// Copy arguments to the host memory.
	kernelgen_callback_data_t* data_host = NULL;
	int err = cuMemAllocHost((void**)&data_host, szdata);
	if (err) THROW("Error in cuMemAllocHost " << err);
	err = cuMemcpyDtoHAsync(data_host, data_dev, szdata,
		kernel->target[runmode].monitor_kernel_stream);
	if (err) THROW("Error in cuMemcpyDtoHAsync " << err);
	err = cuStreamSynchronize(
		kernel->target[runmode].monitor_kernel_stream);
	if (err) THROW("Error in cuStreamSynchronize " << err);

	// Perform hostcall using FFI.
	TargetData* TD = new TargetData(kernel->module);					
	ffiInvoke(kernel, (func_t)kernel_func, data_host->FunctionTy,
		data_host->StructTy, (void*)data_host, TD);

	//err = cuMemFreeHost(data_host);
	if (err) THROW("Error in cuMemFreeHost " << err);
}

