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

#include "runtime.h"
#include "util.h"

#include "llvm/DerivedTypes.h"
#include "llvm/Target/TargetData.h"

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

static list<struct mmap_t> mmappings;

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
	mmappings.push_back(mmap);

	if (verbose & KERNELGEN_VERBOSE_DATAIO)
		cout << "Mapped memory " << map << "(" << base << " - " <<
		align << ") + " << size << endl;

	err = cuMemcpyDtoHAsync(base, base, size,
		active_kernel->target[runmode].monitor_kernel_stream);
	if (err) THROW("Error in cuMemcpyDtoH " << err);
	err = cuStreamSynchronize(
		active_kernel->target[runmode].monitor_kernel_stream);
	if (err) THROW("Error in cuStreamSynchronize " << err);

	if (verbose & KERNELGEN_VERBOSE_DATAIO)
		cout << "Mapped memory " << (char*)base - align << "(" << base << " - " <<
		align << ") + " << size << endl;
}

typedef void (*func_t)();

void kernelgen_hostcall(
	kernel_t* kernel, FunctionType* FTy,
	StructType* StructTy, void* params)
{
	// Compile native kernel, if there is source code.
	kernel_func_t* func =
		&kernel->target[KERNELGEN_RUNMODE_NATIVE].binary;
	*func = compile(KERNELGEN_RUNMODE_NATIVE, kernel);

	Dl_info info;
	if (verbose & KERNELGEN_VERBOSE_HOSTCALL)
	{
		if (dladdr((void*)*func, &info))
			cout << "Host function call " << info.dli_sname << endl;
		else
			cout << "Host kernel call " << *func << endl;
	}

	TargetData* TD = new TargetData(kernel->module);

	if (szpage == -1)
		szpage = sysconf(_SC_PAGESIZE);

	active_kernel = kernel;

	// Skip first two fields, that are FunctionType and
	// StructureType itself, respectively.
	unsigned ArgBytes = 0;
	unsigned NumArgs = StructTy->getNumElements() - 2;

	// Also skip the last field, if return value is not void.
	// In this case the last field would be the return value
	// buffer itself that is neither function argument, nor
	// return value buffer pointer.
	// So, without first two fileds and without last field
	// NumArgs would only cover pointers to function arguments
	// and a pointer to return value buffer.
	Type* RetTy = FTy->getReturnType();
	if (!RetTy->isVoidTy()) NumArgs--;

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

				list<struct mmap_t>::iterator mmapping = mmappings.begin();
				for ( ; mmapping != mmappings.end(); mmapping++)
				{
					if (mmapping->addr == (char*)base - align)
						break;
				}

				if (mmapping == mmappings.end())
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
					struct mmap_t mmapping;
					mmapping.addr = map;
					mmapping.size = size;
					mmapping.align = align;								
					mmappings.push_back(mmapping);
				}
				else
				{
					// Remap the existing mapping, if it is required
					// to be larger.
					if (size + align > mmapping->size + mmapping->align)
					{
						void* remap = mremap((char*)base - align,
							mmapping->size + mmapping->align, size + align, 0);
						if (remap == (void*)-1)
							THROW("Cannot map host memory onto " << base << " + " << size);
					
						// Store new size & align.
						mmapping->size = size;
						mmapping->align = align;
					}
				}

				// Copy device memory to host mapped memory.
				int err = cuMemcpyDtoHAsync(base, base, size,
					kernel->target[runmode].monitor_kernel_stream);
				if (err) THROW("Error in cuMemcpyDtoHAsync");

				if (verbose & KERNELGEN_VERBOSE_DATAIO)
					cout << "Mapped memory " << (char*)base - align << "(" << base << " - " <<
					align << ") + " << size << endl;
			}
		}
	}

	ffi_type* rtype = NULL;
	rtype = ffiTypeFor(RetTy);
	if (!RetTy->isVoidTy()) NumArgs--;

	ffi_cif cif;
	if (ffi_prep_cif(&cif, FFI_DEFAULT_ABI, NumArgs,
		rtype, &args[0]) != FFI_OK)
		THROW("Error in fi_prep_cif");

	// Set address of return value buffer. Also mmapped
	// transparently in the same way as pointer arguments.
	void* ret = NULL;
	if (!RetTy->isVoidTy())
	{
		NumArgs++;
		Type* ArgTy = StructTy->getElementType(
			StructTy->getNumElements() - 2);
		int offset = layout->getElementOffset(
			StructTy->getNumElements() - 2);
		ret = *(void**)((char*)params + offset);
	}

	// Register SIGSEGV signal handler to catch
	// accesses to GPU memory and remebmer the original handler.
	struct sigaction sa_new, sa_old;
	sa_new.sa_flags = SA_SIGINFO;
	sigemptyset(&sa_new.sa_mask);
	sa_new.sa_sigaction = sighandler;
	if (sigaction(SIGSEGV, &sa_new, &sa_old) == -1)
		THROW("Error in sigaction " << errno);
        
        if (verbose & KERNELGEN_VERBOSE_HOSTCALL)
        	cout << "Starting hostcall to " << (void*)*func << endl;

	// Synchronize pending mmapped data transfers.
	int err = cuStreamSynchronize(
		kernel->target[runmode].monitor_kernel_stream);
	if (err) THROW("Error in cuStreamSynchronize " << err);

	ffi_call(&cif, (func_t)*func, ret, values.data());

        if (verbose & KERNELGEN_VERBOSE_HOSTCALL)
        	cout << "Finishing hostcall to " << (void*)*func << endl;
	
	// Unregister SIGSEGV signal handler and resore the
	// original handler.
	if (sigaction(SIGSEGV, &sa_old, &sa_new) == -1)
        	THROW("Error in sigaction " << errno);

	// Refresh arguments and return value pointer.
	for (int i = 0; i < NumArgs; i++)
	{
		Type* ArgTy = StructTy->getElementType(i + 2);
		int offset = layout->getElementOffset(i + 2);
		size_t size = TD->getTypeStoreSize(ArgTy);
		void* address = (void*)((char*)params + offset);
		memcpy(address, values[i], size);
	}

	// Copy data back from host-mapped memory to device.
	for (list<struct mmap_t>::iterator i = mmappings.begin(), e = mmappings.end(); i != e; i++)
	{
		// TODO: transfer hangs when size is not a multiplier of some power of two.
		// Currently the guess is 16. So, do we need to pad all arrays to 16?..
		struct mmap_t mmap = *i;
		size_t size = mmap.size;
		if (size % 16) size -= mmap.size % 16;
		int err = cuMemcpyHtoDAsync(
			(char*)mmap.addr + mmap.align, (char*)mmap.addr + mmap.align, size,
			kernel->target[runmode].monitor_kernel_stream);
		if (err) THROW("Error in cuMemcpyHtoDAsync " << err);
		if (verbose & KERNELGEN_VERBOSE_DATAIO)
			cout << "mmap.addr = " << mmap.addr << ", mmap.align = " <<
				mmap.align << ", mmap.size = " << mmap.size << " (" << size << ")" << endl;
	}
	
	// Synchronize and unmap previously mapped host memory.
	err = cuStreamSynchronize(
		kernel->target[runmode].monitor_kernel_stream);
	if (err) THROW("Error in cuStreamSynchronize " << err);
	for (list<struct mmap_t>::iterator i = mmappings.begin(), e = mmappings.end(); i != e; i++)
	{
		struct mmap_t mmap = *i;
                err = munmap(mmap.addr, mmap.size + mmap.align);
                if (err == -1)
                	THROW("Cannot unmap memory from " << mmap.addr <<
                		" + " << mmap.size + mmap.align);
        }
        
        mmappings.clear();
        
        if (verbose & KERNELGEN_VERBOSE_HOSTCALL)
        	cout << "Finished hostcall handler" << endl;
}

