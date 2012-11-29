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

#include <iomanip>
#include <mhash.h>

#include "llvm/Constants.h"
#include "llvm/Function.h"
#include "llvm/ExecutionEngine/GenericValue.h"
#include "llvm/Support/IRReader.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/TypeBuilder.h"
#include "llvm/Target/TargetData.h"
#include "llvm/Support/raw_os_ostream.h"

using namespace kernelgen;
using namespace kernelgen::bind::cuda;
using namespace kernelgen::runtime;
using namespace llvm;
using namespace std;

#ifdef KERNELGEN_LOAD_KERNELS_LAZILY
// Read source into LLVM module for the specified kernel.
void load_kernel(Kernel* kernel) {
	LLVMContext& context = getGlobalContext();

	SMDiagnostic diag;
	if (!kernel)
		THROW("Invalid kernel item");

	// Load IR from source.
	MemoryBuffer* buffer = MemoryBuffer::getMemBuffer(kernel->source);
	Module* m = ParseIR(buffer, diag, context);
	if (!m)
		THROW(
				kernel->name << ":" << diag.getLineNo() << ": " << diag.getLineContents() << ": " << diag.getMessage());
	m->setModuleIdentifier(kernel->name + "_module");

	for (Module::iterator fi = m->begin(), fe = m->end(); fi != fe; fi++)
		for (Function::iterator bi = fi->begin(), be = fi->end(); bi != be;
				bi++)
			for (BasicBlock::iterator ii = bi->begin(), ie = bi->end();
					ii != ie; ii++) {
				// Check if instruction in focus is a call.
				CallInst* call = dyn_cast<CallInst>(cast<Value>(ii));
				if (!call)
					continue;

				// Check if function is called (needs -instcombine pass).
				Function* callee = call->getCalledFunction();
				if (!callee)
					continue;
				if (!callee->isDeclaration())
					continue;
				if (callee->getName() != "kernelgen_launch")
					continue;

				// Get the called function name from the metadata node.
				MDNode* nameMD = call->getMetadata("kernelgen_launch");
				if (!nameMD)
					THROW("Cannot find kernelgen_launch metadata");
				if (nameMD->getNumOperands() != 1)
					THROW(
							"Unexpected kernelgen_launch metadata number of operands");
				ConstantDataArray* nameArray = dyn_cast<ConstantDataArray>(
						nameMD->getOperand(0));
				if (!nameArray)
					THROW("Invalid kernelgen_launch metadata operand");
				if (!nameArray->isCString())
					THROW("Invalid kernelgen_launch metadata operand");
				string name = "__kernelgen_"
						+ (string) nameArray->getAsCString();
				VERBOSE("Launcher invokes kernel " << name << "\n");

				// Permanently assign launcher first argument with the address
				// of the called kernel function structure (for fast access).
				Kernel* kernel = kernels[name];
				if (!kernel)
					THROW(
							"Cannot get the name of kernel invoked by kernelgen_launch");
				call->setArgOperand(0,
						ConstantExpr::getIntToPtr(
								ConstantInt::get(Type::getInt64Ty(context),
										(uint64_t) kernel),
								Type::getInt8PtrTy(context)));
			}

	kernel->source = "";
	raw_string_ostream ir(kernel->source);
	ir << (*m);
	kernel->loaded = true;
	kernel->module = m;
	//m->dump();
}
#endif

// Launch the specified kernel.
int kernelgen_launch(Kernel* kernel, unsigned long long szdata,
		unsigned long long szdatai, CallbackData* data) {
#ifdef KERNELGEN_LOAD_KERNELS_LAZILY
	// Load kernel source, if lazy load is enabled.
	if (!kernel->loaded)
		load_kernel(kernel);
#endif
	if (!kernel->target[RUNMODE].supported)
		return -1;

	VERBOSE("Kernel function call " << kernel->name << "\n");

	// Lookup for kernel in table, only if it has at least
	// one scalar to compute hash footprint. Otherwise, compile
	// "generalized" kernel.
	KernelFunc kernel_func = kernel->target[RUNMODE].binary;
	if (szdatai && (kernel->name != "__kernelgen_main")) {
		// Initialize hashing engine.
		MHASH td = mhash_init(MHASH_MD5);
		if (td == MHASH_FAILED)
			THROW("Cannot inilialize mhash");

		// Compute hash, depending on the runmode.
		void * args;
		switch (RUNMODE) {
		case KERNELGEN_RUNMODE_NATIVE: {
			mhash(td, &data->args, szdatai);
			args = data;
			break;
		}
		case KERNELGEN_RUNMODE_CUDA: {
			void* monitor_stream = kernel->target[RUNMODE].MonitorStream;

			// Copy launch arguments from host to device.
			// In order to determine the precompiled kernel hash,
			// only integer arguments are needed (first szdatai bytes).
			// In order to perform verbose pointers tracking for
			// debug purposes, all arguments are needed.
			size_t size = (settings.getVerboseMode() != Verbose::Disable) ? szdata : szdatai;
			char* content = (char*) malloc(size);
			int err = cuMemHostRegister(content, size, 0);
			if (err)
				THROW("Error in cuMemHostRegister " << err);
			err = cuMemcpyDtoHAsync(content, &data->args, size, monitor_stream);
			if (err)
				THROW("Error in cuMemcpyDtoHAsync " << err);
			err = cuStreamSynchronize(monitor_stream);
			if (err)
				THROW("Error in cuStreamSynchronize " << err);
			mhash(td, content, szdatai);

			// TODO: Unpin and free the device buffer?
			//err = cuMemFreeHost(content);
			if (err)
				THROW("Error in cuMemFreeHost " << err);

			args = malloc(2 * sizeof(void *) + szdatai);
			memcpy((char *) args + 2 * sizeof(void *), content, szdatai);

			break;
		}
		case KERNELGEN_RUNMODE_OPENCL: {
			THROW("Unsupported runmode" << RUNMODE);
		}
		default:
			THROW("Unknown runmode " << RUNMODE);
		}
		unsigned char hash[16];
		mhash_deinit(td, hash);
		if (settings.getVerboseMode() != Verbose::Disable)
		{
			stringstream xstrhash;
			for (int i = 0; i < 16; i++)
				xstrhash << setfill('0') << setw(2) << hex << (int)(hash[i]);
			VERBOSE(kernel->name << " @ 0x" << xstrhash.str() << "\n");
		}

		// Check if kernel with the specified hash is
		// already compiled.
		string strhash((char*) hash, 16);
		binaries_map_t& binaries = kernel->target[RUNMODE].binaries;
		binaries_map_t::iterator binary = binaries.find(strhash);
		if (binary == binaries.end()) {
			VERBOSE("No prebuilt kernel, compiling...\n");

			// Compile kernel for the specified target.
			// Function may return NULL in case the kernel is
			// unexpected to be non-parallel - this must be
			// recorded to cache as well.
			kernel_func = Compile(RUNMODE, kernel, NULL, args, szdata, szdatai);
			binaries[strhash] = kernel_func;
		} else
			kernel_func = (*binary).second;
	} else {
		// Compile and store the universal binary.
		if (!kernel_func) {
			VERBOSE("No prebuilt kernel, compiling...\n");

			// If the universal binary cannot be compiled or is
			// not parallel, then mark kernel unsupported for
			// entire target.
			kernel_func = Compile(RUNMODE, kernel);
			kernel->target[RUNMODE].binary = kernel_func;
			if (!kernel_func)
				kernel->target[RUNMODE].supported = false;
		}
	}

	if (!kernel_func)
		return -1;

	// Execute kernel, depending on target.
	switch (RUNMODE) {
	case KERNELGEN_RUNMODE_NATIVE: {
		KernelFunc native_kernel_func = (KernelFunc) kernel_func;
		timer t;
		{
			native_kernel_func(data);
		}
		VERBOSE(Verbose::Perf << kernel->name << " time = " <<
				t.get_elapsed() << " sec\n"	<< Verbose::Default);
		break;
	}
	case KERNELGEN_RUNMODE_CUDA: {
		// If this is the main kernel being lauched,
		// first launch GPU monitor kernel, then launch
		// target kernel. Otherwise - vise versa.
		if (kernel->name != "__kernelgen_main") {
			// Launch GPU loop kernel, if it is compiled.
			dim3 blockDim = kernel->target[RUNMODE].blockDim;
			dim3 gridDim = kernel->target[RUNMODE].gridDim;
			VERBOSE(Verbose::Always << Verbose::Cyan <<
					"Launching kernel " << kernel->name << "\n" <<
					"    blockDim = " << blockDim << "\n" << "    gridDim = " <<
					gridDim << "\n" << Verbose::Reset << Verbose::Flush);
			timer t;
			float kernel_time;
			{
				size_t szshmem = 0;
				int err = cudyLaunch((CUDYfunction) kernel_func, gridDim.x,
						gridDim.y, gridDim.z, blockDim.x, blockDim.y,
						blockDim.z, szshmem, &data,
						kernel->target[RUNMODE].MonitorStream,
						&kernel_time);
				if (err)
					THROW("Error in cudyLaunch " << err);
			}

			// Wait for loop kernel completion.
			int err = cuStreamSynchronize(
					kernel->target[RUNMODE].MonitorStream);
			if (err)
				THROW("Error in cuStreamSynchronize " << err);

			VERBOSE(Verbose::Always << Verbose::Cyan <<
					"Finishing kernel " << kernel->name << "\n" <<
					Verbose::Reset << Verbose::Default << Verbose::Flush);

			VERBOSE(Verbose::Perf << kernel->name << " time = " << t.get_elapsed() << " sec\n" <<
					"only the kernel execution time = " << kernel_time << " sec\n" << Verbose::Default);
			break;
		}

		// Create host-pinned callback structure buffer.
		struct kernelgen_callback_t* callback =
				(struct kernelgen_callback_t*) malloc(
						sizeof(struct kernelgen_callback_t));
		int err = cuMemHostRegister(callback,
				sizeof(struct kernelgen_callback_t), 0);
		if (err)
			THROW("Error in cuMemHostRegister " << err);

		// Launch main GPU kernel.
		{
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
			void* kernel_func_args[] = { (void*) &data };
			int err = cuLaunchKernel((void*) kernel_func, gridDim.x, gridDim.y,
					gridDim.z, blockDim.x, blockDim.y, blockDim.z, szshmem,
					kernel->target[RUNMODE].KernelStream, kernel_func_args,
					NULL);
			if (err)
				THROW("Error in cuLaunchKernel " << err);
		}

		// Launch monitor GPU kernel.
		{
			// Codegen monitor kernel module.
			{
				Kernel monitor;
				monitor.name = "kernelgen_monitor";
				monitor.target[KERNELGEN_RUNMODE_CUDA].MonitorStream =
						kernel->target[RUNMODE].MonitorStream;
				monitor_kernel = kernelgen::runtime::Codegen(KERNELGEN_RUNMODE_CUDA,
						&monitor, monitor_module);
			}

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
			char args[256];
			memcpy(args, &kernel->target[RUNMODE].callback, sizeof(void*));
			int err = cudyLaunch((CUDYfunction) monitor_kernel, gridDim.x,
					gridDim.y, gridDim.z, blockDim.x, blockDim.y, blockDim.z,
					szshmem, args,
					kernel->target[RUNMODE].MonitorStream, NULL);
			if (err)
				THROW("Error in cudyLaunch " << err);
		}

		while (1) {
			// Wait for monitor kernel completion.
			int err = cuStreamSynchronize(
					kernel->target[RUNMODE].MonitorStream);
			if (err)
				THROW("Error in cuStreamSynchronize " << err);

			// Copy callback structure back to host memory and
			// check the state.
			err = cuMemcpyDtoHAsync(callback, kernel->target[RUNMODE].callback,
					sizeof(struct kernelgen_callback_t),
					kernel->target[RUNMODE].MonitorStream);
			if (err)
				THROW("Error in cuMemcpyDtoHAsync");
			err = cuStreamSynchronize(
					kernel->target[RUNMODE].MonitorStream);
			if (err)
				THROW("Error in cuStreamSynchronize " << err);
			switch (callback->state) {
			case KERNELGEN_STATE_INACTIVE: {
				VERBOSE("Kernel " << kernel->name << " has finished\n");
				break;
			}
			case KERNELGEN_STATE_LOOPCALL: {
				// Launch the loop kernel.
				callback->kernel->target[RUNMODE].MonitorStream =
						kernel->target[RUNMODE].MonitorStream;
				if (kernelgen_launch(callback->kernel, callback->szdata,
						callback->szdatai, callback->data) != -1)
					break;

				// If kernel is not supported on device, launch it as
				// a host call.
				if (!callback->kernel->target[RUNMODE].supported) {
					timer t;
					LLVMContext& context = kernel->module->getContext();
					FunctionType* FunctionTy = TypeBuilder<void(types::i<32>*),
							true>::get(context);
					StructType* StructTy = StructType::get(
							Type::getInt8PtrTy(context),
							Type::getInt8PtrTy(context),
							Type::getInt8PtrTy(context), NULL);
					CallbackData data;
					data.args = callback->data;
					kernelgen_hostcall(callback->kernel, FunctionTy, StructTy,
							&data);
					VERBOSE(Verbose::Perf << callback->kernel->name << " time = " <<
							t.get_elapsed() << " sec\n" << Verbose::Default);
					break;
				}

				// Otherwise, kernel is supported, but not parallelizable.
				// In this case indicate __kernelgen_main must run single-threaded
				// fallback branch where kernel's code is embedded directly into
				// __kernelgen_main.
				int state = KERNELGEN_STATE_FALLBACK;
				err = cuMemcpyHtoDAsync(
						&kernel->target[RUNMODE].callback->state, &state,
						sizeof(int),
						kernel->target[RUNMODE].MonitorStream);
				if (err)
					THROW("Error in cuMemcpyDtoHAsync " << err);
				err = cuStreamSynchronize(
						kernel->target[RUNMODE].MonitorStream);
				if (err)
					THROW("Error in cuStreamSynchronize " << err);

				break;
			}
			case KERNELGEN_STATE_HOSTCALL: {
				// Copy arguments to the host memory.
				CallbackData* data =
						(CallbackData*) malloc(callback->szdata);
				int err = cuMemHostRegister(data, callback->szdata, 0);
				if (err)
					THROW("Error in cuMemHostRegister " << err);
				err = cuMemcpyDtoHAsync(data, callback->data, callback->szdata,
						kernel->target[RUNMODE].MonitorStream);
				if (err)
					THROW("Error in cuMemcpyDtoHAsync " << err);
				err = cuStreamSynchronize(
						kernel->target[RUNMODE].MonitorStream);
				if (err)
					THROW("Error in cuStreamSynchronize " << err);

				timer t;
				{
					kernelgen_hostcall(callback->kernel, data->FunctionTy,
							data->StructTy, data);
				}
				VERBOSE(Verbose::Perf << callback->kernel->name << " time = " <<
						t.get_elapsed() << " sec\n" << Verbose::Default);

				//err = cuMemHostUnregister(data);
				//if (err) THROW("Error in cuMemHostUnregister " << err);
				//free(data);
				break;
			}
			default:
				THROW("Unknown callback state : " << callback->state);
			}

			if (callback->state == KERNELGEN_STATE_INACTIVE)
				break;

			// Launch monitor GPU kernel.
			{
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
				char args[256];
				memcpy(args, &kernel->target[RUNMODE].callback, sizeof(void*));
				int err = cudyLaunch((CUDYfunction) monitor_kernel, gridDim.x,
						gridDim.y, gridDim.z, blockDim.x, blockDim.y,
						blockDim.z, szshmem, args,
						kernel->target[RUNMODE].MonitorStream, NULL);
				if (err)
					THROW("Error in cudyLaunch " << err);
			}
		}

		// Finally, sychronize kernel stream.
		err = cuStreamSynchronize(kernel->target[RUNMODE].KernelStream);
		if (err)
			THROW("Error in cuStreamSynchronize " << err);

		err = cuMemHostUnregister(callback);
		if (err)
			THROW("Error in cuMemHostUnregister " << err);
		free(callback);

		break;
	}
	case KERNELGEN_RUNMODE_OPENCL: {
		// TODO: Launch kernel using OpenCL API
		THROW("Unsupported runmode" << RUNMODE);
		break;
	}
	default:
		THROW("Unknown runmode " << RUNMODE);
	}

	return 0;
}

