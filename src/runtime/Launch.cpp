//===- Launch.cpp - Kernels launcher API ----------------------------------===//
//
//     KernelGen -- A prototype of LLVM-based auto-parallelizing Fortran/C
//        compiler for NVIDIA GPUs, targeting numerical modeling code.
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements launching of parallel loops on supported architectures.
//
//===----------------------------------------------------------------------===//

#include "Runtime.h"
#include "Timer.h"

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
using namespace kernelgen::utils;
using namespace llvm;
using namespace std;

// Read source into LLVM module for the specified kernel.
void load_kernel(Kernel* kernel) {
	LLVMContext& context = getGlobalContext();

	if (!kernel)
		THROW("Invalid kernel item");

	// Load IR from source.
	string err;
	MemoryBuffer* buffer = MemoryBuffer::getMemBuffer(
			StringRef(kernel->source.data(), kernel->source.size()), "", false);
	Module* m = ParseBitcodeFile(buffer, context, &err);
	if (!m)
		THROW(kernel->name << ": " << err);
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

	SmallVector<char, 128> moduleBitcode;
	raw_svector_ostream moduleBitcodeStream(moduleBitcode);
	WriteBitcodeToFile(m, moduleBitcodeStream);
	moduleBitcodeStream.flush();
	kernel->source.assign(moduleBitcode.data(), moduleBitcode.size());
	kernel->loaded = true;
	kernel->module = m;
}

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

	// In case we could be launching kernel from the host call -
	// synchronize its GPU memory modifications before doing anything.
	kernelgen_hostcall_memsync();

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
			void* monitor_stream = cuda_context->getSecondaryStream();

			// Copy launch arguments from host to device.
			// In order to determine the precompiled kernel hash,
			// only integer arguments are needed (first szdatai bytes).
			// In order to perform verbose pointers tracking for
			// debug purposes, all arguments are needed.
			size_t size = (settings.getVerboseMode() != Verbose::Disable) ? szdata : szdatai;
			vector<char> vcontent;
			vcontent.resize(size);
			char* content = &vcontent[0];
			CU_SAFE_CALL(cuMemcpyDtoHAsync(content, &data->args, size, monitor_stream));
			CU_SAFE_CALL(cuStreamSynchronize(monitor_stream));
			mhash(td, content, szdatai);

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
			float only_kernel_time;
			size_t szshmem = 0;
			CU_SAFE_CALL(cudyLaunch((CUDYfunction) kernel_func, gridDim.x,
					gridDim.y, gridDim.z, blockDim.x, blockDim.y,
					blockDim.z, szshmem, &data,
					cuda_context->getSecondaryStream(),
					&only_kernel_time));

			// Wait for loop kernel completion.
			CU_SAFE_CALL(cuStreamSynchronize(cuda_context->getSecondaryStream()));

			double time = t.get_elapsed();

			VERBOSE(Verbose::Always << Verbose::Cyan <<
					"Finishing kernel " << kernel->name << "\n" <<
					Verbose::Reset << Verbose::Default << Verbose::Flush);

			VERBOSE(Verbose::Perf << kernel->name << " time = " << time << " sec\n" <<
					kernel->name << " only kernel time = " << only_kernel_time << " sec\n" <<
					Verbose::Default);
			break;
		}

		// Create host-pinned callback structure buffer.
		struct kernelgen_callback_t callback;
		CU_SAFE_CALL(cuMemHostRegister(&callback,
				sizeof(struct kernelgen_callback_t), 0));

		// Launch main GPU kernel.
		{
			// The main kernel is launched on unit grid.
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

			// Main kernel takes one argument, which is a pointer onto main()
			// arguments aggregate. Additionally, main kernel takes another
			// pointer, which is a pointer onto GPU memory buffer for storing
			// main kernel LEPC.
			vector<void*> vargs;
			vargs.resize(2);
			vargs[0] = (void*)data;
			vargs[1] = (void*)kernelgen::runtime::cuda_context->getLEPCBufferPtr();
			size_t szvargs = vargs.size() * sizeof(void*);
			void* params[] =
			{
				CU_LAUNCH_PARAM_BUFFER_POINTER, &vargs[0],
				CU_LAUNCH_PARAM_BUFFER_SIZE, &szvargs,
				CU_LAUNCH_PARAM_END
			};
			CU_SAFE_CALL(cuLaunchKernel((void*) kernel_func, gridDim.x, gridDim.y,
					gridDim.z, blockDim.x, blockDim.y, blockDim.z, szshmem,
					cuda_context->getPrimaryStream(), NULL, params));
		}

		// Launch monitor GPU kernel.
		{
			// Codegen monitor kernel module.
			{
				Kernel monitor;
				monitor.name = "kernelgen_monitor";
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
			CU_SAFE_CALL(cudyLaunch((CUDYfunction) monitor_kernel, gridDim.x,
					gridDim.y, gridDim.z, blockDim.x, blockDim.y, blockDim.z,
					szshmem, args, cuda_context->getSecondaryStream(), NULL));
		}

		while (1) {
			// Wait for monitor kernel completion.
			CU_SAFE_CALL(cuStreamSynchronize(cuda_context->getSecondaryStream()));

			// Copy callback structure back to host memory and
			// check the state.
			CU_SAFE_CALL(cuMemcpyDtoHAsync(
					&callback, kernel->target[RUNMODE].callback,
					sizeof(struct kernelgen_callback_t),
					cuda_context->getSecondaryStream()));
			CU_SAFE_CALL(cuStreamSynchronize(cuda_context->getSecondaryStream()));
			switch (callback.state) {
			case KERNELGEN_STATE_INACTIVE: {
				VERBOSE("Kernel " << kernel->name << " has finished\n");
				break;
			}
			case KERNELGEN_STATE_LOOPCALL: {
				// Launch the loop kernel.
				if (kernelgen_launch(callback.kernel, callback.szdata,
						callback.szdatai, callback.data) != -1)
					break;

				// If kernel is not supported on device, launch it as
				// a host call.
				if (!callback.kernel->target[RUNMODE].supported) {
					timer t;
					LLVMContext& context = kernel->module->getContext();
					FunctionType* FunctionTy = TypeBuilder<void(types::i<32>*),
							true>::get(context);
					StructType* StructTy = StructType::get(
							Type::getInt8PtrTy(context),
							Type::getInt8PtrTy(context),
							Type::getInt8PtrTy(context), NULL);
					CallbackData data;
					data.args = callback.data;
					kernelgen_hostcall(callback.kernel, FunctionTy, StructTy,
							&data);
					VERBOSE(Verbose::Perf << callback.kernel->name << " time = " <<
							t.get_elapsed() << " sec\n" << Verbose::Default);
					break;
				}

				// Otherwise, kernel is supported, but not parallelizable.
				// In this case indicate __kernelgen_main must run single-threaded
				// fallback branch where kernel's code is embedded directly into
				// __kernelgen_main.
				int state = KERNELGEN_STATE_FALLBACK;
				CU_SAFE_CALL(cuMemcpyHtoDAsync(
						&kernel->target[RUNMODE].callback->state, &state,
						sizeof(int), cuda_context->getSecondaryStream()));
				CU_SAFE_CALL(cuStreamSynchronize(cuda_context->getSecondaryStream()));

				break;
			}
			case KERNELGEN_STATE_HOSTCALL: {
				// Copy arguments to the host memory.
				vector<char> vdata;
				vdata.resize(callback.szdata);
				CallbackData* data = (CallbackData*)&vdata[0];
				CU_SAFE_CALL(cuMemcpyDtoHAsync(data, callback.data, callback.szdata,
						cuda_context->getSecondaryStream()));
				CU_SAFE_CALL(cuStreamSynchronize(cuda_context->getSecondaryStream()));

				timer t;
				{
					kernelgen_hostcall(callback.kernel, data->FunctionTy,
							data->StructTy, data);
				}
				VERBOSE(Verbose::Perf << callback.kernel->name << " time = " <<
						t.get_elapsed() << " sec\n" << Verbose::Default);
				break;
			}
			default:
				THROW("Unknown callback state : " << callback.state);
			}

			if (callback.state == KERNELGEN_STATE_INACTIVE)
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
				CU_SAFE_CALL(cudyLaunch((CUDYfunction) monitor_kernel, gridDim.x,
						gridDim.y, gridDim.z, blockDim.x, blockDim.y,
						blockDim.z, szshmem, args,
						cuda_context->getSecondaryStream(), NULL));
			}
		}

		// Finally, synchronize kernel stream.
		CU_SAFE_CALL(cuStreamSynchronize(cuda_context->getPrimaryStream()));
		CU_SAFE_CALL(cuMemHostUnregister(&callback));

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

