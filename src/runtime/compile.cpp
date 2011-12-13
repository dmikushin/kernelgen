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

#include "llvm/Constants.h"
#include "llvm/LLVMContext.h"
#include "llvm/Instructions.h"
#include "llvm/Module.h"
#include "llvm/PassManager.h"
#include "llvm/Analysis/Passes.h"
#include "llvm/Support/FormattedStream.h"
#include "llvm/Support/IRReader.h"
#include "llvm/Support/Host.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/PassManagerBuilder.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/ToolOutputFile.h"
#include "llvm/Support/TypeBuilder.h"
#include "llvm/Target/TargetData.h"
#include "llvm/Target/TargetRegistry.h"
#include "llvm/Target/TargetSelect.h"
#include "llvm/Transforms/Scalar.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include "llvm/Transforms/Utils/Cloning.h"
#include "llvm/Transforms/Utils/FunctionUtils.h"

#include "polly/LinkAllPasses.h"
#include "CodeGeneration.h"

#include "io.h"
#include "util.h"
#include "runtime.h"

#include <cstdlib>
#include <dlfcn.h>
#include <fstream>
#include <list>

using namespace util::io;
using namespace llvm;
using namespace polly;
using namespace std;

// Target machines for runmodes.
auto_ptr<TargetMachine> kernelgen::targets[KERNELGEN_RUNMODE_COUNT];

static PassManager getPollyPassManager(Module* m)
{
	PassManager polly;
	PassRegistry &Registry = *PassRegistry::getPassRegistry();
	initializeCore(Registry);
	initializeScalarOpts(Registry);
	initializeIPO(Registry);
	initializeAnalysis(Registry);
	initializeIPA(Registry);
	initializeTransformUtils(Registry);
	initializeInstCombine(Registry);
	initializeInstrumentation(Registry);
	initializeTarget(Registry);

	polly.add(new TargetData(m));
	polly.add(createBasicAliasAnalysisPass());	// -basicaa
	polly.add(createPromoteMemoryToRegisterPass());	// -mem2reg
	polly.add(createCFGSimplificationPass());	// -simplifycfg
	polly.add(createInstructionCombiningPass());	// -instcombine
	polly.add(createTailCallEliminationPass());	// -tailcallelim
	polly.add(createLoopSimplifyPass());		// -loop-simplify
	polly.add(createLCSSAPass());			// -lcssa
	polly.add(createLoopRotatePass());		// -loop-rotate
	polly.add(createLCSSAPass());			// -lcssa
	polly.add(createLoopUnswitchPass());		// -loop-unswitch
	polly.add(createInstructionCombiningPass());	// -instcombine
	polly.add(createLoopSimplifyPass());		// -loop-simplify
	polly.add(createLCSSAPass());			// -lcssa
	polly.add(createIndVarSimplifyPass());		// -indvars
	polly.add(createLoopDeletionPass());		// -loop-deletion
	polly.add(createInstructionCombiningPass());	// -instcombine		
	polly.add(createCodePreperationPass());		// -polly-prepare
	polly.add(createRegionSimplifyPass());		// -polly-region-simplify
	polly.add(createIndVarSimplifyPass());		// -indvars
	polly.add(createBasicAliasAnalysisPass());	// -basicaa
	polly.add(createScheduleOptimizerPass());	// -polly-optimize-isl

	return polly;
}

char* kernelgen::runtime::compile(
	int runmode, kernel_t* kernel, Module* module)
{
	Module* m = module;
	LLVMContext &context = getGlobalContext();
	if (!m)
	{
		// Load LLVM IR source into module.
		SMDiagnostic diag;
		MemoryBuffer* buffer = MemoryBuffer::getMemBuffer(kernel->source);
		m = ParseIR(buffer, diag, context);
		if (!m)
			THROW(kernel->name << ":" << diag.getLineNo() << ": " <<
				diag.getLineContents() << ": " << diag.getMessage());
		m->setModuleIdentifier(kernel->name + "_module");
		kernel->module = m;
	}
	
	//m->dump();

	// Dump result of polly passes without codegen, if requested
	// for testing purposes.
	char* dump_polly = getenv("kernelgen_dump_polly");
	if (dump_polly)
	{
		std::auto_ptr<Module> m_clone;
		m_clone.reset(CloneModule(m));
		PassManager polly = getPollyPassManager(m_clone.get());
		polly.run(*m_clone.get());
		m_clone.get()->dump();
	}

	PassManager polly = getPollyPassManager(m);
	
	// Emit target assembly and binary image, depending
	// on runmode.
	switch (runmode)
	{
		case KERNELGEN_RUNMODE_NATIVE :
		{
			// Create target machine for NATIVE target and get its target data.
			if (!targets[KERNELGEN_RUNMODE_NATIVE].get())
			{
				InitializeAllTargets();
				InitializeAllTargetMCs();
				InitializeAllAsmPrinters();
				InitializeAllAsmParsers();

				Triple triple(m->getTargetTriple());
				if (triple.getTriple().empty())
					triple.setTriple(sys::getHostTriple());
				string err;
				const Target* target = TargetRegistry::lookupTarget(triple.getTriple(), err);
				if (!target)
					THROW("Error auto-selecting target for module '" << err << "'." << endl <<
						"Please use the -march option to explicitly pick a target.");
				targets[KERNELGEN_RUNMODE_NATIVE].reset(target->createTargetMachine(
					triple.getTriple(), "", "", Reloc::PIC_, CodeModel::Default));
				if (!targets[KERNELGEN_RUNMODE_NATIVE].get())
					THROW("Could not allocate target machine");

				// Override default to generate verbose assembly.
				targets[KERNELGEN_RUNMODE_NATIVE].get()->setAsmVerbosityDefault(true);
			}

			// Apply the Polly codegen for native target.
			polly.add(polly::createCodeGenerationPass()); // -polly-codegen
			polly.run(*m);

			// Dump result of polly passes with codegen, if requested
			// for testing purposes.
			char* dump_pollygen = getenv("kernelgen_dump_pollygen");
			if (dump_pollygen) m->dump();
			
			const TargetData* tdata = 
				targets[KERNELGEN_RUNMODE_NATIVE].get()->getTargetData();
			PassManager manager;
			manager.add(new TargetData(*tdata));

			// Setup output stream.
			string bin_string;
			raw_string_ostream bin_stream(bin_string);
			formatted_raw_ostream stream(bin_stream);

			// Ask the target to add backend passes as necessary.
			if (targets[KERNELGEN_RUNMODE_NATIVE].get()->addPassesToEmitFile(manager, stream,
				TargetMachine::CGFT_ObjectFile, CodeGenOpt::Aggressive))
				THROW("Target does not support generation of this file type");

			manager.run(*m);
			
			// Flush the resulting object binary to the
			// underlying string.
			stream.flush();

			//cout << bin_string;

			// Dump generated kernel object to first temporary file.
			cfiledesc tmp1 = cfiledesc::mktemp("/tmp/");
			{
				fstream tmp_stream;
				tmp_stream.open(tmp1.getFilename().c_str(),
					fstream::binary | fstream::out | fstream::trunc);
				tmp_stream << bin_string;
				tmp_stream.close();
			}
			
			// Link first and second objects together into third one.
			cfiledesc tmp2 = cfiledesc::mktemp("/tmp/");
			{
				string linker = "ld";
				std::list<string> linker_args;
				linker_args.push_back("-shared");
				linker_args.push_back("-o");
				linker_args.push_back(tmp2.getFilename());
				linker_args.push_back(tmp1.getFilename());
				if (verbose)
				{
					cout << linker;
					for (std::list<string>::iterator it = linker_args.begin();
						it != linker_args.end(); it++)
						cout << " " << *it;
					cout << endl;
				}
				execute(linker, linker_args, "", NULL, NULL);
			}

			// Do not return anything if module is explicitly specified.
			if (module) return NULL;

			// Load linked image and extract kernel entry point.
			void* handle = dlopen(tmp2.getFilename().c_str(),
				RTLD_NOW | RTLD_GLOBAL | RTLD_DEEPBIND);
			if (!handle)
				THROW("Cannot dlopen " << dlerror());
			
			kernel_func_t kernel_func = (kernel_func_t)dlsym(handle, kernel->name.c_str());
			if (!kernel_func)
				THROW("Cannot dlsym " << dlerror());
			
			if (verbose)
				cout << "Loaded '" << kernel->name << "' at: " << (void*)kernel_func << endl;

			return (char*)kernel_func;
		}
		case KERNELGEN_RUNMODE_CUDA :
		{
			// Apply the Polly codegen for native target.
			Pass* codegenPass;
			codegenPass = kernelgen::createCodeGenerationPass();
			polly.add(codegenPass); // -polly-codegen
			polly.run(*m);

			// Convert external functions CallInst-s into
			// host callback form. Do not convert CallInst-s
			// to device-resolvable intrinsics (syscalls and math).
			static string cuda_intrinsics[] =
			{
				#include "cuda_syscalls.h"
				#include "cuda_intrinsics.h"
				"printf", "puts",
				"kernelgen_threadIdx_x", "kernelgen_threadIdx_y", "kernelgen_threadIdx_z",
				"kernelgen_blockIdx_x", "kernelgen_blockIdx_y", "kernelgen_blockIdx_z",
				"kernelgen_blockDim_x", "kernelgen_blockDim_y", "kernelgen_blockDim_z",
				"kernelgen_gridDim_x", "kernelgen_gridDim_y", "kernelgen_girdDim_z",
				"kernelgen_launch", "kernelgen_finish"
			};
			Type* int32Ty = Type::getInt32Ty(context);
			std::vector<CallInst*> old_calls;
			Function* hostcall = Function::Create(
				TypeBuilder<void(types::i<8>*, types::i<64>, types::i<32>*), true>::get(context),
				GlobalValue::ExternalLinkage, "kernelgen_hostcall", m);

			// Wrap host calls in the kernel code.
			vector<CallInst*> erase_calls;
			Function* f = m->getFunction(kernel->name);
			for (Function::iterator bb = f->begin(); bb != f->end(); bb++)
				for (BasicBlock::iterator ii = bb->begin(), ie = bb->end(); ii != ie; ii++)
				{
					// Check if instruction in focus is a call.
					CallInst* call = dyn_cast<CallInst>(cast<Value>(ii));
					if (!call) continue;

					// Check if function is called (needs -instcombine pass).
					Function* callee = call->getCalledFunction();
					if (!callee) continue;
					if (!callee->isDeclaration()) continue;
					if (callee->isIntrinsic()) continue;

					// Check function is natively supported.
					bool native = false;
					for (int i = 0; i < sizeof(cuda_intrinsics) / sizeof(std::string); i++)
					{
						if (callee->getName() == cuda_intrinsics[i])
						{
							native = true;
							if (verbose)
								cout << "native: " << callee->getName().data() << endl;
							break;
						}
					}
					if (native) continue;

					if (verbose)
						cout << "hostcall: " << callee->getName().data() << endl;

					// Locate entire hostcall in the native code.
					void* host_func = (void*)dlsym(NULL, callee->getName().data());
					if (!host_func) THROW("Cannot dlsym " << dlerror());

					// Fill the arguments types structure.
					// First, place pointer to the function type.
					// Second, place pointer to the structure itself.
					std::vector<Type*> ArgTypes;
					ArgTypes.push_back(Type::getInt8PtrTy(context));
					ArgTypes.push_back(Type::getInt8PtrTy(context));
					for (unsigned i = 0, e = call->getNumArgOperands(); i != e; ++i)
						ArgTypes.push_back(call->getArgOperand(i)->getType());

					// Lastly, add the type of return value, if not void.
					Type* retTy = callee->getReturnType();
					if (!retTy->isVoidTy())
						ArgTypes.push_back(retTy);

					// Allocate memory for the struct.
					StructType *StructArgTy = StructType::get(
						context, ArgTypes, false /* isPacked */);
					AllocaInst* Struct = new AllocaInst(StructArgTy, 0, "", call);
		
					// Initially, fill struct with zeros.
					IRBuilder<> Builder(Struct);
					CallInst* MI = Builder.CreateMemSet(Struct,
						Constant::getNullValue(Type::getInt8Ty(context)),
						ConstantExpr::getSizeOf(StructArgTy), 1);

					// Store the function type.
					{
						// Generate index.
						Value *Idx[2];
						Idx[0] = Constant::getNullValue(Type::getInt32Ty(context));
						Idx[1] = ConstantInt::get(Type::getInt32Ty(context), 0);

						// Get address of "inputs[i]" in struct
						GetElementPtrInst *GEP = GetElementPtrInst::Create(
							Struct, Idx, "", call);

						// Store to that address.
						Type* type = callee->getFunctionType();
						StoreInst *SI = new StoreInst(ConstantExpr::getIntToPtr(
							ConstantInt::get(Type::getInt64Ty(context),
							(uint64_t)type), Type::getInt8PtrTy(context)),
							GEP, "", call);
					}

					// Store the struct type itself
					{
						// Generate index.
						Value *Idx[2];
						Idx[0] = Constant::getNullValue(Type::getInt32Ty(context));
						Idx[1] = ConstantInt::get(Type::getInt32Ty(context), 1);

						// Get address of "inputs[i]" in struct
						GetElementPtrInst *GEP = GetElementPtrInst::Create(
							Struct, Idx, "", call);

						// Store to that address.
						StructType *StructArgTy = StructType::get(
							context, ArgTypes, false /* isPacked */);
						StoreInst *SI = new StoreInst(ConstantExpr::getIntToPtr(
							ConstantInt::get(Type::getInt64Ty(context),
							(uint64_t)StructArgTy), Type::getInt8PtrTy(context)),
							GEP, "", call);
					}

				    	// Store input values to arguments struct.
					for (unsigned i = 0, e = call->getNumArgOperands(); i != e; ++i)
					{
						// Generate index.
						Value *Idx[2];
						Idx[0] = Constant::getNullValue(Type::getInt32Ty(context));
						Idx[1] = ConstantInt::get(Type::getInt32Ty(context), i + 2);

						// Get address of "inputs[i]" in struct
						GetElementPtrInst *GEP = GetElementPtrInst::Create(
							Struct, Idx, "", call);

						// Store to that address.
						StoreInst *SI = new StoreInst(call->getArgOperand(i), GEP, "", call);
					}

					// Store pointer to the host call function entry point.
					SmallVector<Value*, 16> call_args;
					call_args.push_back(ConstantExpr::getIntToPtr(
						ConstantInt::get(Type::getInt64Ty(context),
						(uint64_t)host_func), Type::getInt8PtrTy(context)));
					
					// Store the sizeof structure.
					call_args.push_back(ConstantExpr::getSizeOf(StructArgTy));

					// Store pointer to aggregated arguments struct
					// to the new call args list.
					Instruction* IntPtrToStruct = CastInst::CreatePointerCast(
						Struct, PointerType::getInt32PtrTy(context), "", call);
					call_args.push_back(IntPtrToStruct);
		
					// Emit call to kernelgen_hostcall.
					CallInst *newcall = CallInst::Create(hostcall, call_args, "", call);
					newcall->setCallingConv(call->getCallingConv());
					newcall->setAttributes(call->getAttributes());
					newcall->setDebugLoc(call->getDebugLoc());

					// Replace function from device module.
					if (retTy->isVoidTy())
						call->replaceAllUsesWith(newcall);
					else
					{
						// Generate index.
						Value *Idx[2];
						Idx[0] = Constant::getNullValue(Type::getInt32Ty(context));
						Idx[1] = ConstantInt::get(Type::getInt32Ty(context),
							call->getNumArgOperands() + 1);

						GetElementPtrInst *GEP = GetElementPtrInst::Create(
							Struct, Idx, "", call);

						LoadInst* LI = new LoadInst(GEP, "", call);
						call->replaceAllUsesWith(LI);
					}
					erase_calls.push_back(call);
					callee->setVisibility(GlobalValue::DefaultVisibility);
					callee->setLinkage(GlobalValue::ExternalLinkage);
				}

			for (vector<CallInst*>::iterator i = erase_calls.begin(),
				ie = erase_calls.end(); i != ie; i++)
				(*i)->eraseFromParent();

			// Optimize module.
			{
				PassManager manager;
				manager.add(createLowerSetJmpPass());
				PassManagerBuilder builder;
				builder.Inliner = createFunctionInliningPass();
				builder.OptLevel = 3;
				builder.DisableSimplifyLibCalls = true;
				builder.populateModulePassManager(manager);
				manager.run(*m);
			}

			m->dump();

			// Create target machine for CUDA target and get its target data.
			if (!targets[KERNELGEN_RUNMODE_CUDA].get())
			{
				InitializeAllTargets();
				InitializeAllTargetMCs();
				InitializeAllAsmPrinters();
				InitializeAllAsmParsers();

				const Target* target = NULL;
				Triple triple(m->getTargetTriple());
				if (triple.getTriple().empty())
					triple.setTriple(sys::getHostTriple());
				for (TargetRegistry::iterator it = TargetRegistry::begin(),
					ie = TargetRegistry::end(); it != ie; ++it)
				{
					if (!strcmp(it->getName(), "c"))
					{
						target = &*it;
						break;
					}
				}

				if (!target)
					THROW("LLVM is built without C Backend support");

				targets[KERNELGEN_RUNMODE_CUDA].reset(target->createTargetMachine(
					triple.getTriple(), "", "", Reloc::PIC_, CodeModel::Default));
				if (!targets[KERNELGEN_RUNMODE_CUDA].get())
					THROW("Could not allocate target machine");

				// Override default to generate verbose assembly.
				targets[KERNELGEN_RUNMODE_CUDA].get()->setAsmVerbosityDefault(true);
			}
			
			PassManager manager;

			// Setup output stream.
			string bin_string;
			raw_string_ostream bin_stream(bin_string);
			formatted_raw_ostream stream(bin_stream);

			// Ask the target to add backend passes as necessary.
			if (targets[KERNELGEN_RUNMODE_CUDA].get()->addPassesToEmitFile(manager, stream,
				TargetMachine::CGFT_AssemblyFile, CodeGenOpt::Aggressive))
				THROW("Target does not support generation of this file type");

			manager.run(*m);
			
			// Flush the resulting object binary to the
			// underlying string.
			stream.flush();

			cout << bin_string;

			return nvopencc(bin_string, kernel->name);

			break;
		}
		case KERNELGEN_RUNMODE_OPENCL :
		{
			THROW("Unsupported runmode" << runmode);
			break;
		}
		default :
			THROW("Unknown runmode " << runmode);
	}
	
	return NULL;
}

