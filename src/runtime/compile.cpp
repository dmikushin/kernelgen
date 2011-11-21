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
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetRegistry.h"
#include "llvm/Target/TargetSelect.h"
#include "llvm/Transforms/Scalar.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include "llvm/Transforms/Utils/Cloning.h"
#include "llvm/Transforms/Utils/FunctionUtils.h"

#include "polly/LinkAllPasses.h"

#include "io.h"
#include "util.h"
#include "runtime.h"

#include <dlfcn.h>
#include <fstream>
#include <list>

using namespace util::io;
using namespace llvm;
using namespace polly;
using namespace std;

static auto_ptr<TargetMachine> mcpu[KERNELGEN_RUNMODE_COUNT];

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
	}
	
	//m->dump();

	PassManager manager;
	Pass* codegenPass;
	{
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

		manager.add(new TargetData(m));
		manager.add(createBasicAliasAnalysisPass());		// -basicaa
		manager.add(createPromoteMemoryToRegisterPass());	// -mem2reg
		manager.add(createCFGSimplificationPass());		// -simplifycfg
		manager.add(createInstructionCombiningPass());		// -instcombine
		manager.add(createTailCallEliminationPass());		// -tailcallelim
		manager.add(createLoopSimplifyPass());			// -loop-simplify
		manager.add(createLCSSAPass());				// -lcssa
		manager.add(createLoopRotatePass());			// -loop-rotate
		manager.add(createLCSSAPass());				// -lcssa
		manager.add(createLoopUnswitchPass());			// -loop-unswitch
		manager.add(createInstructionCombiningPass());		// -instcombine
		manager.add(createLoopSimplifyPass());			// -loop-simplify
		manager.add(createLCSSAPass());				// -lcssa
		manager.add(createIndVarSimplifyPass());		// -indvars
		manager.add(createLoopDeletionPass());			// -loop-deletion
		manager.add(createInstructionCombiningPass());		// -instcombine		
		manager.add(createCodePreperationPass());		// -polly-prepare
		manager.add(createRegionSimplifyPass());		// -polly-region-simplify
		manager.add(createIndVarSimplifyPass());		// -indvars
		manager.add(createBasicAliasAnalysisPass());		// -basicaa
		manager.add(createScheduleOptimizerPass());		// -polly-optimize-isl
		codegenPass = createCodeGenerationPass();
		manager.add(codegenPass);				// -polly-codegen
		manager.run(*m);
	}
	
	// Emit target assembly and binary image, depending
	// on runmode.
	switch (runmode)
	{
		case KERNELGEN_RUNMODE_NATIVE :
		{
			// Create target machine for NATIVE target and get its target data.
			if (!mcpu[KERNELGEN_RUNMODE_NATIVE].get())
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
				mcpu[KERNELGEN_RUNMODE_NATIVE].reset(target->createTargetMachine(
					triple.getTriple(), "", "", Reloc::PIC_, CodeModel::Default));
				if (!mcpu[KERNELGEN_RUNMODE_NATIVE].get())
					THROW("Could not allocate target machine");

				// Override default to generate verbose assembly.
				mcpu[KERNELGEN_RUNMODE_NATIVE].get()->setAsmVerbosityDefault(true);
			}
			
			const TargetData* tdata = 
				mcpu[KERNELGEN_RUNMODE_NATIVE].get()->getTargetData();
			PassManager manager;
			manager.add(new TargetData(*tdata));

			// Setup output stream.
			string bin_string;
			raw_string_ostream bin_stream(bin_string);
			formatted_raw_ostream stream(bin_stream);

			// Ask the target to add backend passes as necessary.
			if (mcpu[KERNELGEN_RUNMODE_NATIVE].get()->addPassesToEmitFile(manager, stream,
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

			// Load linked image and extract kernel entry point.
			void* handle = dlopen(tmp2.getFilename().c_str(),
				RTLD_NOW | RTLD_GLOBAL | RTLD_DEEPBIND);
			if (!handle)
				THROW("Cannot dlopen " << dlerror());

			// Do not return anything if module is explicitly specified.
			if (module) return NULL;
			
			kernel_func_t kernel_func = (kernel_func_t)dlsym(handle, kernel->name.c_str());
			if (!kernel_func)
				THROW("Cannot dlsym " << dlerror());
			
			if (verbose)
				cout << "Loaded '" << kernel->name << "' at: " << (void*)kernel_func << endl;

			return (char*)kernel_func;
		}
		case KERNELGEN_RUNMODE_CUDA :
		{
			// Convert external functions CallInst-s into
			// host callback form. Do not convert CallInst-s
			// to device-resolvable intrinsics (syscalls and math).
			static string cuda_intrinsics[] =
			{
				#include "cuda_syscalls.h"
				#include "cuda_intrinsics.h"
				"printf",
				"kernelgen_launch"
			};
			Type* int32Ty = Type::getInt32Ty(context);
			std::vector<CallInst*> old_calls;
			Function* hostcall = Function::Create(
				TypeBuilder<void(types::i<8>*, types::i<32>*), true>::get(context),
				GlobalValue::ExternalLinkage, "kernelgen_hostcall", m);

			// Vector of wrapper functions considered for inclusion
			// into host module.
			map<Function*, string> funcs;
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

					// Extract call into separate basic block.
					BasicBlock* bbcall = SplitBlock(bb, call, codegenPass);
					if (bbcall->getInstList().size() > 1)
					{
						BasicBlock::iterator start = bbcall->begin();
						start++;
						SplitBlock(bbcall, start, codegenPass);
					}

					// Extract basic block with call into function.
					Function* func = ExtractBasicBlock(bbcall, true);
					funcs[func] = callee->getName();

					// Stop processing the current block, which is finished.
					break;
				}

			// Replace newly inserted function calls with kernelgen_hostcall.
			vector<GlobalValue*> host_calls;
			vector<CallInst*> erase_calls;
			for (Function::iterator bb = f->begin(); bb != f->end(); bb++)
				for (BasicBlock::iterator ii = bb->begin(); ii != bb->end(); ii++)
				{
					// Check if instruction in focus is a call.
					CallInst* call = dyn_cast<CallInst>(cast<Value>(ii));
					if (!call) continue;

					// Check if function is called (needs -instcombine pass).
					Function* callee = call->getCalledFunction();
					if (!callee) continue;							
					if (funcs.find(callee) == funcs.end()) continue;

					// Start forming new function call argument list
					// with aggregated struct pointer.
					Instruction* cast = CastInst::CreatePointerCast(
						call->getArgOperand(0), PointerType::getInt32PtrTy(context),
						"cast", call);
					SmallVector<Value*, 16> call_args;
					call_args.push_back(cast);
			
					// Create a constant array holding original called
					// function name.
					Constant* name = ConstantArray::get(
						context, funcs[callee], true);
			
					// Create and initialize the memory buffer for name.
					ArrayType* nameTy = dyn_cast<ArrayType>(name->getType());
					AllocaInst* nameAlloc = new AllocaInst(nameTy, "", call);
					StoreInst* nameInit = new StoreInst(name, nameAlloc, "", call);
					Value* Idx[2];
					Idx[0] = Constant::getNullValue(Type::getInt32Ty(context));
					Idx[1] = ConstantInt::get(Type::getInt32Ty(context), 0);
					GetElementPtrInst* namePtr =
						GetElementPtrInst::Create(nameAlloc, Idx, "", call);

					// Insert extra argument - the pointer to the
					// original function string name.
					call_args.insert(call_args.begin(), namePtr);

					// Create new function call with new call arguments
					// and copy old call properties.
					CallInst* newcall = CallInst::Create(
						hostcall, call_args, "", call);
					newcall->setCallingConv(call->getCallingConv());
					newcall->setAttributes(call->getAttributes());
					newcall->setDebugLoc(call->getDebugLoc());

					// Replace function from device module and add it
					// to host module, if it is not already there.					
					call->replaceAllUsesWith(newcall);
					erase_calls.push_back(call);
					callee->setVisibility(GlobalValue::DefaultVisibility);
					callee->setLinkage(GlobalValue::ExternalLinkage);
					callee->setName("__kernelgen_" + funcs[callee]);
					host_calls.push_back(callee);
				}

			for (vector<CallInst*>::iterator i = erase_calls.begin(),
				ie = erase_calls.end(); i != ie; i++)
				(*i)->eraseFromParent();

			Module* hostm = CloneModule(m);
			hostm->getFunction(kernel->name)->eraseFromParent();

			// Remove host wrappers functions from device module.
			{
				PassManager manager;
				manager.add(new TargetData(m));
				manager.add(createGVExtractionPass(host_calls, true));
				manager.add(createGlobalDCEPass());
				manager.add(createStripDeadDebugInfoPass());
				manager.add(createStripDeadPrototypesPass());
				manager.run(*m);
			}
			
			// Optimize both device and host modules.
			{
				PassManager manager;
				manager.add(createLowerSetJmpPass());
				PassManagerBuilder builder;
				builder.Inliner = createFunctionInliningPass();
				builder.OptLevel = 3;
				builder.DisableSimplifyLibCalls = true;
				builder.populateModulePassManager(manager);
				manager.run(*m);
				manager.run(*hostm);
			}

			m->dump();
			//hostm->dump();
			
			// Compile host code, using native target compiler.
			compile(KERNELGEN_RUNMODE_NATIVE, kernel, hostm);

			// Create target machine for CUDA target and get its target data.
			if (!mcpu[KERNELGEN_RUNMODE_CUDA].get())
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

				mcpu[KERNELGEN_RUNMODE_CUDA].reset(target->createTargetMachine(
					triple.getTriple(), "", "", Reloc::PIC_, CodeModel::Default));
				if (!mcpu[KERNELGEN_RUNMODE_CUDA].get())
					THROW("Could not allocate target machine");

				// Override default to generate verbose assembly.
				mcpu[KERNELGEN_RUNMODE_CUDA].get()->setAsmVerbosityDefault(true);
			}
			
			PassManager manager;

			// Setup output stream.
			string bin_string;
			raw_string_ostream bin_stream(bin_string);
			formatted_raw_ostream stream(bin_stream);

			// Ask the target to add backend passes as necessary.
			if (mcpu[KERNELGEN_RUNMODE_CUDA].get()->addPassesToEmitFile(manager, stream,
				TargetMachine::CGFT_AssemblyFile, CodeGenOpt::Aggressive))
				THROW("Target does not support generation of this file type");

			manager.run(*m);
			
			// Flush the resulting object binary to the
			// underlying string.
			stream.flush();

			cout << bin_string;

			// Dump generated kernel object to first temporary file.
			cfiledesc tmp1 = cfiledesc::mktemp("/tmp/");
			{
				fstream tmp_stream;
				tmp_stream.open(tmp1.getFilename().c_str(),
					fstream::binary | fstream::out | fstream::trunc);
				tmp_stream << bin_string;
				tmp_stream.close();
			}

			// Replace $kernel_name$ with an actual name.
			{
				string sed = "sed";
				std::list<string> sed_args;
				sed_args.push_back("-i");
				sed_args.push_back("s/\\\\\\$kernel_name\\\\\\$/" + kernel->name + "/");
				sed_args.push_back(tmp1.getFilename());
				if (verbose)
				{
					cout << sed;
					for (std::list<string>::iterator it = sed_args.begin();
						it != sed_args.end(); it++)
						cout << " " << *it;
					cout << endl;
				}
				execute(sed, sed_args, "", NULL, NULL);
			}

			// Compile CUDA code in temporary file.
			cfiledesc tmp2 = cfiledesc::mktemp("/tmp/");
			{
				string nvcc = "nvcc";
				std::list<string> nvcc_args;
				nvcc_args.push_back("-D__CUDA_DEVICE_FUNC__");
				nvcc_args.push_back("--x");
				nvcc_args.push_back("cu");
				nvcc_args.push_back("-c");
				nvcc_args.push_back(tmp1.getFilename());
				nvcc_args.push_back("-o");
				nvcc_args.push_back(tmp2.getFilename());
				if (verbose)
				{
					cout << nvcc;
                                        for (std::list<string>::iterator it = nvcc_args.begin();
                                                it != nvcc_args.end(); it++)
                                                cout << " " << *it;
                                        cout << endl;
                                }
                                execute(nvcc, nvcc_args, "", NULL, NULL);
			}
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

