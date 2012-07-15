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

#include "llvm/Constants.h"
#include "llvm/LLVMContext.h"
#include "llvm/Module.h"
#include "llvm/Support/IRBuilder.h"
#include "llvm/Support/TypeBuilder.h"

#include <dlfcn.h>

using namespace kernelgen::runtime;
using namespace llvm;
using namespace std;

// Wrap call instruction into host function call wrapper.
CallInst* kernelgen::runtime::wrapCallIntoHostcall(CallInst* call, kernel_t* kernel)
{
	LLVMContext &context = getGlobalContext();
	Function* callee = call->getCalledFunction();

	if (verbose)
		cout << "Host call: " << callee->getName().data() << endl;

	// Locate entire hostcall in the native code.
	void* host_func = (void*)dlsym(NULL, callee->getName().data());
	if (!host_func) THROW("Cannot dlsym " << dlerror());
	
	kernel->target[KERNELGEN_RUNMODE_NATIVE].binary = (kernel_func_t)host_func;

	// The host call launcher prototype to be added
	// to entire module.
	Module* m = call->getParent()->getParent()->getParent(); 
	Function* hostcall = m->getFunction("kernelgen_hostcall");
	if (!hostcall)
		hostcall = Function::Create(
			TypeBuilder<void(types::i<8>*, types::i<64>,
				types::i<64>, types::i<32>*), true>::get(context),
			GlobalValue::ExternalLinkage, "kernelgen_hostcall", m);

	// Fill the arguments types structure.
	// First, place pointer to the function type.
	// Second, place pointer to the structure itself.
	std::vector<Type*> ArgTypes;
	ArgTypes.push_back(Type::getInt8PtrTy(context));
	ArgTypes.push_back(Type::getInt8PtrTy(context));
	for (unsigned i = 0, e = call->getNumArgOperands(); i != e; ++i)
		ArgTypes.push_back(call->getArgOperand(i)->getType());

	// Lastly, add the type of return value, if not void.
	// First, store pointer to return value, and then store
	// the actual value placeholder itself:
	// struct {
	//     ...
	//     retTy* pret;
	//     retTy ret;
	// };
	// This way transparency between host & device memory
	// buffer for return value could be organized and handled
	// in the same way as for pointer arguments.
	Type* retTy = callee->getReturnType();
	if (!retTy->isVoidTy())
	{
		ArgTypes.push_back(retTy->getPointerTo());
		ArgTypes.push_back(retTy);
	}

	// Allocate memory for the struct.
	StructType *StructArgTy = StructType::get(
		context, ArgTypes, false /* isPacked */);
	AllocaInst* Struct = new AllocaInst(StructArgTy, 0, "", call);

	// Initially, fill struct with zeros.
	IRBuilder<> Builder(call);
	CallInst* MI = Builder.CreateMemSet(Struct,
		Constant::getNullValue(Type::getInt8Ty(context)),
		ConstantExpr::getSizeOf(StructArgTy), 1);

	Value* Idx[2];
	Idx[0] = Constant::getNullValue(Type::getInt32Ty(context));

	// Store the function type.
	{
		Idx[1] = ConstantInt::get(Type::getInt32Ty(context), 0);
		GetElementPtrInst *GEP = GetElementPtrInst::Create(
			Struct, Idx, "", call);
		Type* type = callee->getFunctionType();
		StoreInst* SI = new StoreInst(ConstantExpr::getIntToPtr(
			ConstantInt::get(Type::getInt64Ty(context),
			(uint64_t)type), Type::getInt8PtrTy(context)),
			GEP, false, call);
	}

	// Store the struct type itself.
	{
		Idx[1] = ConstantInt::get(Type::getInt32Ty(context), 1);
		GetElementPtrInst *GEP = GetElementPtrInst::Create(
			Struct, Idx, "", call);
		StructType* StructArgTy = StructType::get(
			context, ArgTypes, false /* isPacked */);
		StoreInst* SI = new StoreInst(ConstantExpr::getIntToPtr(
			ConstantInt::get(Type::getInt64Ty(context),
			(uint64_t)StructArgTy), Type::getInt8PtrTy(context)),
			GEP, false, call);
	}

    	// Store actual arguments to arguments struct.
	for (unsigned i = 0, e = call->getNumArgOperands(); i != e; ++i)
	{
		Idx[1] = ConstantInt::get(Type::getInt32Ty(context), i + 2);
		GetElementPtrInst *GEP = GetElementPtrInst::Create(
			Struct, Idx, "", call);
		StoreInst* SI = new StoreInst(call->getArgOperand(i), GEP, false, call);
	}

	// Store pointer to return value buffer: pret = &ret.
	if (!retTy->isVoidTy())
	{
		Idx[1] = ConstantInt::get(Type::getInt32Ty(context),
			call->getNumArgOperands() + 2);
		GetElementPtrInst* GEP1 = GetElementPtrInst::Create(
			Struct, Idx, "", call);
		Idx[1] = ConstantInt::get(Type::getInt32Ty(context),
			call->getNumArgOperands() + 3);
		GetElementPtrInst* GEP2 = GetElementPtrInst::Create(
			Struct, Idx, "", call);
		StoreInst* SI = new StoreInst(GEP2, GEP1, false, call);
	}

	// Store pointer to the host call function entry point.
	SmallVector<Value*, 16> call_args;
	call_args.push_back(ConstantExpr::getIntToPtr(
		ConstantInt::get(Type::getInt64Ty(context),
		(uint64_t)kernel), Type::getInt8PtrTy(context)));

	// Store the sizeof structure.
	call_args.push_back(ConstantExpr::getSizeOf(StructArgTy));

	// TODO: store szdatai.
	call_args.push_back(Constant::getNullValue(Type::getInt64Ty(context)));

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
	newcall->setOnlyReadsMemory(false);

	// Replace function from device module.
	if (retTy->isVoidTy())
		call->replaceAllUsesWith(newcall);
	else
	{
		// Generate index.
		Idx[1] = ConstantInt::get(Type::getInt32Ty(context),
			call->getNumArgOperands() + 3);
		GetElementPtrInst* GEP = GetElementPtrInst::Create(
			Struct, Idx, "", call);
		LoadInst* LI = new LoadInst(GEP, "", call);
		call->replaceAllUsesWith(LI);
	}
	callee->setVisibility(GlobalValue::DefaultVisibility);
	callee->setLinkage(GlobalValue::ExternalLinkage);
	return newcall;
}

