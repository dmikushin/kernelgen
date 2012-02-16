//===- CodeExtractor.cpp - Pull code region into a new function -----------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the interface to tear out a code region, such as an
// individual loop or a parallel section, into a new function, replacing it with
// a call to the new function.
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Utils/FunctionUtils.h"
#include "llvm/Constants.h"
#include "llvm/DerivedTypes.h"
#include "llvm/Instructions.h"
#include "llvm/Intrinsics.h"
#include "llvm/LLVMContext.h"
#include "llvm/Module.h"
#include "llvm/Pass.h"
#include "llvm/Analysis/Dominators.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/Verifier.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/IRBuilder.h"
#include "llvm/Support/raw_os_ostream.h"
#include "llvm/Support/TypeBuilder.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Transforms/Utils/ValueMapper.h"
#include "llvm/Transforms/Utils/Cloning.h"

#include "BranchedLoopExtractor.h"

#include <algorithm>
#include <set>
#include <vector>

using namespace llvm;

#include <iostream>
using namespace std;

namespace
{
class BranchedCodeExtractor
{
	typedef SetVector<Value*> Values;
	SetVector<BasicBlock*> BlocksToExtract;

	SetVector<BasicBlock*> OriginalLoopBlocks;
	SetVector<BasicBlock*> ClonedLoopBlocks;

	DominatorTree* DT;
	unsigned NumExitBlocks;
	Type *RetTy;

public:
	BranchedCodeExtractor(DominatorTree* dt = 0)
		: DT(dt), NumExitBlocks(~0U), OriginalLoopBlocks(BlocksToExtract) {}

	CallInst* ExtractCodeRegion(const std::vector<BasicBlock*> &code);

	bool isEligible(const std::vector<BasicBlock*> &code);

private:
	/// definedInRegion - Return true if the specified value is defined in the
	/// extracted region.
	bool definedInRegion(Value *V) const {
		if (Instruction *I = dyn_cast<Instruction>(V))
			if (BlocksToExtract.count(I->getParent()))
				return true;
		return false;
	}

	/// definedInCaller - Return true if the specified value is defined in the
	/// function being code extracted, not in the region being extracted.
	/// These values must be passed in as live-ins to the function.
	bool definedInCaller(Value *V) const {
		if (isa<Argument>(V)) return true;
		if (Instruction *I = dyn_cast<Instruction>(V))
			if (!BlocksToExtract.count(I->getParent()))
				return true;
		return false;
	}

	void severSplitPHINodes(BasicBlock *&Header);
	void splitReturnBlocks();
	void findInputsOutputs(Values &inputs, Values &outputs);

	void moveCodeToFunction(Function *newFunction);

	void emitCallAndSwitchStatement(Function *newFunction,
	                                BasicBlock *newHeader,
	                                Values &inputs,
	                                Values &outputs);
	void makeFunctionBody(Function * LoopFunction,
	                      BasicBlock * header,
	                      BasicBlock * clonedHeader,
	                      Values inputs, Values outputs,
	                      std::vector<BasicBlock*> & ExitBlocks,
	                      ValueToValueMapTy & allToAllMap);

	CallInst *createCallAndBranch(Function* LaunchFunc, Function* KernelFunc,
	                              Values & inputs,Values & outputs,
	                              BasicBlock * callAndBranchBlock,
	                              BasicBlock * header,
	                              BasicBlock * loadAndSwitchExitBlock,
	                              AllocaInst * &Struct );


	void createLoadsAndSwitch(CallInst *callLoopFuncInst,
	                          Values & inputs, Values & outputs,
	                          BasicBlock * loadAndSwitchExitBlock,
	                          const std::vector<BasicBlock*> & ExitBlocks,
	                          ValueToValueMapTy & OutputsToLoadInstMap,
	                          AllocaInst * Struct);

	void updatePhiNodes( Values & outputs, ValueToValueMapTy & OutputsToLoadInstMap,
	                     BasicBlock * loadAndSwitchBlock,
	                     std::vector<BasicBlock*> & ExitBlocks);

};
}

/// severSplitPHINodes - If a PHI node has multiple inputs from outside of the
/// region, we need to split the entry block of the region so that the PHI node
/// is easier to deal with.
void BranchedCodeExtractor::severSplitPHINodes(BasicBlock *&Header)
{
	unsigned NumPredsFromRegion = 0;
	unsigned NumPredsOutsideRegion = 0;

	if (Header != &Header->getParent()->getEntryBlock()) {
		// if header is not the first block in function

		PHINode *PN = dyn_cast<PHINode>(Header->begin());
		if (!PN) return;  // No PHI nodes.

		// If the header node contains any PHI nodes, check to see if there is more
		// than one entry from outside the region.  If so, we need to sever the
		// header block into two.
		for (unsigned i = 0, e = PN->getNumIncomingValues(); i != e; ++i)
			if (BlocksToExtract.count(PN->getIncomingBlock(i)))
				++NumPredsFromRegion;
			else
				++NumPredsOutsideRegion;

		// If there is one (or fewer) predecessor from outside the region, we don't
		// need to do anything special.
		if (NumPredsOutsideRegion <= 1) return;
	}

	// Otherwise, we need to split the header block into two pieces: one
	// containing PHI nodes merging values from outside of the region, and a
	// second that contains all of the code for the block and merges back any
	// incoming values from inside of the region.
	BasicBlock::iterator AfterPHIs = Header->getFirstNonPHI();
	BasicBlock *NewBB = Header->splitBasicBlock(AfterPHIs,
	                    Header->getName()+".ce");

	// We only want to code extract the second block now, and it becomes the new
	// header of the region.
	BasicBlock *OldPred = Header;
	BlocksToExtract.remove(OldPred);
	BlocksToExtract.insert(NewBB);
	Header = NewBB;

	// Okay, update dominator sets. The blocks that dominate the new one are the
	// blocks that dominate TIBB plus the new block itself.
	if (DT)
		DT->splitBlock(NewBB);

	// Okay, now we need to adjust the PHI nodes and any branches from within the
	// region to go to the new header block instead of the old header block.
	if (NumPredsFromRegion) {
		PHINode *PN = cast<PHINode>(OldPred->begin());
		// Loop over all of the predecessors of OldPred that are in the region,
		// changing them to branch to NewBB instead.
		for (unsigned i = 0, e = PN->getNumIncomingValues(); i != e; ++i)
			if (BlocksToExtract.count(PN->getIncomingBlock(i))) {
				TerminatorInst *TI = PN->getIncomingBlock(i)->getTerminator();
				TI->replaceUsesOfWith(OldPred, NewBB);
			}

		// Okay, everything within the region is now branching to the right block, we
		// just have to update the PHI nodes now, inserting PHI nodes into NewBB.
		for (AfterPHIs = OldPred->begin(); isa<PHINode>(AfterPHIs); ++AfterPHIs) {
			PHINode *PN = cast<PHINode>(AfterPHIs);
			// Create a new PHI node in the new region, which has an incoming value
			// from OldPred of PN.
			PHINode *NewPN = PHINode::Create(PN->getType(), 1 + NumPredsFromRegion,
			                                 PN->getName()+".ce", NewBB->begin());

			// All uses of old phi-nodes in BlocksToExtract replace by uses of new phi-nodes
			for(Value::use_iterator Usr = PN->use_begin(), Usr_end = PN->use_end(); Usr != Usr_end; Usr++) {
				Instruction * user = cast<Instruction>(*Usr);
				if(user && BlocksToExtract.count(user -> getParent()))
					user -> replaceUsesOfWith(PN,NewPN);
			}

			//New phi-nodes use old phi-nodes as incoming value from OldPred
			NewPN->addIncoming(PN, OldPred);

			// Loop over all of the incoming value in PN, moving them to NewPN if they
			// are from the extracted region.
			for (unsigned i = 0; i != PN->getNumIncomingValues(); ++i) {
				if (BlocksToExtract.count(PN->getIncomingBlock(i))) {
					NewPN->addIncoming(PN->getIncomingValue(i), PN->getIncomingBlock(i));
					PN->removeIncomingValue(i);
					--i;
				}
			}

		}
	}
}

void BranchedCodeExtractor::splitReturnBlocks()
{
	for (SetVector<BasicBlock*>::iterator I = BlocksToExtract.begin(),
	     E = BlocksToExtract.end(); I != E; ++I)
		if (ReturnInst *RI = dyn_cast<ReturnInst>((*I)->getTerminator())) {
			BasicBlock *New = (*I)->splitBasicBlock(RI, (*I)->getName()+".ret");
			if (DT) {
				// Old dominates New. New node dominates all other nodes dominated
				// by Old.
				DomTreeNode *OldNode = DT->getNode(*I);
				SmallVector<DomTreeNode*, 8> Children;
				for (DomTreeNode::iterator DI = OldNode->begin(), DE = OldNode->end();
				     DI != DE; ++DI)
					Children.push_back(*DI);

				DomTreeNode *NewNode = DT->addNewBlock(New, *I);

				for (SmallVector<DomTreeNode*, 8>::iterator I = Children.begin(),
				     E = Children.end(); I != E; ++I)
					DT->changeImmediateDominator(*I, NewNode);
			}
		}
}

// findInputsOutputs - Find inputs to, outputs from the code region.
//
void BranchedCodeExtractor::findInputsOutputs(Values &inputs, Values &outputs)
{
	std::set<BasicBlock*> ExitBlocks;

	SetVector<Value *> setOfIntegerInputs;
	SetVector<Value *> setOfNonIntegerInputs;

	for (SetVector<BasicBlock*>::const_iterator ci = BlocksToExtract.begin(),
	     ce = BlocksToExtract.end(); ci != ce; ++ci) {
		BasicBlock *BB = *ci;

		for (BasicBlock::iterator I = BB->begin(), E = BB->end(); I != E; ++I) {
			// If a used value is defined outside the region, it's an input.  If an
			// instruction is used outside the region, it's an output.

			// Let all integer inputs to go first (to simplify
			// generated kernels identification).
			for (User::op_iterator O = I->op_begin(), E = I->op_end(); O != E; ++O)
				if (definedInCaller(*O) && (*O)->getType()->isIntegerTy())
					if(!setOfIntegerInputs.count(*O))
							setOfIntegerInputs.insert(*O);
					
			for (User::op_iterator O = I->op_begin(), E = I->op_end(); O != E; ++O)
				if (definedInCaller(*O) && !(*O)->getType()->isIntegerTy())
					if(!setOfNonIntegerInputs.count(*O))
							setOfNonIntegerInputs.insert(*O);
							
			// Consider uses of this instruction (outputs).
			for (Value::use_iterator UI = I->use_begin(), E = I->use_end();
			     UI != E; ++UI)
				if (!definedInRegion(*UI)) {
					outputs.insert(I);
					break;
				}
		} // for: insts

     
		// Keep track of the exit blocks from the region.
		TerminatorInst *TI = BB->getTerminator();
		for (unsigned i = 0, e = TI->getNumSuccessors(); i != e; ++i)
			if (!BlocksToExtract.count(TI->getSuccessor(i)))
				ExitBlocks.insert(TI->getSuccessor(i));
	} // for: basic blocks
    inputs.insert(setOfIntegerInputs.begin(), setOfIntegerInputs.end());
    inputs.insert(setOfNonIntegerInputs.begin(), setOfNonIntegerInputs.end());
	
	NumExitBlocks = ExitBlocks.size();
}

/// FindPhiPredForUseInBlock - Given a value and a basic block, find a PHI
/// that uses the value within the basic block, and return the predecessor
/// block associated with that use, or return 0 if none is found.
static BasicBlock* FindPhiPredForUseInBlock(Value* Used, BasicBlock* BB)
{
	for (Value::use_iterator UI = Used->use_begin(),
	     UE = Used->use_end(); UI != UE; ++UI) {
		PHINode *P = dyn_cast<PHINode>(*UI);
		if (P && P->getParent() == BB)
			return P->getIncomingBlock(UI);
	}

	return 0;
}

bool BranchedCodeExtractor::isEligible(const std::vector<BasicBlock*> &code)
{
	// Deny code region if it contains allocas or vastarts.
	for (std::vector<BasicBlock*>::const_iterator BB = code.begin(), e=code.end();
	     BB != e; ++BB)
		for (BasicBlock::const_iterator I = (*BB)->begin(), Ie = (*BB)->end();
		     I != Ie; ++I)
			if (isa<AllocaInst>(*I))
				return false;
			else if (const CallInst *CI = dyn_cast<CallInst>(I))
				if (const Function *F = CI->getCalledFunction())
					if (F->getIntrinsicID() == Intrinsic::vastart)
						return false;
	return true;
}

SetVector<BasicBlock *> * CloneCodeRegion(const SetVector<BasicBlock*> &code,
        RemapFlags remapFlags, ValueToValueMapTy & VMap,
        const char *NameSuffix, ClonedCodeInfo *CodeInfo)
{

	SetVector<BasicBlock*> *NewCodeRegion	= new SetVector<BasicBlock*>();

	// Loop over all of the basic blocks in the function, cloning them as
	// appropriate.  Note that we save BE this way in order to handle cloning of
	// recursive functions into themselves.

	for (SetVector<BasicBlock*>::const_iterator BI = code.begin(), BE = code.end();
	     BI != BE; ++BI) {

		const BasicBlock * BB = *BI;

		// Create a new basic block and copy instructions into it!
		BasicBlock *CBB = CloneBasicBlock(BB, VMap, NameSuffix, NULL, CodeInfo);
		VMap[BB] = CBB;                       // Add basic block mapping.
		//NewCodeRegion->push_back(CBB);
		NewCodeRegion->insert(CBB);

		assert(!dyn_cast<ReturnInst>(CBB->getTerminator()) &&
		       "there can not be return instructions");
	}

	// Loop over all of the instructions in the function, fixing up operand
	// references as we go.  This uses VMap to do all the hard work.

	for (SetVector<BasicBlock*>::iterator BB = NewCodeRegion -> begin(),
	     BE = NewCodeRegion -> end(); BB != BE; ++BB)
		// Loop over all instructions, fixing each one as we find it...
		for (BasicBlock::iterator II = (*BB)->begin(); II != (*BB)->end(); ++II)
			RemapInstruction(II, VMap, remapFlags);

	return NewCodeRegion;
}

void BranchedCodeExtractor::makeFunctionBody(Function * LoopFunction,
        BasicBlock * header,
        BasicBlock * clonedHeader,
        Values inputs, Values outputs,
        std::vector<BasicBlock*> & ExitBlocks, ValueToValueMapTy & allToAllMap)
{
	LLVMContext &context = header->getContext();

	// The new function needs a root node because other nodes can branch to the
	// head of the region, but the entry node of a function cannot have preds.
	BasicBlock *FuncRoot = BasicBlock::Create(context,"Loop Function Root");
	FuncRoot->getInstList().push_back(BranchInst::Create(clonedHeader));
	LoopFunction->getBasicBlockList().push_back(FuncRoot);

	// insert cloned loop body

	Function::BasicBlockListType &functionBlocks = LoopFunction->getBasicBlockList();
	for (SetVector<BasicBlock*>::const_iterator i = ClonedLoopBlocks.begin(),
	     e = ClonedLoopBlocks.end(); i != e; ++i) {
		// Insert this basic block into the loop function
		functionBlocks.push_back(*i);
	}

	// Select the last argument of the loop function.
	Function::arg_iterator AI = LoopFunction->arg_end();
	AI--;

	// Fill the arguments types structure.
	// First, place pointer to the function type.
	// Second, place pointer to the structure itself.
	std::vector<Type*> paramTy;
	paramTy.push_back(Type::getInt8PtrTy(context));
	paramTy.push_back(Type::getInt8PtrTy(context));

	// Add the types of the input values to the function's argument list
	for (Values::const_iterator i = inputs.begin(), e = inputs.end(); i != e; ++i)
		paramTy.push_back((*i)->getType());

	// Add the types of the output values to the function's argument list.
	for (Values::const_iterator I = outputs.begin(), E = outputs.end(); I != E; ++I)
		paramTy.push_back((*I)->getType());

	// Aggregate args types into struct type
	PointerType *StructPtrType;
	if (inputs.size() + outputs.size() > 0)
		StructPtrType = PointerType::getUnqual(
		                    StructType::get(context, paramTy, false /* isPacked */));

	Value* structArg = CastInst::CreatePointerCast(AI, StructPtrType, "",
	                   FuncRoot->getTerminator());

	// Rewrite all users of the inputs in the cloned region to use the
	// arguments (or appropriate addressing into struct) instead.
	for (unsigned i = 0, e = inputs.size(); i != e; i++) {
		Value* RewriteVal;

		Value *Idx[2];
		Idx[0] = Constant::getNullValue(Type::getInt32Ty(context));
		Idx[1] = ConstantInt::get(Type::getInt32Ty(context), i + 2);

		// Terminator of function root
		TerminatorInst *TI = FuncRoot->getTerminator();

		// Create instruction to take address of "inputs[i]" in struct,
		// insert it before terminator.
		GetElementPtrInst *GEP = GetElementPtrInst::Create(
		                             structArg, Idx, "load_ptr_" + inputs[i]->getName(), TI);

		// create LoadInstruction from adress, which returned by instruction GEP
		// inserted it before terminator
		RewriteVal = new LoadInst(GEP, "load_" + inputs[i]->getName(), TI);

		// users of argument "inputs[i]"
		std::vector<User*> Users(inputs[i]->use_begin(), inputs[i]->use_end());
		for (std::vector<User*>::iterator user = Users.begin(), userE = Users.end();
		     user != userE; ++user)
			if (Instruction* inst = dyn_cast<Instruction>(*user))
				if (ClonedLoopBlocks.count(inst->getParent()))
					inst->replaceUsesOfWith(inputs[i], RewriteVal);
	}

	// Loop over all of the PHI nodes in the header block, and change any
	// references to the old incoming edge to be the new incoming edge.
	for (BasicBlock::iterator I = clonedHeader->begin(); isa<PHINode>(I); ++I) {
		PHINode *PN = cast<PHINode>(I);
		for (unsigned i = 0, e = PN->getNumIncomingValues(); i != e; ++i)
			if (!ClonedLoopBlocks.count(PN->getIncomingBlock(i)))
				PN->setIncomingBlock(i, FuncRoot);
	}


	///////////////////////////////////////////////
	// insert stores to outputs and return block //
	///////////////////////////////////////////////


	// Since there may be multiple exits from the original region, make the new
	// function return an unsigned, switch on that number. This loop iterates
	// over all of the blocks in the fuction body (ClonedLoopBlocks), updating any terminator
	// instructions in the fuction body that branch to blocks that are
	// not in it. In every created ReturnBlock insert StoreInst for each output.
	Function::arg_iterator OutputArgBegin = LoopFunction->arg_begin();
	unsigned FirstOut = inputs.size() + 2;
	std::advance(OutputArgBegin, FirstOut);

	std::map<BasicBlock*, BasicBlock*> ExitBlockMap;
	unsigned switchVal = 0;

	for (SetVector<BasicBlock*>::const_iterator i = ClonedLoopBlocks.begin(),
	     e = ClonedLoopBlocks.end(); i != e; ++i) {
		TerminatorInst *TI = (*i)->getTerminator();
		for (unsigned i = 0, e = TI->getNumSuccessors(); i != e; ++i)
			if (!ClonedLoopBlocks.count(TI->getSuccessor(i))) {
				BasicBlock *OldTarget = TI->getSuccessor(i);
				// add a new basic block to map, which returns the appropriate value
				BasicBlock *&NewTarget = ExitBlockMap[OldTarget];

				if (!NewTarget) {
					// If we don't already have an exit stub for this
					// destination, create one in the end of LoopFunction

					NewTarget = BasicBlock::Create(context,
					                               OldTarget->getName() + ".exitStub",
					                               LoopFunction);
					unsigned SuccNum = switchVal++;
					ExitBlocks.push_back(OldTarget);

					Value * brVal = ConstantInt::get(Type::getInt32Ty(context), SuccNum);

					// TODO: do not allow loops jumping somewhere outside
					//ReturnInst *NTRet = ReturnInst::Create(context, brVal, NewTarget);
					ReturnInst *NTRet = ReturnInst::Create(context, 0, NewTarget);

					// Restore values just before we exit
					Function::arg_iterator OAI = OutputArgBegin;
					for (unsigned out = 0, e = outputs.size(); out != e; ++out) {
						Value *Idx[2];
						Idx[0] = Constant::getNullValue(Type::getInt32Ty(context));
						Idx[1] = ConstantInt::get(Type::getInt32Ty(context),
						                          FirstOut+out);

						GetElementPtrInst *GEP = GetElementPtrInst::Create(
						                             structArg, Idx, "store_ptr_" + outputs[out]->getName(),
						                             NTRet);

						new StoreInst(allToAllMap[outputs[out]], GEP, NTRet);
					}
				}
				// rewrite the original branch instruction with this new target
				TI->setSuccessor(i, NewTarget);
				//ExitBlockMap[OldTarget] = NewTarget;
			}

	}
	assert(NumExitBlocks == ExitBlocks.size() && "have to handle all exit blocks");

	return;
}

CallInst* BranchedCodeExtractor::createCallAndBranch(
    Function* LaunchFunc, Function* KernelFunc,
    Values& inputs, Values& outputs,
    BasicBlock* callAndBranchBlock, BasicBlock* header,
    BasicBlock* loadAndSwitchExitBlock, AllocaInst* &Struct)
{
	std::vector<Value*> params, StructValues, ReloadOutputs, Reloads;

	LLVMContext &context = LaunchFunc->getContext();

	// Add inputs as params, or to be filled into the struct.
	// Also calculate the number of integer fields.
	unsigned int numints = 0;
	for (Values::iterator i = inputs.begin(), e = inputs.end(); i != e; ++i) {
		StructValues.push_back(*i);

		// Calculate the total size of integer inputs.
		Type* type = (*i)->getType();
		if (type->isIntegerTy()) numints++;
	}

	// Create allocas for the outputs
	for (Values::iterator i = outputs.begin(), e = outputs.end(); i != e; ++i)
		StructValues.push_back(*i);

	// Fill the arguments types structure.
	// First, place pointer to the function type.
	// Second, place pointer to the structure itself.
	std::vector<Type*> ArgTypes;
	ArgTypes.push_back(Type::getInt8PtrTy(context));
	ArgTypes.push_back(Type::getInt8PtrTy(context));
	for (Values::iterator v = StructValues.begin(),
	     ve = StructValues.end(); v != ve; ++v)
		ArgTypes.push_back((*v)->getType());

	// Allocate memory for the struct at the beginning of
	// function, which contains the Loop.
	StructType* StructArgTy = StructType::get(
	                              context, ArgTypes, false /* isPacked */);
	Struct = new AllocaInst(StructArgTy, 0, "",
	                        callAndBranchBlock->getParent()->begin()->begin());

	// Initially, fill struct with zeros.
	IRBuilder<> Builder(
	    callAndBranchBlock->getParent()->begin()->begin());
	CallInst* MI = Builder.CreateMemSet(Struct,
	                                    Constant::getNullValue(Type::getInt8Ty(context)),
	                                    ConstantExpr::getSizeOf(StructArgTy), 1);

	Value* Idx[2];
	Idx[0] = Constant::getNullValue(Type::getInt32Ty(context));

	// Store input values to arguments struct.
	for (unsigned i = 0, e = inputs.size(); i != e; ++i) {
		Idx[1] = ConstantInt::get(Type::getInt32Ty(context), i + 2);
		GetElementPtrInst *GEP = GetElementPtrInst::Create(
		                             Struct, Idx, "" + StructValues[i]->getName(),
		                             callAndBranchBlock);
		StoreInst *SI = new StoreInst(StructValues[i], GEP, false,
		                              callAndBranchBlock);
	}

	// Create a constant array holding original called
	// function name.
	Constant* name = ConstantArray::get(
	                     context, KernelFunc->getName(), true);

	// Create and initialize the memory buffer for name.
	ArrayType* nameTy = cast<ArrayType>(name->getType());
	AllocaInst* nameAlloc = new AllocaInst(nameTy, "", callAndBranchBlock);
	StoreInst* nameInit = new StoreInst(name, nameAlloc, false, callAndBranchBlock);
	Idx[1] = ConstantInt::get(Type::getInt32Ty(context), 0);
	GetElementPtrInst* namePtr = GetElementPtrInst::Create(
	                                 nameAlloc, Idx, "", callAndBranchBlock);

	// Add pointer to the original function string name.
	params.push_back(namePtr);

	// Store the size of the aggregated arguments struct
	// to the new call arguments list.
	params.push_back(ConstantExpr::getSizeOf(StructArgTy));

	// Store the total size of all integer fields in
	// aggregated arguments struct.
	Constant* size =
	    Constant::getNullValue(Type::getInt64Ty(context));;
	if (numints)
		/*size = ConstantExpr::getAdd(size, ConstantExpr::getAdd(
			ConstantExpr::getOffsetOf(StructArgTy, numints),
			ConstantExpr::getSizeOf(ArgTypes[numints - 1])));*/
		size = ConstantExpr::getSub(
		           ConstantExpr::getAdd(ConstantExpr::getOffsetOf(StructArgTy, (numints-1) + 2), //смещение последнего целочисленного параметра
		                                ConstantExpr::getSizeOf(ArgTypes[(numints - 1) + 2])),         //размер последнего целочисленного параметра
		           ConstantExpr::getOffsetOf(StructArgTy,  2));                                  //смещение первого параметра
	params.push_back(size);

	// Store the pointer to the aggregated arguments struct
	// to the new call args list.
	Instruction* IntPtrToStruct = CastInst::CreatePointerCast(
	                                  Struct, PointerType::getInt32PtrTy(context), "", callAndBranchBlock);
	params.push_back(IntPtrToStruct);

	// Emit the call to the function
	CallInst* call = CallInst::Create(
	                     LaunchFunc, params, NumExitBlocks > 1 ? "targetBlock" : "");
	callAndBranchBlock->getInstList().push_back(call);

	Value* Cond = new ICmpInst(*callAndBranchBlock, ICmpInst::ICMP_EQ,
	                           call, ConstantInt::get(Type::getInt32Ty(context), -1));
	BranchInst::Create(header, loadAndSwitchExitBlock, Cond, callAndBranchBlock);

	return call;
}

void BranchedCodeExtractor::createLoadsAndSwitch(
    CallInst *callLoopFuncInst,
    Values & inputs, Values & outputs,
    BasicBlock * loadAndSwitchExitBlock,
    const std::vector<BasicBlock*> & ExitBlocks,
    ValueToValueMapTy & OutputsToLoadInstMap,
    AllocaInst * Struct)
{
	LLVMContext &Context = callLoopFuncInst-> getCalledFunction()->getContext();
	unsigned FirstOut = inputs.size() + 2;

	// Reload the outputs passed in by reference
	for (unsigned i = 0, e = outputs.size(); i != e; ++i) {
		Value *Output = 0;

		Value *Idx[2];
		Idx[0] = Constant::getNullValue(Type::getInt32Ty(Context));
		Idx[1] = ConstantInt::get(Type::getInt32Ty(Context), FirstOut + i);

		GetElementPtrInst *GEP = GetElementPtrInst::Create(
		                             Struct, Idx,"gep_reload_" + outputs[i]->getName());
		loadAndSwitchExitBlock->getInstList().push_back(GEP);
		Output = GEP;

		// create LoadInst to load value of "outputs[i]" from specified address
		LoadInst *load = new LoadInst(Output, outputs[i]->getName()+".reload");
		loadAndSwitchExitBlock->getInstList().push_back(load);

		// map outputs[i] to created load instruction
		// late, we use that map to update relationshpis
		OutputsToLoadInstMap[outputs[i]] = load;
	}

	// Now we can emit a switch statement using the call as a value.
	SwitchInst *TheSwitch =
	    SwitchInst::Create(callLoopFuncInst,
	                       loadAndSwitchExitBlock, // заменить на блок, который выдавал бы ошибку
	                       NumExitBlocks?NumExitBlocks:1 , loadAndSwitchExitBlock);

	Type *OldFnRetTy = loadAndSwitchExitBlock->getParent()->getReturnType();
	switch (NumExitBlocks) {
	case 0:
		// There are no successors (the block containing the switch itself), which
		// means that previously this was the last part of the function, and hence
		// this should be rewritten as a `ret'

		// Check if the function should return a value
		if (OldFnRetTy->isVoidTy()) {
			ReturnInst::Create(Context, 0, TheSwitch);  // Return void
		} else if (OldFnRetTy == TheSwitch->getCondition()->getType()) {
			// return what we have
			ReturnInst::Create(Context, TheSwitch->getCondition(), TheSwitch);
		} else {
			// Otherwise we must have code extracted an unwind or something, just
			// return whatever we want.
			ReturnInst::Create(Context,
			                   Constant::getNullValue(OldFnRetTy), TheSwitch);
		}

		TheSwitch->eraseFromParent();

		break;
	case 1:
		// Only a single destination, change the switch into an unconditional
		// branch.
		BranchInst::Create(ExitBlocks[0], TheSwitch);
		TheSwitch->eraseFromParent();
		break;
	case 2:
		BranchInst::Create(ExitBlocks[1], ExitBlocks[0],
		                   callLoopFuncInst, TheSwitch);
		TheSwitch->eraseFromParent();
		break;
	default:
		// Otherwise, add case fo every ExitBlock
		for(int ExitBlock = 0; ExitBlock < NumExitBlocks; ExitBlock++)
			TheSwitch -> addCase(ConstantInt::get(Type::getInt32Ty(Context),ExitBlock),
			                     ExitBlocks[ExitBlock]);
		break;
	}
}

void BranchedCodeExtractor::updatePhiNodes(
    Values& outputs, ValueToValueMapTy& OutputsToLoadInstMap,
    BasicBlock* loadAndSwitchBlock,
    std::vector<BasicBlock*>& ExitBlocks)
{
	SetVector<Value *> Users;
	if(!NumExitBlocks) return;
	ValueToValueMapTy OutputsToPhiNodes[NumExitBlocks];
	for(int i = 0; i < NumExitBlocks; i++  ) {
		BasicBlock * ExitBlock = ExitBlocks[i];

		unsigned NumPreds = 0;
		for(pred_iterator predBlock = pred_begin(ExitBlock), predBlockE = pred_end(ExitBlock);
		    predBlock != predBlockE; predBlock++)
			NumPreds++;


		for(Values::iterator Out = outputs.begin(), Outputs_end = outputs.end(); Out != Outputs_end; Out++) {
			Instruction *I = dyn_cast<Instruction>(*Out);
			bool catched = false;

			Users.clear();
			SetVector<Value *> Users(I->use_begin(),I->use_end());

			for(BasicBlock::iterator Inst = ExitBlock->begin(), E = ExitBlock->getFirstNonPHI();
			    Inst != E; Inst++) {
				PHINode * phi_node = dyn_cast<PHINode>(Inst);
				if(Users.count(phi_node)) {
					catched = true;
					phi_node ->  addIncoming(OutputsToLoadInstMap[I],loadAndSwitchBlock);
					OutputsToPhiNodes[i][I] = phi_node;
					break;
				}
			}
			if(!catched) {
				// add new phi-node
				PHINode * newPN = PHINode::Create(I->getType(), NumPreds,
				                                  "", ExitBlock -> getFirstNonPHI());

				for(pred_iterator predBlock = pred_begin(ExitBlock), predBlockE = pred_end(ExitBlock);
				    predBlock != predBlockE; predBlock++)
					if( *predBlock != loadAndSwitchBlock) {
						if(OriginalLoopBlocks.count(*predBlock)) newPN -> addIncoming(I, *predBlock);
						else
							newPN -> addIncoming(UndefValue::get(I->getType()), *predBlock);
					} else newPN -> addIncoming(OutputsToLoadInstMap[I],loadAndSwitchBlock);
				OutputsToPhiNodes[i][I] = newPN;
			}

		}
	}

	/////////////////////////
	// for current version //
	/////////////////////////

	for(Values::iterator Out = outputs.begin(), Outputs_end = outputs.end();
	    Out != Outputs_end; Out++) {
		Instruction *I = dyn_cast<Instruction>(*Out);
		for (Value::use_iterator UI = I->use_begin(), E = I->use_end(); UI != E; ++UI) {
			Instruction * User = dyn_cast<Instruction>(*UI);
			if(User && !OriginalLoopBlocks.count( User->getParent()) && User != OutputsToPhiNodes[0][I]) {
				User->replaceUsesOfWith(I,OutputsToPhiNodes[0][I]);
			}
		}
	}


}

/// ExtractRegion - Removes a loop from a function, replaces it with a call to
/// new function. Returns pointer to the new function.
///
/// algorithm:
///
/// find inputs and outputs for the region
///
/// for inputs: add to function as args, map input instr* to arg#
/// for outputs: add allocas for scalars,
///             add to func as args, map output instr* to arg#
///
/// rewrite func to use argument #s instead of instr*
///
/// for each scalar output in the function: at every exit, store intermediate
/// computed result back into memory.
///
CallInst *BranchedCodeExtractor::
ExtractCodeRegion(const std::vector<BasicBlock*> &code)
{
	if (!isEligible(code)) return NULL;

	// 1) Find inputs, outputs
	// 2) Construct new function
	//  * Add allocas for defs, pass as args by reference
	//  * Pass in uses as args
	// 3) Move code region, add call instr to func

	BlocksToExtract.insert(code.begin(), code.end());

	Values inputs;
	Values outputs; // instructions in !!OriginalLoop!!, which used outside the Loop

	// Assumption: this is a single-entry code region, and the header is the first
	// block in the region.
	BasicBlock *header = code[0];

	// Check if the region has edges from outside region of CFG
	for (unsigned i = 1, e = code.size(); i != e; ++i)
		for (pred_iterator PI = pred_begin(code[i]), E = pred_end(code[i]);
		     PI != E; ++PI) {
			// Make assertion if predesesor is not in the region
			assert(BlocksToExtract.count(*PI) &&
			       "No blocks in this region may have entries from outside the region"
			       " except for the first block!");
		}

	// If we have to split PHI nodes or the entry block, do so now.
	severSplitPHINodes(header);

	// If we have any return instructions in the region, split those blocks so
	// that the return is not in the region.
	splitReturnBlocks();

	// Find inputs to, outputs from the code region.
	findInputsOutputs(inputs, outputs);

	assert((NumExitBlocks != 0) &&
	       "there must be at least one exit block from code region"
	       "for current version");

	assert((NumExitBlocks == 1 ||  outputs.size()==0) &&
	       "there can be only one exit block from code region"
	       "or can not by any outputs"
	       "for current version");

	ClonedCodeInfo CodeInfo;
	ValueToValueMapTy VMap;
	SetVector<BasicBlock*>* clonedCode = CloneCodeRegion(
	        BlocksToExtract, RF_IgnoreMissingEntries, VMap, ".cloned", &CodeInfo);

	ClonedLoopBlocks.insert(clonedCode->begin(),clonedCode->end());
	OriginalLoopBlocks.insert(BlocksToExtract.begin(), BlocksToExtract.end());

	LLVMContext& context = header->getContext();
	Function *parentFunction = header->getParent();
	Module* m = parentFunction->getParent();

	// This takes place of the original loop
	BasicBlock *loadAndSwitchExitBlock = BasicBlock::Create(
	        header->getContext(), "loadOutputsAndSwitchExit",
	        parentFunction, header);
	BasicBlock *callAndBranchBlock = BasicBlock::Create(
	                                     header->getContext(), "callAndBranch",
	                                     parentFunction, loadAndSwitchExitBlock);

	// Add launch function declaration to module, if it is not already there.
	Function* launchFunction = m->getFunction("kernelgen_launch");
	if (!launchFunction)
		launchFunction = Function::Create(
		                     TypeBuilder<types::i<32>(types::i<8>*, types::i<64>,
		                             types::i<64>, types::i<32>*), true>::get(context),
		                     GlobalValue::ExternalLinkage, "kernelgen_launch", m);

	// Construct new loop function based on inputs/outputs & add allocas for all defs.
	// This function returns unsigned, outputs will go back by reference.
	Function* loopFunction = Function::Create(
	                             TypeBuilder<void(types::i<32>*), true>::get(context),
	                             GlobalValue::GlobalValue::ExternalLinkage,
	                             parentFunction->getName() + "_loop_" + header->getName(), m);
	Function::arg_iterator AI = loopFunction->arg_begin();
	AI -> setName("args");
	// Reset to default visibility.
	loopFunction->setVisibility(GlobalValue::DefaultVisibility);

	// If the old function is no-throw, so is the new one.
	if (parentFunction->doesNotThrow())
		loopFunction->setDoesNotThrow(true);

	// Rename Blocks.
	for (SetVector<BasicBlock*>::iterator BB = OriginalLoopBlocks.begin(),
	     BB_end = OriginalLoopBlocks.end(); BB != BB_end; BB++)
		(*BB)->setName((*BB)->getName() + ".orig");

	header->setName(header->getName() + ".header");
	BasicBlock* clonedHeader = cast<BasicBlock>(VMap[header]);
	clonedHeader->setName(clonedHeader->getName() + ".header");

	// Make function body.
	std::vector<BasicBlock*> ExitBlocks;
	makeFunctionBody(loopFunction, header, clonedHeader,
	                 inputs, outputs, ExitBlocks, VMap);

	// Make call and make branch after call.
	AllocaInst* Struct;
	CallInst* callLoopFuctionInst = createCallAndBranch(
	                                    launchFunction, loopFunction,
	                                    inputs, outputs, callAndBranchBlock,
	                                    header, loadAndSwitchExitBlock, Struct);

	// Rewrite branches to basic blocks outside of the loop to new dummy blocks
	// within the new function. This must be done before we lose track of which
	// blocks were originally in the code region.
	std::vector<User*> Users(header->use_begin(), header->use_end());
	for (unsigned i = 0, e = Users.size(); i != e; i++) {
		// The BasicBlock which contains the branch is not in the region
		// modify the branch target to a new block
		if (TerminatorInst* TI = dyn_cast<TerminatorInst>(Users[i]))
			if (!OriginalLoopBlocks.count(TI->getParent()) &&
			    (callAndBranchBlock != TI->getParent()) &&
			    TI->getParent()->getParent() == parentFunction) {
				BasicBlock * outsideBlock = TI -> getParent();

				// Loop over all phi-nodes
				for (BasicBlock::iterator PN = header->begin(); isa<PHINode>(PN); PN++) {
					PHINode * PHI_node = dyn_cast<PHINode>(PN);
					for (int v = 0; v < PHI_node->getNumIncomingValues(); v++)
						if (PHI_node->getIncomingBlock(v) == outsideBlock)
							PHI_node->setIncomingBlock(v, callAndBranchBlock);
				}
				TI->replaceUsesOfWith(header, callAndBranchBlock);
			}
	}

	// Create load instructions to load from each output
	// Insert switch instruction to select one of the exit blocks
	ValueToValueMapTy OutputsToLoadInstMap;
	createLoadsAndSwitch(callLoopFuctionInst, inputs,outputs,
	                     loadAndSwitchExitBlock, ExitBlocks, OutputsToLoadInstMap,Struct);

	// Look at all successors of the codeReplacer block. If any of these blocks
	// had PHI nodes in them, we need to add new branch from loadAndSwtchExitBlock
	for (int i = 0; i < NumExitBlocks; i++) {
		BasicBlock * ExitBlock = ExitBlocks[i];
		for (BasicBlock::iterator I = ExitBlock -> begin(); isa<PHINode>(I); ++I) {
			PHINode* PN = cast<PHINode>(I);
			for (unsigned i = 0, e = PN->getNumIncomingValues(); i != e; i++)
				if (OriginalLoopBlocks.count(PN->getIncomingBlock(i)) &&
				    !outputs.count(PN -> getIncomingValue(i))) {
					PN->addIncoming(PN->getIncomingValue(i),
					                loadAndSwitchExitBlock);
					break;
				}
		}
	}

	// Create or update PHI-nodes for each output
	// It's wery important, because we added new way in CFG to compute outputs
	updatePhiNodes(outputs, OutputsToLoadInstMap, loadAndSwitchExitBlock, ExitBlocks);

	DT->DT->recalculate(*parentFunction);
	DT->DT->recalculate(*loopFunction);

	if (verifyFunction(*loopFunction))
		cout << "verifyFunction failed!";
	return callLoopFuctionInst;
}

// ExtractBasicBlock - slurp a natural loop into a brand new function.
CallInst* llvm::BranchedExtractLoop(DominatorTree &DT, Loop* L)
{
	return BranchedCodeExtractor(&DT).ExtractCodeRegion(L->getBlocks());
}
