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

#include <stack>
#include <algorithm>
#include <set>
#include <fstream>
#include <vector>

//#define KERNELGEN_PRIVATIZE

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

			CallInst *ExtractCodeRegion(Loop *L, LoopInfo &LI );

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
			                              AllocaInst * &Struct,
			                              ValueToValueMapTy & allToAllMap);


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
				if (definedInCaller(*O) && ((*O)->getType()->isIntegerTy() || (*O)->getType()->isPointerTy()))
					if(!setOfIntegerInputs.count(*O))
						setOfIntegerInputs.insert(*O);

			for (User::op_iterator O = I->op_begin(), E = I->op_end(); O != E; ++O)
				if (definedInCaller(*O) && !((*O)->getType()->isIntegerTy() || (*O)->getType()->isPointerTy()))
					if(!setOfNonIntegerInputs.count(*O))
						setOfNonIntegerInputs.insert(*O);

			// Consider uses of this instruction (outputs).
			for (Value::use_iterator UI = I->use_begin(), E = I->use_end();
			     UI != E; ++UI)
				if (!definedInRegion(*UI)) {
					outputs.insert(I);
					break;
				}
#ifdef KERNELGEN_PRIVATIZE			
	        for (User::op_iterator O = I->op_begin(), E = I->op_end(); O != E; ++O) {
				Value *operand = O->get();
				bool needToPrivatize;

				needToPrivatize = false;
				if (isa<GlobalValue>(*operand) && !isa<Function>(*operand))
					needToPrivatize=true;
				else if(isa<Constant>(*operand)) {
					stack<User*> notHandled;
					notHandled.push(cast<User>(operand));
					while(!notHandled.empty() && !needToPrivatize) {
						//get next operand and remove it from stack
						User *current = notHandled.top();
						notHandled.pop();
						// walk on it's operands
						int numOfCurrentOperands = current->getNumOperands();
						for(int operandIndex = 0; operandIndex < numOfCurrentOperands; operandIndex++) {
							// if it is a global - break, else save operand
							Value * operandVal = current->getOperand(operandIndex);
							if(isa<GlobalValue>(*operandVal) && !isa<Function>(*operandVal)) {
								needToPrivatize=true;
								break;
							} else if(isa<User>(*operandVal))
								notHandled.push(cast<User>(operandVal));
						}
					}
				}
				if(needToPrivatize) {
					outs() << "Seen global variable or constant: \n";
					outs().indent(4) << *operand << "\n";
					if (operand->getType()->isIntegerTy() || operand->getType()->isPointerTy())
						setOfIntegerInputs.insert(operand);
					else
						setOfNonIntegerInputs.insert(operand);
				}
				if (isa<GlobalVariable>(*operand))
					outputs.insert(operand);
			}
#endif
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

	// Add the types of the input values to the function's argument list.
	Type* i1Ty = Type::getInt1Ty(context);
	Type* i8Ty = Type::getInt8Ty(context);
	for (Values::const_iterator I = inputs.begin(), E = inputs.end(); I != E; ++I)
	{
		if ((*I)->getType() == i1Ty)
		{
			// Special case: store i1 type as i8.
			paramTy.push_back(i8Ty);
			continue;
		}

		paramTy.push_back((*I)->getType());
	}

	// Add the types of the output values to the function's argument list.
	for (Values::const_iterator I = outputs.begin(), E = outputs.end(); I != E; ++I)
	{
		if ((*I)->getType() == i1Ty)
		{
			// Special case: store i1 type as i8.
			paramTy.push_back(i8Ty);
			continue;
		}
	
		paramTy.push_back((*I)->getType());
	}

	// Aggregate args types into struct type
	PointerType *StructPtrType;
	Value* structArg = NULL;
	if (inputs.size() + outputs.size() > 0) {
		StructPtrType = PointerType::getUnqual(
			StructType::get(context, paramTy, false /* isPacked */));
		structArg = CastInst::CreatePointerCast(AI, StructPtrType, "",
			FuncRoot->getTerminator());
	}

	// Rewrite all users of the inputs in the cloned region to use the
	// arguments (or appropriate addressing into struct) instead.
	TerminatorInst *TI = FuncRoot->getTerminator();
	for (unsigned i = 0, e = inputs.size(); i != e; i++) {
		Value *Idx[2];
		Idx[0] = Constant::getNullValue(Type::getInt32Ty(context));
		Idx[1] = ConstantInt::get(Type::getInt32Ty(context), i + 2);

		// Create instruction to take address of "inputs[i]" in struct,
		// insert it before terminator.
		GetElementPtrInst *GEP = GetElementPtrInst::Create(
			structArg, Idx, "load_ptr_" + inputs[i]->getName(), TI);

		// create LoadInstruction from adress, which returned by instruction GEP
		// inserted it before terminator
		Value* RewriteVal;
		if (inputs[i]->getType() == i1Ty)
		{
			// Special case: cast i8 to i1, if input type is i1.
			LoadInst* LI = new LoadInst(GEP, "load_" + inputs[i]->getName(), TI);
			RewriteVal = CastInst::CreateIntegerCast(LI, i1Ty, false, "", TI);
		}
		else
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
	unsigned FirstOut = inputs.size() + 2;
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
						OldTarget->getName() + ".exitStub", LoopFunction);
					unsigned SuccNum = switchVal++;
					ExitBlocks.push_back(OldTarget);

					Value * brVal = ConstantInt::get(Type::getInt32Ty(context), SuccNum);

					// TODO: do not allow loops jumping somewhere outside
					//ReturnInst *NTRet = ReturnInst::Create(context, brVal, NewTarget);
					ReturnInst *NTRet = ReturnInst::Create(context, 0, NewTarget);

					// Restore values just before we exit
					for (unsigned out = 0, e = outputs.size(); out != e; ++out) {
#ifdef KERNELGEN_PRIVATIZE
						GlobalValue* GV = dyn_cast<GlobalValue>(outputs[out]);
						if (GV) continue;
#endif
						Value *Idx[2];
						Idx[0] = Constant::getNullValue(Type::getInt32Ty(context));
						Idx[1] = ConstantInt::get(Type::getInt32Ty(context),
						                          FirstOut + out);

						GetElementPtrInst *GEP = GetElementPtrInst::Create(structArg,
							Idx, "store_ptr_" + outputs[out]->getName(), NTRet);

						if (outputs[out]->getType() == i1Ty)
						{
							// Special case: cast i1 to i8, if output type is i1.
							CastInst* CI = CastInst::CreateIntegerCast(
								allToAllMap[outputs[out]], i8Ty, false, "", NTRet);
							new StoreInst(CI, GEP, NTRet);
							continue;
						}

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
    BasicBlock* loadAndSwitchExitBlock, AllocaInst* &Struct,
    ValueToValueMapTy & allToAllMap)
{
	std::vector<Value*> params, StructValues, ReloadOutputs, Reloads;

	LLVMContext &context = LaunchFunc->getContext();

	// Add inputs as params, or to be filled into the struct.
	// Also calculate the number of integer fields.
	unsigned int numints = 0;
	for (Values::iterator i = inputs.begin(), e = inputs.end(); i != e; ++i) {
		StructValues.push_back(*i);

		// Calculate the total count of integer and pointer inputs.
		Type* type = (*i)->getType();
		if (type->isIntegerTy() || type->isPointerTy()) numints++;
	}

	// Create a list of outputs
	for (Values::iterator i = outputs.begin(), e = outputs.end(); i != e; ++i)
		StructValues.push_back(*i);

	// Fill the arguments types structure.
	// First, place pointer to the function type.
	// Second, place pointer to the structure itself.
	Type* i1Ty = Type::getInt1Ty(context);
	Type* i8Ty = Type::getInt8Ty(context);
	std::vector<Type*> ArgTypes;
	ArgTypes.push_back(Type::getInt8PtrTy(context));
	ArgTypes.push_back(Type::getInt8PtrTy(context));
	for (Values::iterator v = StructValues.begin(),
		ve = StructValues.end(); v != ve; ++v)
	{
		if ((*v)->getType() == i1Ty)
		{
			// Special case: store i1 type as i8.
			ArgTypes.push_back(i8Ty);
			continue;
		}

		ArgTypes.push_back((*v)->getType());
	}

	// Allocate memory for the struct at the beginning of
	// function, which contains the Loop.
	StructType* StructArgTy = StructType::get(
		context, ArgTypes, false /* isPacked */);
								  
	IRBuilder<> Builder(callAndBranchBlock->getParent()->begin()->begin());
	Struct = Builder.CreateAlloca(StructArgTy, 0, "");

	// Initially, fill struct with zeros.
	CallInst* MI = Builder.CreateMemSet(Struct,
		Constant::getNullValue(Type::getInt8Ty(context)),
		ConstantExpr::getSizeOf(StructArgTy), 1);

	Value* Idx[2];
	Idx[0] = Constant::getNullValue(Type::getInt32Ty(context));

	// Store input values to arguments struct.
	for (unsigned i = 0, e = inputs.size(); i != e; ++i) {
		Idx[1] = ConstantInt::get(Type::getInt32Ty(context), i + 2);
		GetElementPtrInst *GEP = GetElementPtrInst::Create(
			Struct, Idx, "" + StructValues[i]->getName(), callAndBranchBlock);
		
		if (StructValues[i]->getType() == i1Ty)
		{
			// Special case: cast i1 to i8, if input type is i1.
			CastInst* CI = CastInst::CreateIntegerCast(
				StructValues[i], i8Ty, false, "", callAndBranchBlock);
			new StoreInst(CI, GEP, false, callAndBranchBlock);
			continue;
		}

		new StoreInst(StructValues[i], GEP, false, callAndBranchBlock);
	}


	// Create a metadata node holding a constant array
	// with original called function name.
	Value* name[] = { ConstantDataArray::getString(
		context, KernelFunc->getName(), true)
	};
	MDNode* nameMD = MDNode::get(context, name);

	// Add pointer to the original function string name
	// (to be set later on).
	params.push_back(
		Constant::getNullValue(Type::getInt8PtrTy(context)));

	// Store the size of the aggregated arguments struct
	// to the new call arguments list.
	params.push_back(ConstantExpr::getSizeOf(StructArgTy));

	// Store the total size of all integer fields in
	// aggregated arguments struct.
	Constant* size =
	    Constant::getNullValue(Type::getInt64Ty(context));
	if (numints) {
		size = ConstantExpr::getSub(
			// The offset of the last integer argument.
			ConstantExpr::getAdd(ConstantExpr::getOffsetOf(StructArgTy, (numints - 1) + 2),
				// The size of the last integer argument.
				ConstantExpr::getSizeOf(ArgTypes[(numints - 1) + 2])),
				// The offset of the first argument.
				ConstantExpr::getOffsetOf(StructArgTy, 2));
	}
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

	// Attach metadata node with the called function name.
	call->setMetadata("kernelgen_launch", nameMD);

	Value* Cond = new ICmpInst(*callAndBranchBlock, ICmpInst::ICMP_EQ,
		call, ConstantInt::get(Type::getInt32Ty(context), -1));
	BranchInst::Create(header, loadAndSwitchExitBlock, Cond, callAndBranchBlock);
#ifdef KERNELGEN_PRIVATIZE
	// Restore output values just after the exit
	unsigned FirstOut = inputs.size() + 2;
	for (unsigned out = 0, e = outputs.size(); out != e; ++out) {
                GlobalValue* GV = dyn_cast<GlobalValue>(outputs[out]);
                if (!GV) continue;

		Value *Idx[2];
		Idx[0] = Constant::getNullValue(Type::getInt32Ty(context));
		Idx[1] = ConstantInt::get(Type::getInt32Ty(context), FirstOut + out);

		GetElementPtrInst *GEP = GetElementPtrInst::Create(
			Struct, Idx, "store_ptr_" + outputs[out]->getName(),
			loadAndSwitchExitBlock);

		if (outputs[out]->getType() == i1Ty)
		{
			// Special case: cast i1 to i8, if input type is i1.
			CastInst* CI = CastInst::CreateIntegerCast(GEP, i8Ty, false, "", loadAndSwitchExitBlock);
			new StoreInst(allToAllMap[outputs[out]], CI, loadAndSwitchExitBlock);
			continue;
		}

		new StoreInst(allToAllMap[outputs[out]], GEP, loadAndSwitchExitBlock);
	}
#endif
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
	Type* i1Ty = Type::getInt1Ty(Context);
	Type* i8Ty = Type::getInt8Ty(Context);
	for (unsigned i = 0, e = outputs.size(); i != e; ++i) {
		Value *Idx[2];
		Idx[0] = Constant::getNullValue(Type::getInt32Ty(Context));
		Idx[1] = ConstantInt::get(Type::getInt32Ty(Context), FirstOut + i);

		GetElementPtrInst *GEP = GetElementPtrInst::Create(
		                             Struct, Idx,"gep_reload_" + outputs[i]->getName());
		loadAndSwitchExitBlock->getInstList().push_back(GEP);

		// create LoadInst to load value of "outputs[i]" from specified address		
		Instruction *load;
		if (outputs[i]->getType() == i1Ty)
		{
			// Special case: cast i1 to i8, if input type is i1.
			LoadInst* LI = new LoadInst(GEP, outputs[i]->getName()+".reload");
			loadAndSwitchExitBlock->getInstList().push_back(LI);
			load = CastInst::CreateIntegerCast(LI, i1Ty, false, "");
		}
		else
		{
			load = new LoadInst(GEP, outputs[i]->getName()+".reload");
		}
		
		loadAndSwitchExitBlock->getInstList().push_back(load);

		// map outputs[i] to created load instruction
		// late, we use that map to update relationshpis
		OutputsToLoadInstMap[outputs[i]] = load;
	}

	// Now we can emit a switch statement using the call as a value.
	SwitchInst *TheSwitch = SwitchInst::Create(callLoopFuncInst,
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
		/*case 2:
			BranchInst::Create(ExitBlocks[1], ExitBlocks[0],
			                   callLoopFuncInst, TheSwitch);
			TheSwitch->eraseFromParent();
			break;*/
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
	if(NumExitBlocks == 0) return;
	if(outputs.size() == 0) return;
	
	BasicBlock * ExitBlock = ExitBlocks[0];
	
	unsigned NumPreds = 0;
	for(pred_iterator predBlock = pred_begin(ExitBlock), predBlockE = pred_end(ExitBlock);
	    predBlock != predBlockE; predBlock++)
		NumPreds++;

	for (Values::iterator Out = outputs.begin(), Outputs_end = outputs.end(); Out != Outputs_end; Out++) {
#ifdef KERNELGEN_PRIVATIZE	
		GlobalValue* GV = dyn_cast<GlobalValue>(*Out);
		if (GV) continue;
#endif
		Value *I = *Out;
		I->setName("Output");
		
		SetVector<Value *> Users;
		Users = SetVector<Value *>(I->use_begin(),I->use_end());

		for (BasicBlock::iterator Inst = ExitBlock->begin(), E = ExitBlock->getFirstNonPHI();
			Inst != E; Inst++) {
			PHINode * phi_node = dyn_cast<PHINode>(Inst);
			if (Users.count(phi_node))
			{
				phi_node->addIncoming(OutputsToLoadInstMap[I],loadAndSwitchBlock);
				Users.remove(phi_node);
			}
		}

		PHINode* newPN = PHINode::Create(
			I->getType(), NumPreds, "newPHINode", ExitBlock -> begin());

		for(pred_iterator predBlock = pred_begin(ExitBlock), predBlockE = pred_end(ExitBlock);
		    predBlock != predBlockE; predBlock++)
			if( *predBlock != loadAndSwitchBlock) {
				if(OriginalLoopBlocks.count(*predBlock)) newPN -> addIncoming(I, *predBlock);
				else
					newPN -> addIncoming(UndefValue::get(I->getType()), *predBlock);
			} else newPN -> addIncoming(OutputsToLoadInstMap[I],loadAndSwitchBlock);

		
		for (SetVector<Value *>::iterator UI = Users.begin(), E = Users.end(); UI != E; ++UI) {
			Instruction * User = dyn_cast<Instruction>(*UI);
			if(User && !OriginalLoopBlocks.count( User->getParent())) {
				User->replaceUsesOfWith(I, newPN);
				//User->setName("UserOfOutput");
			}
		}
		Users.clear();
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
ExtractCodeRegion(Loop *L, LoopInfo &LI )
{
	const std::vector<BasicBlock*> &code = L->getBlocks();
	if (!isEligible(code)) return NULL;
	static int i = 0;


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

	LLVMContext& context = header->getContext();
	Function *parentFunction = header->getParent();
	Module* m = parentFunction->getParent();

	// If we have any return instructions in the region, split those blocks so
	// that the return is not in the region.
	splitReturnBlocks();

	// Find inputs to, outputs from the code region.
	findInputsOutputs(inputs, outputs);

	if (NumExitBlocks != 1) {
		outs().changeColor(raw_ostream::YELLOW);
		outs() << "KernelGen dropped loop: There must be only one exit block from the code region\n";
		outs().resetColor();
		return NULL;
	}

	ClonedCodeInfo CodeInfo;
	ValueToValueMapTy VMap;
	auto_ptr<SetVector<BasicBlock*> > clonedCode;
	clonedCode.reset(CloneCodeRegion(BlocksToExtract,
	                                 RF_IgnoreMissingEntries, VMap, ".cloned", &CodeInfo));

	ClonedLoopBlocks.insert(clonedCode.get()->begin(),clonedCode.get()->end());
	OriginalLoopBlocks.insert(BlocksToExtract.begin(), BlocksToExtract.end());



	// This takes place of the original loop
	BasicBlock *loadAndSwitchExitBlock = BasicBlock::Create(
	        header->getContext(), "loadOutputsAndSwitchExit",
	        parentFunction, header);
	BasicBlock *callAndBranchBlock = BasicBlock::Create(
	                                     header->getContext(), "callAndBranch",
	                                     parentFunction, loadAndSwitchExitBlock);

	Loop *parentLoop = NULL;
	if((parentLoop = (L -> getParentLoop()))) {
		parentLoop -> addBasicBlockToLoop(loadAndSwitchExitBlock,LI.getBase());
		parentLoop -> addBasicBlockToLoop(callAndBranchBlock, LI.getBase());
	}

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

        // FIXME: DM: the loop function should inherit all attributes of
        // the parent function?

        // If the old function is no-throw, so is the new one.
        if (parentFunction->doesNotThrow())
                loopFunction->setDoesNotThrow(true);

	// Never inline the extracted function.
	const AttrListPtr attr = loopFunction->getAttributes();
	const AttrListPtr attr_new = attr.addAttr(~0U, Attribute::NoInline);
	loopFunction->setAttributes(attr_new);

	// Reset to default visibility.
	loopFunction->setVisibility(GlobalValue::DefaultVisibility);

	// Attach a metadata node indicating the function is extracted.
	Value* name[] = { ConstantDataArray::getString(
		context, loopFunction->getName(), true)
	};
	MDNode* nameMD = MDNode::get(context, name);
	NamedMDNode *NMD = m->getOrInsertNamedMetadata("kernelgen.extracted");
	NMD->addOperand(nameMD);

	// Rename Blocks.
	for (SetVector<BasicBlock*>::iterator BB = OriginalLoopBlocks.begin(),
	     BB_end = OriginalLoopBlocks.end(); BB != BB_end; BB++)
		(*BB)->setName((*BB)->getName() + "_orig");

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
	                                    header, loadAndSwitchExitBlock, Struct, VMap);

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
	//DT->DT->recalculate(*loopFunction);

	if (verifyFunction(*loopFunction))
		cout << "verifyFunction failed!";

	return callLoopFuctionInst;
}

namespace llvm
{
// ExtractBasicBlock - slurp a natural loop into a brand new function.
 CallInst* BranchedExtractLoop(DominatorTree& DT,LoopInfo &LI, Loop *L)
{
	return BranchedCodeExtractor(&DT).ExtractCodeRegion(L,LI);
}

}

