#include "llvm/Target/TargetData.h"
#include "llvm/Instructions.h"
#include "llvm/Type.h"
#include "llvm/Pass.h"
#include "llvm/Function.h"
#include "llvm/Module.h"
#include "llvm/Constants.h"
#include <set>
#include <map>
#include <iostream>
using namespace llvm;
using namespace std;
class FixPointers : public ModulePass
{
	public:
		static char ID;
		FixPointers()
			:ModulePass(ID) {}
		void FixPointersInModule(Module *m);
		virtual bool runOnModule(Module &M) {
			FixPointersInModule(&M);
			return true;
		}
};
char FixPointers::ID = 0;
static RegisterPass<FixPointers>
Z("fix-using-of-malloc", "Remove extra bit casts of malloc's calls");
Pass* createFixPointersPass()
{
	return new FixPointers();
}
void FixPointers::FixPointersInModule(Module *m)
{
	vector<GetElementPtrInst *> GEPs;
	TargetData targetData(m);
	for(Module::iterator function = m->begin(), function_end = m->end();
	    function != function_end; function++)
		for(Function::iterator block = function->begin(),block_end = function->end();
		    block!=block_end; block++)
			for(BasicBlock::iterator instruction = block->begin(), instruction_end = block->end();
			    instruction!=instruction_end; instruction++)
				if(isa<GetElementPtrInst>(*instruction))
					GEPs.push_back(cast<GetElementPtrInst>(instruction));
	for(vector<GetElementPtrInst *>::iterator GEPs_iterator = GEPs.begin(), GEPs_iterator_end = GEPs.end();
	    GEPs_iterator != GEPs_iterator_end; GEPs_iterator++) {
		GetElementPtrInst* GEPInst = *GEPs_iterator;
		if(GEPInst->getNumIndices() == 1 &&
		   isa<Instruction>(GEPInst->getOperand(0)) &&
		   targetData.getTypeStoreSize(GEPInst -> getType() ->getElementType())==1 ) {
			Value * GEPIndex = *GEPInst->idx_begin(); // Index argument of GEP
			int64_t constantIndex = 0;
			bool isConstant = false;
			int64_t indexCoefficient = 0;
			BinaryOperator * instIndex = NULL;
			// <handle non constant indexes >
			if( isa<BinaryOperator>(*GEPIndex) ) {
				instIndex = dyn_cast<BinaryOperator>(GEPIndex);
				switch(instIndex->getOpcode()) {
				case Instruction::Mul: { // index is realIndex*indexCoefficient
					if( isa<ConstantInt>(*instIndex -> getOperand(0)))
						instIndex->swapOperands();//move Constant to second place
					ConstantInt *secondOperand;
					//first operand must be regular Value, second - Constant
					//Otherwise, how to define which argument is realIndex?
					if(!isa<ConstantInt>(*instIndex -> getOperand(0)) &&
					   (secondOperand = dyn_cast<ConstantInt>(instIndex -> getOperand(1))))
						//FIXME: IndexCoefficiesnt is signed value
						indexCoefficient = secondOperand -> getSExtValue();
					else instIndex = NULL;
				}
				break;
				default:
					//TODO: review other possible suitable cases
					instIndex = NULL;
					break;
				}
			}// <handle non constant indexes>
			else if(isa<ConstantInt>(*GEPIndex)) {
				constantIndex = dyn_cast<ConstantInt>(GEPIndex)-> getSExtValue();
				isConstant = true;
			}// </ handle constant indexes>
			if(!instIndex && !isConstant) continue;
			for(Value::use_iterator userOfGep = GEPInst -> use_begin(), userOfGep_end = GEPInst -> use_end();
			    userOfGep != userOfGep_end; userOfGep++)
				if(isa<BitCastInst>(**userOfGep)) {
					BitCastInst * castInst = cast<BitCastInst>(*userOfGep);
					if(castInst -> getDestTy() -> isPointerTy()) {
						PointerType * newPointerType = cast<PointerType>(castInst -> getDestTy());
						int typeStoreSize = targetData.getTypeStoreSize(newPointerType -> getElementType());
						//create newBitCast : (newPointerType)ptr, if there is no such
						Instruction* newBitCast = new BitCastInst(GEPInst->getOperand(0),newPointerType,"newBitCast");
						newBitCast -> insertAfter(cast<Instruction>(GEPInst->getOperand(0)));
						Instruction* newGEPInst;
						//create index for new GEP
						if(isConstant && (constantIndex % typeStoreSize == 0)) {
							//create newGEPInst : &newBitCast[constantIndex/typeStoreSize]
							Value * newIndex = ConstantInt::getSigned(GEPIndex->getType(),constantIndex / typeStoreSize);
							vector<Value *> Idx;
							Idx.push_back(newIndex);
							newGEPInst = GetElementPtrInst::Create(newBitCast, Idx, "newGEPInst");
						} else if(indexCoefficient % typeStoreSize == 0 ) {
							Instruction* newIndex = instIndex->clone();
							newIndex ->setOperand(1,ConstantInt::getSigned(instIndex->getType(),indexCoefficient / typeStoreSize));
							newIndex -> setName("newIndex");
							newIndex -> insertAfter(instIndex);
							vector<Value *> Idx;
							Idx.push_back(newIndex);
							newGEPInst = GetElementPtrInst::Create(newBitCast, Idx, "newGEPInst");
						}
						assert(newGEPInst);
						newGEPInst -> insertAfter(GEPInst);
						//replace all uses of oldBitCast by newGEPInst and erase oldBitCast
						castInst -> replaceAllUsesWith(newGEPInst);
						//castInst -> eraseFromParent();
					}
				}
		}
	}
}
class MoveUpCasts : public ModulePass
{
	public:
		static char ID;
		MoveUpCasts()
			:ModulePass(ID) {}
		void MoveUpCastsInModule(Module *m);
		virtual bool runOnModule(Module &M) {
			MoveUpCastsInModule(&M);
			return true;
		}
};
char MoveUpCasts::ID = 0;
static RegisterPass<MoveUpCasts>
A("move-up-casts", "Move each bitcast to it's argument");
Pass* createMoveUpCastsPass()
{
	return new MoveUpCasts();
}
void MoveUpCasts::MoveUpCastsInModule(Module *m)
{
	vector<BitCastInst *> castInsts;
	for(Module::iterator function = m->begin(), function_end = m->end();
	    function != function_end; function++)
		for(Function::iterator block = function->begin(),block_end = function->end();
		    block!=block_end; block++)
			for(BasicBlock::iterator instruction = block->begin(), instruction_end = block->end();
			    instruction!=instruction_end; instruction++)
				if(isa<BitCastInst>(*instruction)) {
					BitCastInst * castInst = cast<BitCastInst>(instruction);
					if(isa<Instruction>(castInst->getOperand(0))) {
						castInsts.push_back(castInst);
					}
				}
	for(int i = 0; i < castInsts.size(); i++) {
		castInsts[i]->removeFromParent();
		castInsts[i]->insertAfter(cast<Instruction>(castInsts[i]->getOperand(0)));
	}
	/*map<BitCastInst *,BitCastInst *> equalBitCasts;
	for(int i = 0; i < castInsts.size(); i++)
		for(int j = 0; j < castInsts.size(); j++) {
			if(castInsts[i]->getOperand(0) == castInsts[j]->getOperand(0) &&
			   castInsts[i]->getType() == castInsts[j]->getType()
			  ) equalBitCasts[castInsts[i]] = castInsts[j];
		}
	for(map<BitCastInst *,BitCastInst *>::iterator equalPair = equalBitCasts.begin(),equalPair_end = equalBitCasts.end();
	equalPair != equalPair_end; equalPair++)
	{
	 equalPair->second -> replaceAllUsesWith(equalPair->first);
	  equalPair->second -> eraseFromParent();
	}*/
}

