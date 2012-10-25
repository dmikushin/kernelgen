#include "llvm/Target/TargetData.h"
#include "llvm/Instructions.h"
#include "llvm/Type.h"
#include "llvm/Pass.h"
#include "llvm/Function.h"
#include "llvm/Module.h"
#include "llvm/Constants.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_os_ostream.h"
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
Z("fix-pointers", "Remove extra bit casts of malloc's calls");
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
	//for(vector<GetElementPtrInst *>::iterator GEPs_iterator = GEPs.begin(), GEPs_iterator_end = GEPs.end();
	//    GEPs_iterator != GEPs_iterator_end; GEPs_iterator++) {
	//GetElementPtrInst* GEPInst = *GEPs_iterator;

	for(int i = 0 ; i < GEPs.size(); i++) {
		GetElementPtrInst* GEPInst = GEPs[i];
		if(GEPInst->getNumIndices() == 1 &&
		   isa<Instruction>(GEPInst->getOperand(0)) &&
		   targetData.getTypeAllocSize(GEPInst -> getType() ->getElementType())==1 ) {

			Value * GEPIndex = *GEPInst->idx_begin();

			bool isConstant = false;
			bool isInstruction = false;

			int64_t constantIndex = 0;
			BinaryOperator * instructionIndex = NULL;
			int64_t indexCoefficient = 0;

			// <handle non constant indexes >
			if( isa<BinaryOperator>(*GEPIndex) ) {
				instructionIndex = cast<BinaryOperator>(GEPIndex);
				switch(instructionIndex->getOpcode()) {
				case Instruction::Mul: { // index is realIndex*indexCoefficient
					if( isa<ConstantInt>(*instructionIndex -> getOperand(0)))
						instructionIndex->swapOperands();//move Constant to second place
					//first operand must be regular Value, second - Constant
					//Otherwise, how to define which argument is realIndex?
					if(!isa<Constant>(*instructionIndex -> getOperand(0)) &&
					   (isa<ConstantInt>(*instructionIndex -> getOperand(1)))) {
						//FIXME: IndexCoefficiesnt is signed value
						ConstantInt *secondOperand = cast<ConstantInt>(instructionIndex -> getOperand(1));
						indexCoefficient = secondOperand -> getSExtValue();
						isInstruction = true;
					}
				}
				break;
				default:
					//TODO: review other possible suitable cases
					break;
				}
			}// <handle non constant indexes>
			else if(isa<ConstantInt>(*GEPIndex)) {
				constantIndex = dyn_cast<ConstantInt>(GEPIndex)-> getSExtValue();
				isConstant = true;
			}// </ handle constant indexes>
			if(!isInstruction && !isConstant) continue;

			for(Value::use_iterator userOfGep = GEPInst -> use_begin(), userOfGep_end = GEPInst -> use_end();
			    userOfGep != userOfGep_end; userOfGep++)
				if(isa<BitCastInst>(**userOfGep)) {
					BitCastInst * castInst = cast<BitCastInst>(*userOfGep);
					if(castInst -> getDestTy() -> isPointerTy()) {
						PointerType * newPointerType = cast<PointerType>(castInst -> getDestTy());
						int typeAllocSize = targetData.getTypeAllocSize(newPointerType -> getElementType());

						/*{
							outs() << "<------------------------------- One more GEP Inst ------------------------------->" << "\n";
							PointerType * asdf = cast<PointerType>(cast<Instruction>(GEPInst->getOperand(0))->getOperand(0)->getType());
							outs() << " Real type : ";
						    outs() << "<-----| "<<*(asdf->getElementType()) << " |---->\n";

							outs().indent(6) << *(GEPInst->getOperand(0)) << "\n";
							if(isInstruction) {
								assert(instIndex);
								outs().indent(6) << *instIndex << " with coefficient " << indexCoefficient << "\n";
							}
							if(isConstant) {
								outs().indent(6) << " contsantIndex: " << constantIndex << "\n";
							}
							outs().indent(6) << *GEPInst << "\n";
							outs().indent(6) << *castInst << "\n";
							outs() << "<--------------------------------------------------------------------------------->" << "\n\n";
						}*/

						if(typeAllocSize == 0) {

							Type * elementType = newPointerType -> getElementType();
							if(elementType->isArrayTy())
							{
								while(elementType->isArrayTy())
									elementType = cast<ArrayType>(elementType)->getElementType();
								typeAllocSize=targetData.getTypeAllocSize(elementType);
								newPointerType = PointerType::getUnqual(elementType);
							}
							else
							{
							    outs().changeColor(raw_ostream::BLUE);
							    outs() << "KernelGen can not remove bit cast: typeAlloc size is zero\n";
							    outs().resetColor();
							    continue;
							}
						}
						//create newBitCast : (newPointerType)ptr, if there is no such
						Instruction* newBitCast = new BitCastInst(GEPInst->getOperand(0),newPointerType,"newBitCast");
						//newBitCast -> insertAfter(cast<Instruction>(GEPInst->getOperand(0)));
						newBitCast -> insertBefore(GEPInst);
						Instruction* newGEPInst = NULL;
						//create index for new GEP
						if(isConstant && (constantIndex % typeAllocSize == 0)) {
							//create newGEPInst : &newBitCast[constantIndex/typeAllocSize]
							Value * newIndex = ConstantInt::getSigned(GEPIndex->getType(),constantIndex / typeAllocSize);
							vector<Value *> Idx;
							Idx.push_back(newIndex);
							newGEPInst = GetElementPtrInst::Create(newBitCast, Idx, "newGEPInst");
						} else if(isInstruction && (indexCoefficient % typeAllocSize == 0) ) {

							Instruction* newIndex = instructionIndex->clone();
							newIndex ->setOperand(1,ConstantInt::getSigned(instructionIndex->getType(),indexCoefficient / typeAllocSize));
							newIndex -> setName("newIndex");
							newIndex -> insertAfter(instructionIndex);
							vector<Value *> Idx;
							Idx.push_back(newIndex);
							newGEPInst = GetElementPtrInst::Create(newBitCast, Idx, "newGEPInst");
						}
						if(newGEPInst) {
							newGEPInst -> insertAfter(GEPInst);
							if(newPointerType != castInst -> getDestTy())
							{
								// bit cast to array
								Instruction* newBitCastToArray = new BitCastInst(newGEPInst,castInst -> getDestTy(),"newBitCastToArray");
							    newBitCastToArray->insertAfter(newGEPInst);
								castInst -> replaceAllUsesWith(newBitCastToArray);
							}
							else
							    castInst -> replaceAllUsesWith(newGEPInst);
						} else {

							outs().changeColor(raw_ostream::BLUE);
							outs() << "KernelGen can not remove bit cast: "
							       "typeAllocSize is not divider of offset\n";
							outs().resetColor();
							continue;
						}
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
		Instruction * operand = cast<Instruction>(castInsts[i]->getOperand(0));
		if(operand -> getParent() != castInsts[i] -> getParent()) {
			castInsts[i]->removeFromParent();
			castInsts[i]->insertBefore(operand -> getParent() -> getTerminator());
		}
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
