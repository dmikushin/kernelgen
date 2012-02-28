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

class FixUsingOfMallocPass : public ModulePass
{
	public:
		static char ID;
		FixUsingOfMallocPass()
			:ModulePass(ID) {}
		void FixUsesOfMalloc(Module *m);
		virtual bool runOnModule(Module &M) {
			FixUsesOfMalloc(&M);
			return true;
		}

};
char FixUsingOfMallocPass::ID = 0;
static RegisterPass<FixUsingOfMallocPass>
Z("fix-using-of-malloc", "Remove extra bit casts of malloc's calls");

Pass* createFixUsingOfMallocPass()
{
	return new FixUsingOfMallocPass();
}

void FixUsingOfMallocPass::FixUsesOfMalloc(Module *m)
{
	TargetData * targetData = new TargetData(m);
	Function * mallocFunction = m->getFunction("malloc");
	if(mallocFunction && mallocFunction -> getNumUses() > 0) {
		// <Handle malloc calls>
		for(Value::use_iterator user = mallocFunction -> use_begin(), user_end = mallocFunction -> use_end();
		    user != user_end; user++) {

			CallInst *mallocCall; // some malloc call
			assert(mallocCall = dyn_cast<CallInst>(*user));
			map<Type *, Instruction *> createdBitCasts; // created bit casts to necessary types

			// <find geps that use ptr, returned by some malloc call>
			for(Value::use_iterator userOfMallocCall = mallocCall -> use_begin(), userOfMallocCall_end = mallocCall -> use_end();
			    userOfMallocCall != userOfMallocCall_end; userOfMallocCall++) {

				GetElementPtrInst * GEPInst; //GetElementPtrInst, which use malloc ptr
				//We are interested in GEPs with one index
				if((GEPInst = dyn_cast<GetElementPtrInst>(*userOfMallocCall)) && GEPInst->getNumIndices() == 1) {

					Value * Index = *GEPInst->idx_begin(); // Index argument of GEP


					int64_t indexCoefficient = 0;
					int64_t constantIndex = 0;

					bool isConstant = false;
					Value * realIndex = NULL;


					BinaryOperator * instIndex;
					// <handle non constant indexes >
					if( (instIndex = dyn_cast<BinaryOperator>(Index)) ) {
						switch(instIndex->getOpcode()) {
						case Instruction::Mul: { // index is realIndex*indexCoefficient

							if( dyn_cast<ConstantInt>(instIndex -> getOperand(0)))
								instIndex->swapOperands();//move Constant to second place
							ConstantInt *secondOperand;

							//first operand must be regular Value, second - Constant
							//Otherwise, how to check shoose which argument is realIndex?
							if(!dyn_cast<ConstantInt>(instIndex -> getOperand(0)) &&
							   (secondOperand = dyn_cast<ConstantInt>(instIndex -> getOperand(1)))) {
								//FIXME: IndexCoefficiesnt is signed value
								indexCoefficient = secondOperand -> getSExtValue();
								realIndex = instIndex -> getOperand(0);
							}
							break;
							case Instruction::Shl: { // index is realIndex << power, i.e. index is realIndex*2^power
								ConstantInt *secondOperand;
								//second operand must be constant
								//Otherwise, we can not retrive it's value
								if((secondOperand = dyn_cast<ConstantInt>(instIndex -> getOperand(1)))) {
									realIndex = instIndex -> getOperand(0);
									uint64_t power = secondOperand -> getZExtValue();
									for(indexCoefficient = 1; power > 0; indexCoefficient*=2, power--);
								}
							}
							break;
							default:
								//TODO: review other possible suitable cases
								break;
							}
						}
					} // </ handle non constant indexes>
					// else
					// <handle constant indexes>
					if(dyn_cast<ConstantInt>(Index)) {
						constantIndex = dyn_cast<ConstantInt>(Index)-> getSExtValue();
						isConstant = true;
					}// </ handle constant indexes>

					// TODO: handle negative indexCoefficients
					assert( indexCoefficient >= 0);

					// The index of GEP is not suitable, go to next GEP !!!!!!!!!!!!
					if(!realIndex && !isConstant) continue;   //!!!!!!!!!!!!!!!!!!!!!!

					map<Type *, GetElementPtrInst *> createdGEPs;

					// <find BitCasts, which are users of GEP>
					for(Value::use_iterator userOfGep = GEPInst -> use_begin(), userOfGep_end = GEPInst -> use_end();
					    userOfGep != userOfGep_end; userOfGep++) {
						BitCastInst * castInst;

						// <handle BitCast>
						if( (castInst = dyn_cast<BitCastInst>(*userOfGep)) ) {
							//we are interested in BitCasts to pointer types
							if(castInst -> getDestTy() -> isPointerTy()) {
								PointerType * newPointerType = dyn_cast<PointerType>(castInst -> getDestTy());
								int typeStoreSize = targetData->getTypeStoreSize(newPointerType -> getElementType());

								if(isConstant && (constantIndex % typeStoreSize == 0)) {
									//create newBitCast : (newPointerType)mallocCall, if there is no such
									if(createdBitCasts.find(newPointerType) == createdBitCasts.end()) {
										createdBitCasts[newPointerType]= new BitCastInst(mallocCall,newPointerType,"newBitCast");
										createdBitCasts[newPointerType] -> insertAfter(mallocCall);
									}
									Value* newBitCast = createdBitCasts[newPointerType];

									//create newGEPInst : &newBitCast[constantIndex/typeStoreSize], if there is no such
									if(createdGEPs.find(newPointerType) == createdGEPs.end()) {
										realIndex = ConstantInt::getSigned(Index->getType(),constantIndex / typeStoreSize);
										vector<Value *> Idx;
										Idx.push_back(realIndex);
										createdGEPs[newPointerType] = GetElementPtrInst::Create(newBitCast, Idx, "newGEPInst",GEPInst);
									}
									Value* newGEPInst = createdGEPs[newPointerType];

									//replace all uses of oldBitCast by newGEPInst and erase oldBitCast
									castInst -> replaceAllUsesWith(newGEPInst);
									castInst -> eraseFromParent();

								} else if(indexCoefficient == targetData->getTypeStoreSize(newPointerType -> getElementType())) {
									//create newBitCast : (newPointerType)mallocCall, if there is no such
									if(createdBitCasts.find(newPointerType) == createdBitCasts.end()) {
										createdBitCasts[newPointerType]= new BitCastInst(mallocCall,newPointerType,"newBitCast");
										createdBitCasts[newPointerType] -> insertAfter(mallocCall);
									}
									Value* newBitCast = createdBitCasts[newPointerType];

									//create newGEPInst : &newBitCast[realIndex], if there is no such
									if(createdGEPs.find(newPointerType) == createdGEPs.end()) {
										vector<Value *> Idx;
										Idx.push_back(realIndex);
										createdGEPs[newPointerType] = GetElementPtrInst::Create(newBitCast, Idx, "newGEPInst",GEPInst);
									}
									Value* newGEPInst = createdGEPs[newPointerType];

									//replace all uses of oldBitCast by newGEPInst and erase oldBitCast
									castInst -> replaceAllUsesWith(newGEPInst);
									castInst -> eraseFromParent();
								}
								// <by llvm optimization passes>
								//If there no uses of old GEP - erase oldGep
								//if there no uses of oldIndex - erase oldIndex
								// </ by llvm optimization passes>
							}
						}// </ handle BitCast>
					}// </ find BitCasts, which are users of GEP>
				}
			} // </ find geps that use ptr, returned by some malloc call>
		} // </ Handle malloc calls>
	}
	cout << "asdf";
}
