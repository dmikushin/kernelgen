#include "LoopGenerators.h"

#include "llvm/Module.h"
#include "llvm/Function.h"
#include "llvm/ADT/Twine.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/ValueMap.h"
#include "llvm/Support/IRBuilder.h"
#include "llvm/Analysis/Verifier.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include "llvm/Analysis/Dominators.h"
#include <string>

using namespace llvm;
using namespace std;

namespace kernelgen
{
Value *createLoopForCUDA(IRBuilder<> *Builder, Value *LB, Value *UB,
  Value *ThreadLB, Value *ThreadUB, Value *ThreadStride,
  const char * dimension, Pass *P, BasicBlock **AfterBlock) {
  Function *F = Builder->GetInsertBlock()->getParent();
  LLVMContext &Context = F->getContext();

  BasicBlock *PreheaderBB = Builder->GetInsertBlock();
  BasicBlock *HeaderBB = BasicBlock::Create(Context, (string)"CUDA.LoopHeader." + dimension, F);
  BasicBlock *BodyBB = BasicBlock::Create(Context, (string)"CUDA.LoopBody." + dimension, F);

  BasicBlock *AfterBB = SplitBlock(PreheaderBB, Builder->GetInsertPoint()++, P);
  AfterBB->setName((string)"CUDA.AfterLoop." + dimension);

  PreheaderBB->getTerminator()->setSuccessor(0, HeaderBB);
  
  //DominatorTree &DT = P->getAnalysis<DominatorTree>();
  //DT.addNewBlock(HeaderBB, PreheaderBB);
  
  Builder->SetInsertPoint(HeaderBB);

  // Use the type of upper and lower bound.
  assert(LB->getType() == UB->getType()
    && "Different types for upper and lower bound.");

  IntegerType *LoopIVType = dyn_cast<IntegerType>(UB->getType());
  assert(LoopIVType && "UB is not integer?");

  // IV
  PHINode *IV = Builder->CreatePHI(LoopIVType, 2, (string)"CUDA.loopiv." + dimension);
  IV->addIncoming(ThreadLB, PreheaderBB);

  // IV increment.
  Value *StrideValue = ThreadStride;
  Value *IncrementedIV = Builder->CreateAdd(IV, StrideValue, (string)"CUDA.next_loopiv." + dimension);

  // Exit condition.
  // Maybe not executed at all.
  // next iteration performed if loop condition is true:
  // InductionVariable <= ThreadUpperBound && InductionVariable <= LoopUpperBound
  Value *ExitCond = Builder->CreateICmpSLE(IV, ThreadUB, (string)"isInThreadBounds." + dimension); 

  Builder->CreateCondBr(ExitCond, BodyBB, AfterBB);
 // DT.addNewBlock(BodyBB, HeaderBB);

  Builder->SetInsertPoint(BodyBB);
  Builder->CreateBr(HeaderBB);
  IV->addIncoming(IncrementedIV, BodyBB);
 // DT.changeImmediateDominator(AfterBB, HeaderBB);

  Builder->SetInsertPoint(BodyBB->begin());
  *AfterBlock = AfterBB;

  return IV;
}


Value *createLoop(Value *LB, Value *UB, Value *Stride,
                         IRBuilder<> &Builder, Pass *P,
                         BasicBlock *&AfterBlock) {
  DominatorTree &DT = P->getAnalysis<DominatorTree>();
  Function *F = Builder.GetInsertBlock()->getParent();
  LLVMContext &Context = F->getContext();

  BasicBlock *PreheaderBB = Builder.GetInsertBlock();
  BasicBlock *HeaderBB = BasicBlock::Create(Context, "polly.loop_header", F);
  BasicBlock *BodyBB = BasicBlock::Create(Context, "polly.loop_body", F);
  BasicBlock *AfterBB = SplitBlock(PreheaderBB, Builder.GetInsertPoint()++, P);
  AfterBB->setName("polly.loop_after");

  PreheaderBB->getTerminator()->setSuccessor(0, HeaderBB);
  DT.addNewBlock(HeaderBB, PreheaderBB);

  Builder.SetInsertPoint(HeaderBB);

  // Use the type of upper and lower bound.
  assert(LB->getType() == UB->getType()
         && "Different types for upper and lower bound.");

  IntegerType *LoopIVType = dyn_cast<IntegerType>(UB->getType());
  assert(LoopIVType && "UB is not integer?");

  // IV
  PHINode *IV = Builder.CreatePHI(LoopIVType, 2, "polly.loopiv");
  IV->addIncoming(LB, PreheaderBB);

  Stride = Builder.CreateZExtOrBitCast(Stride, LoopIVType);
  Value *IncrementedIV = Builder.CreateAdd(IV, Stride, "polly.next_loopiv");

  // Exit condition.
  Value *CMP;
  CMP = Builder.CreateICmpSLE(IV, UB);

  Builder.CreateCondBr(CMP, BodyBB, AfterBB);
  DT.addNewBlock(BodyBB, HeaderBB);

  Builder.SetInsertPoint(BodyBB);
  Builder.CreateBr(HeaderBB);
  IV->addIncoming(IncrementedIV, BodyBB);
  DT.changeImmediateDominator(AfterBB, HeaderBB);

  Builder.SetInsertPoint(BodyBB->begin());
  AfterBlock = AfterBB;

  return IV;
}

}