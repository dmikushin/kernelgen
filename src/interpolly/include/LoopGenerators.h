#include "llvm/Module.h"
#include "llvm/Function.h"
#include "llvm/ADT/Twine.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/IRBuilder.h"
#include "llvm/Analysis/Verifier.h"


namespace llvm
{
class Value;
class Pass;
class BasicBlock;
}

namespace kernelgen
{
llvm::Value *createLoopForCUDA(llvm::IRBuilder<> *Builder, llvm::Value *LB, llvm::Value *UB,
                               llvm::Value *ThreadLB, llvm::Value *ThreadUB, llvm::Value *ThreadStride,
                               const char * dimension, llvm::Pass *P, llvm::BasicBlock **AfterBlock);

llvm::Value *createLoop(llvm::Value *LB, llvm::Value *UB, llvm::Value *Stride,
                        llvm::IRBuilder<> &Builder, llvm::Pass *P,
                        llvm::BasicBlock *&AfterBlock);
}
