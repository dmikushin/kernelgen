#include "llvm/Function.h"
#include "llvm/Transforms/Utils/ValueMapper.h"

using namespace llvm;


namespace kernelgen {

/// LinkFunctionBody - Copy the source function over into the dest function and
/// fix up references to values.  Dest is an external function, and Src is not.
/// Also insert the used functions declarations.
void LinkFunctionBody(llvm::Function *Dst, llvm::Function *Src);
void LinkFunctionBody(Function *Dst, Function *Src, ValueToValueMapTy & ValueMap );
}

