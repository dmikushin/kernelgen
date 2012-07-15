#include "llvm/Function.h"

namespace kernelgen {

/// LinkFunctionBody - Copy the source function over into the dest function and
/// fix up references to values.  Dest is an external function, and Src is not.
/// Also insert the used functions declarations.
void LinkFunctionBody(llvm::Function *Dst, llvm::Function *Src);

}

