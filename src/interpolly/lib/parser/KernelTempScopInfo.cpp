#include "KernelTempScopInfo.h"

namespace kernelgen
{
char KernelTempScopInfo::ID = 0;
}

using namespace kernelgen;
INITIALIZE_PASS_BEGIN(KernelTempScopInfo, "kernel-temp-scop",
                      "Kernelgen TempScopInfo analysis", false, true)

INITIALIZE_PASS_DEPENDENCY(KernelVerification)
INITIALIZE_PASS_DEPENDENCY(LoopInfo)
INITIALIZE_PASS_DEPENDENCY(DominatorTree)
INITIALIZE_PASS_DEPENDENCY(PostDominatorTree)
INITIALIZE_PASS_DEPENDENCY(TargetData)

INITIALIZE_PASS_END(KernelTempScopInfo, "kernel-temp-scop",
                    "Kernelgen TempScopInfo analysis", false, true)
