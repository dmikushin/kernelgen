#include "KernelPrepare.h"

char KernelPrepare::ID = 0;


INITIALIZE_PASS_BEGIN(KernelPrepare, "kernel-prepare",
                      "Kernelgen prepare kernels", false, false)
INITIALIZE_PASS_END(KernelPrepare, "kernel-prepare",
                    "Kernelgen prepare kernels", false, false)
