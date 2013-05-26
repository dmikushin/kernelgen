//===- Platform.h - KernelGen runtime API
//----------------------------------===//
//
//     KernelGen -- A prototype of LLVM-based auto-parallelizing Fortran/C
//        compiler for NVIDIA GPUs, targeting numerical modeling code.
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines KernelGen target platforms.
//
//===----------------------------------------------------------------------===//

#include "llvm/Support/TargetRegistry.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Target/Mangler.h"
#include "llvm/Target/TargetMachine.h"

#include "KernelGen.h"

#include <memory>

namespace kernelgen {
namespace runtime {

// Defines target platform for a runmode.
struct TargetPlatform {
  const llvm::Target *target;
  llvm::TargetMachine *machine;
  llvm::StringRef triple;
  std::auto_ptr<llvm::MCContext> mccontext;
  std::auto_ptr<llvm::Mangler> mangler;

  TargetPlatform(const llvm::Target *target, llvm::TargetMachine *machine,
                 llvm::StringRef triple);
};

// The collection of platforms for every supported runmode.
class TargetPlatforms {
  std::auto_ptr<TargetPlatform> platforms[KERNELGEN_RUNMODE_COUNT];

public:

  // Get target platform for the specified runmode.
  TargetPlatform *operator[](int runmode);
};

extern TargetPlatforms platforms;

}
}
