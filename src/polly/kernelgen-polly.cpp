//===- Compile.cpp - Kernels compiler API ---------------------------------===//
//
//     KernelGen -- A prototype of LLVM-based auto-parallelizing Fortran/C
//        compiler for NVIDIA GPUs, targeting numerical modeling code.
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements compiling of parallel loops on supported architectures.
//
//===----------------------------------------------------------------------===//

#include "polly/LinkAllPasses.h" // must include before "llvm/Transforms/Scalar.h"
#include "polly/RegisterPasses.h"

#include "llvm/Analysis/Verifier.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Instructions.h"
#include "llvm/Linker.h"
#include "llvm/IR/Module.h"
#include "llvm/IRReader/IRReader.h"
#include "llvm/PassManager.h"
#include "llvm/Analysis/Passes.h"
#include "llvm/Support/FormattedStream.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/Target/TargetOptions.h"
#include "llvm/Transforms/IPO/PassManagerBuilder.h"
#include "llvm/Support/raw_os_ostream.h"
#include "llvm/Support/ToolOutputFile.h"
#include "llvm/IR/TypeBuilder.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Transforms/Scalar.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include "llvm/Transforms/Utils/Cloning.h"
#include "llvm/Transforms/IPO.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/FormattedStream.h"
#include "llvm/Support/Host.h"
#include "llvm/Support/TargetRegistry.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Bitcode/ReaderWriter.h"
#include "llvm/Support/system_error.h"
#include "polly/ScopInfo.h"

#include "Runtime.h"

#include <math.h>
#include <cstdlib>
#include <dlfcn.h>
#include <fstream>
#include <stack>
#include <list>
#include <set>
#include <stdio.h>

#include "GlobalDependences.h"

using namespace kernelgen;
using namespace kernelgen::bind::cuda;
using namespace kernelgen::runtime;
using namespace llvm;
using namespace polly;
using namespace std;
namespace polly {
extern cl::opt<bool> CUDA;
};
namespace llvm {
void RemoveStatistics();
}
extern cl::opt<bool> IgnoreAliasing;
//extern cl::opt<bool> AllowNonAffine;

Pass *createSizeOfLoopsPass(vector<Size3> *memForSize3 = NULL,
                            bool *isThereAtLeastOneParallelLoop = NULL);
Pass *createTransformAccessesPass();
Pass *createInspectDependencesPass();
Pass *createScopDescriptionPass();
Pass *createSetRelationTypePass(MemoryAccess::RelationType relationType =
                                    MemoryAccess::RelationType_polly);

void printSpecifiedStatistics(vector<string> statisticsNames) {
  string allStatistics;
  raw_string_ostream stringOS(allStatistics);
  llvm::PrintStatistics(stringOS);
  outs().changeColor(raw_ostream::YELLOW);
  for (int i = 0; i < statisticsNames.size(); i++) {
    string statisticName = statisticsNames[i];
    int start = 0;
    int end = 0;
    while ((start = allStatistics.find(statisticName, end)) != -1) {
      start = allStatistics.rfind('\n', start);
      if (start == -1)
        start == 0;
      end = allStatistics.find('\n', start + 1);
      outs() << allStatistics.substr(start + 1, end - start);
    }
  }
  outs().resetColor();
}

static void registerPollyPreoptPasses(llvm::PassManagerBase &PM) {
  // A standard set of optimization passes partially taken/copied from the
  // set of default optimization passes. It is used to bring the code into
  // a canonical form that can than be analyzed by Polly. This set of passes is
  // most probably not yet optimal. TODO: Investigate optimal set of passes.
  PM.add(llvm::createPromoteMemoryToRegisterPass());
  PM.add(llvm::createInstructionCombiningPass()); // Clean up after IPCP & DAE
  PM.add(llvm::createCFGSimplificationPass());    // Clean up after IPCP & DAE
  PM.add(llvm::createTailCallEliminationPass());  // Eliminate tail calls
  PM.add(llvm::createCFGSimplificationPass());    // Merge & remove BBs
  PM.add(llvm::createReassociatePass());          // Reassociate expressions
  PM.add(llvm::createLoopRotatePass());           // Rotate Loop
  PM.add(llvm::createInstructionCombiningPass());
  PM.add(polly::createIndVarSimplifyPass()); // Canonicalize indvars

  PM.add(polly::createCodePreparationPass());
  //PM.add(polly::createRegionSimplifyPass());

  // FIXME: The next two passes should not be necessary here. They are currently
  //        because of two problems:
  //
  //        1. The RegionSimplifyPass destroys the canonical form of induction
  //           variables,as it produces PHI nodes with incorrectly ordered
  //           operands. To fix this we run IndVarSimplify.
  //
  //        2. IndVarSimplify does not preserve the region information and
  //           the regioninfo pass does currently not recover simple regions.
  //           As a result we need to run the RegionSimplify pass again to
  //           recover them
  PM.add(polly::createIndVarSimplifyPass());
  //PM.add(polly::createRegionSimplifyPass());
}

static bool preopt = false;

static void runPolly(Kernel *kernel, Size3 *sizeOfLoops, bool mode,
                     bool *isThereAtLeastOneParallelLoop) {
  if (preopt) {
    PassManager polly;
    polly.add(new DataLayout(kernel->module));
    registerPollyPreoptPasses(polly);
    //polly.add(polly::createIslScheduleOptimizerPass());
    polly.run(*kernel->module);
  }

  IgnoreAliasing.setValue(true);
  //AllowNonAffine.setValue(true);
  polly::CUDA.setValue(mode);

  if (settings.getVerboseMode() & Verbose::Polly)
    llvm::EnableStatistics();

  //bool debug = ::llvm::DebugFlag;
  //if (verbose)
  //	::llvm::DebugFlag = true;
  vector<Size3> sizes;
  {
    PassManager polly;
    polly.add(new DataLayout(kernel->module));
    //registerPollyPreoptPasses(polly);
    //polly.add(polly::createIslScheduleOptimizerPass());
    if (kernel->name != "__kernelgen_main") {
      polly.add(createTransformAccessesPass()); // create General Form for each
                                                // scop's memory Access
      polly.add(createSizeOfLoopsPass(
          &sizes, isThereAtLeastOneParallelLoop)); // compute size of loops
      polly.add(createScopDescriptionPass());      // print scop description
      // set their current relation types to RelationType_general
      //polly.add(createScopDescriptionPass());
      polly.add(createInspectDependencesPass()); // Dependences run and compute
                                                 // dependences
      // before InspectDependences, but after TransformAccesses
      // and use general form of memory accesses
      polly.add(createSetRelationTypePass()); // set current relation types in
                                              // scop's memory Accesses back to
      // RelationType_polly
      //polly.add(createScopDescriptionPass());
    }
    polly.add(polly::createCodeGenerationPass()); // -polly-codegenn
    // use polly's representation of Memory Accesses
    polly.add(createCFGSimplificationPass());
    polly.run(*kernel->module);
  }
  if (kernel->name != "__kernelgen_main") {
    Size3 SizeOfLoops;
    if (sizes.size() == 0) {
      VERBOSE(Verbose::Polly
              << Verbose::Red
              << "\n    FAIL: No Valid Scops detected in kernel!!!\n\n"
              << Verbose::Default << Verbose::Reset);
    } else {
      // non-negative define sizes
      // if parallelized less than 3 loops then remaining will be -1
      // example:
      //for (c2=0;c2<=122;c2++) {
      //   for (c4=0;c4<=c2+13578;c4++) {
      //      Stmt_polly_stmt_4_cloned(c2,c4);
      //	}
      // }
      // SizeOfLoops : 123 13640 -1
      SizeOfLoops = sizes[0]; // 3-dimensional
      if (sizeOfLoops)
        *sizeOfLoops = SizeOfLoops;
    }

  }
  if (settings.getVerboseMode() & Verbose::Polly) {
    vector<string> statisticsNames;
    statisticsNames.push_back("polly-detect");
    statisticsNames.push_back("runtime-AA");
    printSpecifiedStatistics(statisticsNames);
    llvm::RemoveStatistics();
  }
  //::llvm::DebugFlag = debug;
  VERBOSE(Verbose::Blue << "<------------------ " << kernel->name
                        << ": compile completed ------------------->\n\n"
                        << Verbose::Reset);
}

extern "C" int __regular_main(int argc, char *argv[]) { return 0; }

void usage(const char *filename) {
  cout << "Usage: " << filename << " [+preopt] <filename>" << endl;
}

int main(int argc, char *argv[]) {
  if ((argc != 2) && (argc != 3)) {
    usage(argv[0]);
    return 0;
  }

  const char *filename = argv[1];
  if (argc == 3) {
    if (strcmp(argv[1], "+preopt")) {
      usage(argv[0]);
      return 0;
    }

    preopt = true;
  }

  LLVMContext &context = getGlobalContext();
  SMDiagnostic diag;
  OwningPtr<MemoryBuffer> source;
  error_code err = MemoryBuffer::getFile(filename, source);
  if (err.value() != 0) {
    cerr << "Error loading source from file " << argv[0] << endl;
    return -1;
  }
  Module *m = ParseIR(source.get(), diag, context);
  if (!m) {
    cerr << "Error loading LLVM IR from file " << argv[0] << endl;
    return -2;
  }

  Kernel kernel;
  kernel.module = m;
  Size3 sizeOfLoops;
  bool isThereAtLeastOneParallelLoop;
  runPolly(&kernel, &sizeOfLoops, true, &isThereAtLeastOneParallelLoop);

  source.take();
  return 0;
}
