//===- CLooG.h - CLooG interface --------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// CLooG[1] interface.
//
// The CLooG interface takes a Scop and generates a CLooG AST (clast). This
// clast can either be returned directly or it can be pretty printed to stdout.
//
// A typical clast output looks like this:
//
// for (c2 = max(0, ceild(n + m, 2); c2 <= min(511, floord(5 * n, 3)); c2++) {
//   bb2(c2);
// }
//
// [1] http://www.cloog.org/ - The Chunky Loop Generator
//
//===----------------------------------------------------------------------===//

#ifndef POLLY_CLOOG_H
#define POLLY_CLOOG_H

#include "polly/Config/config.h"
#include "KernelScopPass.h"
#include "cloog/cloog.h"

struct clast_name;
namespace llvm {
  class raw_ostream;
}

namespace kernelgen {
  class Scop;
  class KernelCloog;

  class KernelCloogInfo : public KernelScopPass {
    KernelCloog *C;
    Scop *scop;

  public:
    static char ID;
    KernelCloogInfo() : KernelScopPass(ID), C(0) {}

    /// Write a .cloog input file
    void dump(FILE *F);


    /// Print a source code representation of the program.
    void pprint(llvm::raw_ostream &OS);
	
	void dump();
    /// Create the CLooG AST from this program.
    const struct clast_root *getClast();

    bool runOnScop(Scop &S);
    void printScop(llvm::raw_ostream &OS) const;
    virtual void getAnalysisUsage(AnalysisUsage &AU) const;
    virtual void releaseMemory();
  };
}

namespace llvm {
  class PassRegistry;
  void initializeKernelCloogInfoPass(llvm::PassRegistry&);
}

#endif /* POLLY_CLOOG_H */
