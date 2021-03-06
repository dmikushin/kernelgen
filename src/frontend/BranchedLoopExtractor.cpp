//===- BranchedLoopExtractor.cpp - Extract each loop into a new function --===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// A pass wrapper around the ExtractLoop() scalar transformation.
// This pass creates equivalent function for all natural loops from the
// program if it can, insert call to this function before loop and adds branch
// to switch between original loop execution and function call.
//
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "loop-extract-with-branch"

#include "llvm/IR/Constants.h"
#include "llvm/Transforms/IPO.h"
#include "llvm/IR/Instructions.h"
#include "llvm/PassSupport.h"
#include "llvm/IR/Module.h"
#include "llvm/Pass.h"
#include "llvm/Analysis/Dominators.h"
#include "llvm/Analysis/LoopPass.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Transforms/Scalar.h"
#include "llvm/ADT/Statistic.h"

#include "BranchedLoopExtractor.h"

#include <fstream>
#include <set>
#include <vector>

using namespace llvm;
using namespace std;

STATISTIC(NumBranchedExtracted, "Number of loops extracted");
static int tmpBranchedExtracted = 0;

namespace {
struct BranchedLoopExtractor : public LoopPass {
  static char ID; // Pass identification, replacement for typeid
  unsigned NumLoops;
  Function *currentFunction;
  std::vector<CallInst *> *LoopFunctionCalls;
  explicit BranchedLoopExtractor(unsigned numLoops = ~0)
      : LoopPass(ID), NumLoops(numLoops) {
    LoopFunctionCalls = NULL;
    //initializeBranchedLoopExtractorPass(*PassRegistry::getPassRegistry());
  }

  explicit BranchedLoopExtractor(std::vector<CallInst *> &LFC,
                                 unsigned numLoops = ~0)
      : LoopPass(ID), NumLoops(numLoops), LoopFunctionCalls(&LFC),
        currentFunction(NULL) {
    //initializeBranchedLoopExtractorPass(*PassRegistry::getPassRegistry());
  }

  virtual bool runOnLoop(Loop *L, LPPassManager &LPM);

  virtual void getAnalysisUsage(AnalysisUsage &AU) const {
    AU.addRequiredID(BreakCriticalEdgesID);
    AU.addRequiredID(LoopSimplifyID);
    AU.addRequired<DominatorTree>();
    AU.addRequired<LoopInfo>();
  }
  void recursiveExtractSubLoops(Loop *loop);
};
}
char BranchedLoopExtractor::ID = 0;
Pass *llvm::createBranchedLoopExtractorPass() {
  return new BranchedLoopExtractor();
}
Pass *llvm::createBranchedLoopExtractorPass(std::vector<CallInst *> &LFC) {
  return new BranchedLoopExtractor(LFC);
}

INITIALIZE_PASS_BEGIN(BranchedLoopExtractor, "loop-extract-with-branch",
                      "Extract loops into new functions and add branches",
                      false, false)
INITIALIZE_PASS_DEPENDENCY(BreakCriticalEdges)
INITIALIZE_PASS_DEPENDENCY(LoopSimplify)
INITIALIZE_PASS_DEPENDENCY(DominatorTree)
INITIALIZE_PASS_DEPENDENCY(LoopInfo)
INITIALIZE_PASS_END(BranchedLoopExtractor, "loop-extract-with-branch",
                    "Extract loops into new functions and add branches", false,
                    false)

/*
static void* initializeBranchedLoopExtractorPassOnce (PassRegistry &Registry) {
    initializeBreakCriticalEdgesPass(Registry);
    initializeLoopSimplifyPass(Registry);
    initializeDominatorTreePass(Registry);
    PassInfo *PI = new PassInfo("Extract loops into new functions and add
branches",
                            "loop-extract-with-branch",
							&BranchedLoopExtractor ::ID,
							PassInfo::NormalCtor_t(callDefaultCtor< BranchedLoopExtractor >), false,
false);
    Registry.registerPass(*PI, true); return PI;
}
void llvm::initializeBranchedLoopExtractorPass(PassRegistry &Registry) {
	static volatile sys::cas_flag initialized = 0;
	sys::cas_flag old_val = sys::CompareAndSwap(&initialized, 1, 0);
	if (old_val == 0) {
		initializeBranchedLoopExtractorPassOnce(Registry);
		sys::MemoryFence(); initialized = 2;
	} else {
		sys::cas_flag tmp = initialized; sys::MemoryFence();
		while (tmp != 2) { tmp = initialized; sys::MemoryFence(); }
	}
}*/

struct staticInit {
  staticInit() {
    llvm::initializeBranchedLoopExtractorPass(*PassRegistry::getPassRegistry());
  }
};

static staticInit in;

// createBranchedLoopExtractorPass - This pass extracts all natural loops from
// the
// program into a function if it can.
//
void BranchedLoopExtractor::recursiveExtractSubLoops(Loop *loop) {
  std::vector<Loop *> loops = loop->getSubLoops();
  DominatorTree &DT = getAnalysis<DominatorTree>();
  LoopInfo &LI = getAnalysis<LoopInfo>();

  if (loops.size() > 0) {
    for (std::vector<Loop *>::iterator loop_iter = loops.begin(),
                                       loop_iter_end = loops.end();
         loop_iter != loop_iter_end; loop_iter++) {
      Loop *subLoop = *loop_iter;

      BranchedExtractLoop(DT, LI, subLoop, true);

      if (!subLoop->empty())
        recursiveExtractSubLoops(subLoop);
      //BranchedCodeExtractor(&DT).ExtractCodeRegion(loop,LI);
      //BranchedExtractLoop(DT, LI,subLoop, true);
      outs().changeColor(raw_ostream::RED);
      outs() << "subloop exctracted!!\n";
      outs().resetColor();
    }
  }
}

bool BranchedLoopExtractor::runOnLoop(Loop *L, LPPassManager &LPM) {
  // Only visit top-level loops.

  if (L->getParentLoop())
    return false;

  // If LoopSimplify form is not available, stay out of trouble.
  if (!L->isLoopSimplifyForm())
    return false;

  DominatorTree &DT = getAnalysis<DominatorTree>();
  bool Changed = false;

  // Extract the loop if it was not previously extracted:
  // interate through the kernelgen.extracted metadata nodes and
  // check whether the loop being extracted is already there.
  bool ShouldExtractLoop = true;
  Function *function = L->getHeader()->getParent();
  Module *m = function->getParent();
  string name = function->getName();
  if (NamedMDNode *extracted = m->getNamedMetadata("kernelgen.extracted")) {
    for (int i = 0, e = extracted->getNumOperands(); i != e; ++i) {
      MDNode *node = extracted->getOperand(i);
      ConstantDataArray *nameArray =
          dyn_cast<ConstantDataArray>(node->getOperand(0));
      assert(nameArray && "Invalid kernelgen.extracted metadata operand");
      assert(nameArray->isCString() &&
             "Invalid kernelgen_launch metadata operand");
      string extracted_name = nameArray->getAsCString();
      if (extracted_name == name) {
        ShouldExtractLoop = false;
        break;
      }
    }
  }
  bool extractWithBranch = true;
  LoopInfo &LI = getAnalysis<LoopInfo>();
  if (ShouldExtractLoop) {
    if (NumLoops == 0)
      return Changed;
    --NumLoops;

// Dmitry N. Mikushin: Удаление бранчей удалило юзы
// этих функций, и оптимизация их
// благополучно удалила, как неиспользуемые.
// Николай: может и так, но эти юзы
// определяются при вытаскивании ядра в
// отдельный модуль
// Николай: но альтернативных бранчей в
// мэйне быть не должно, это дублирование
// Dmitry N. Mikushin: ну почему же? А как ты
// реализуешь "внешний непараллельный цикл -
// на GPU" ?
// Николай: А зачем?
// Николай: если очень надо, то вернуть
// оставление бранчей в main можно изменением
// одной строчки
// Dmitry N. Mikushin: эффективность может быть
// разная - одни циклы выгружать имеет смысл,
// другие - нет
// Dmitry N. Mikushin: время покажет, нужно оно нам
// или нет, но сейчас было бы правильнее
// оставить эту гибкость, даже если она
// кажется лишней
// Николай: в branched code extractor поставь true в
// extractCodeRegion, вызываемом для самого внешнего
// цикла
#define PRESERVE_BRANCHES

#ifdef PRESERVE_BRANCHES
    CallInst *Call;
    if ((Call = BranchedExtractLoop(DT, LI, L, true)) != 0)
#endif
        {
      recursiveExtractSubLoops(L);

// After extraction, the loop is replaced by a function call, so
// we shouldn't try to run any more loop passes on it.
#ifndef PRESERVE_BRANCHES
      CallInst *Call = BranchedExtractLoop(DT, LI, L, false);
      if (Call)
#endif
          {
        ++NumBranchedExtracted;
        Changed = true;
      }

      if (LoopFunctionCalls)
        LoopFunctionCalls->push_back(Call);
      //	LPM.deleteLoopFromQueue(L);
    }
  }
  if (NumBranchedExtracted - tmpBranchedExtracted == 1) {
    tmpBranchedExtracted = NumBranchedExtracted;

    outs().changeColor(raw_ostream::GREEN);
    outs() << "KernelGen : NumExtractedLoops = " << tmpBranchedExtracted
           << " CurrentFunction:\"" << L->getHeader()->getParent()->getName()
           << "\" CurrentHeader:\"" << L->getHeader()->getName() << "\"\n";
    outs().resetColor();
  }

  return Changed;
}

namespace {
/// SingleLoopExtractor - For bugpoint.
struct SingleBranchedLoopExtractor : public BranchedLoopExtractor {
  static char ID; // Pass identification, replacement for typeid
  SingleBranchedLoopExtractor() : BranchedLoopExtractor(1) {}
};
} // End anonymous namespace

char SingleBranchedLoopExtractor::ID = 0;

INITIALIZE_PASS(SingleBranchedLoopExtractor, "loop-extract-with-branch-single",
                "Extract at most one loop into a new function and add branch",
                false, false)

// createSingleLoopExtractorPass - This pass extracts one natural loop from the
// program into a function if it can.  This is used by bugpoint.
//
Pass *llvm::createSingleBranchedLoopExtractorPass() {
  return new SingleBranchedLoopExtractor();
}
