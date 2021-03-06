//===- ConstantSubstitution.cpp - LLVM pass for constants substitution ----===//
//
//     KernelGen -- A prototype of LLVM-based auto-parallelizing Fortran/C
//        compiler for NVIDIA GPUs, targeting numerical modeling code.
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// TODO This file implements ...
//
//===----------------------------------------------------------------------===//

#include "llvm/IR/DataLayout.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Type.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Constants.h"
#include <set>
#include <map>
#include <iostream>

#include "Runtime.h"

using namespace kernelgen;
using namespace llvm;

std::map<LoadInst *, int> LoadInstOffsets;

struct UseTreeNode {
  const Value *sourceValue;
  int currentOffset;
};

typedef std::set<UseTreeNode *>::iterator UseIterator;
typedef std::map<LoadInst *, int>::iterator MapIterator;

enum AllowedInstructions {
  GEP = 1,
  Load,
  Cast,
  Store
};

int isAllowedInstruction(const Value *const &someValue) {
  const Instruction *const instruction = dyn_cast<Instruction>(someValue);
  if (!instruction)
    return false;
  if (dyn_cast<GetElementPtrInst>(instruction))
    return GEP;
  if (dyn_cast<LoadInst>(instruction))
    return Load;
  if (dyn_cast<StoreInst>(instruction))
    return Store;
  if (dyn_cast<CastInst>(instruction))
    return Cast;
  return 0;
}

void makeUseTreeForIntegerArgs(UseTreeNode *const &useNode,
                               const DataLayout *const &DL) {
  const Value *sourceValue = useNode->sourceValue;
  for (Value::const_use_iterator useIterator = sourceValue->use_begin(),
                                 useEnd = sourceValue->use_end();
       useIterator != useEnd; useIterator++) {
    const Value *User = *useIterator;
    int inst;

    // FIXME :
    // assume that before using of arg it must be loaded by LoadInst from some
    // GEP or BitCast
    // before LoadInst in each branch of UseTree must be only GEPs and BitCasts
    assert(inst = isAllowedInstruction(User));

    UseTreeNode *newUseNode;
    if (inst == GEP || inst == Cast) {
      newUseNode = new UseTreeNode();
      newUseNode->sourceValue = User;
      newUseNode->currentOffset = useNode->currentOffset;
    }
    switch (inst) {
    case GEP: {
      // GEP increase offset and change type
      // new Type is newUseNode->sourceValue->getType()
      const GetElementPtrInst *GEPInst = dyn_cast<GetElementPtrInst>(User);
      std::vector<Value *> Indices(GEPInst->idx_begin(), GEPInst->idx_end());
      newUseNode->currentOffset += DL->getIndexedOffset(
          useNode->sourceValue->getType(), // old type from which we make GEP
          ArrayRef<Value *>(Indices));     // Indices of GEP
      makeUseTreeForIntegerArgs(newUseNode, DL);
    } break;
    case Cast:
      // Cast change only Type
      // new Type is newUseNode->sourceValue->getType()
      makeUseTreeForIntegerArgs(newUseNode, DL);
      break;
    case Load:
      LoadInstOffsets[(LoadInst *)User] = useNode->currentOffset;
      //alignment?
      break;
    case Store:
      break;
    }
    if (inst == GEP || inst == Cast)
      delete newUseNode;
  }
}

void computeLoadInstOffsets(const Value *sourceArg, const DataLayout *DL) {
  UseTreeNode *useTree = new UseTreeNode();
  useTree->currentOffset = 0;
  useTree->sourceValue = sourceArg;

  // Bypass Use Tree of argument
  // side effect - fill in LoadInstOffsets
  makeUseTreeForIntegerArgs(useTree, DL);
  delete useTree;
}

void ConstantSubstitution(Function *func, void *args) {
  LoadInstOffsets.clear();
  Function::arg_iterator AI = func->arg_begin();
  Value *sourceArg = (Value *)AI;

  // Argument must be a pointer.
  assert(sourceArg->getType()->isPointerTy());
  DataLayout *DL = new DataLayout(func->getParent());

  // Compute offsets of args (each arg is LoadInst from
  // some offset in structure of args).
  computeLoadInstOffsets(sourceArg, DL);

  VERBOSE(Verbose::Polly << "    Integer args substituted:\n"
                         << Verbose::Default);
  for (MapIterator arg = LoadInstOffsets.begin(),
                   argEnd = LoadInstOffsets.end();
       arg != argEnd; arg++) {
    LoadInst *load = arg->first;
    Type *type = load->getType();

    // For each integer arg: replace uses of arg's LoadInst by Constant.
    if (type->isIntegerTy()) {
      assert(DL->getTypeStoreSize(type) <= 8);
      uint64_t value = 0;
      int offset = arg->second;
      memcpy(&value, ((char *)args) + offset, DL->getTypeStoreSize(type));
      ConstantInt *constant = ConstantInt::get(cast<IntegerType>(type), value);
      load->replaceAllUsesWith(constant);
      load->eraseFromParent();
      VERBOSE(Verbose::Polly
              << "        offset = " << arg->second
              << ", value = " << constant->getValue().toString(10, true) << "\n"
              << Verbose::Default);
      continue;
    }
    if (type->isPointerTy()) {
      assert(DL->getTypeStoreSize(type) <= 8);
      uint64_t ptrValue = 0;
      int offset = arg->second;
      memcpy(&ptrValue, ((char *)args) + offset, DL->getTypeStoreSize(type));

      load->replaceAllUsesWith(ConstantExpr::getIntToPtr(
          ConstantInt::get(DL->getIntPtrType(func->getParent()->getContext()),
                           (uint64_t) ptrValue, false),
          type));

      load->eraseFromParent();

      VERBOSE(Verbose::Polly << "        offset = " << arg->second
                             << ", ptrValue = " << ptrValue << "\n"
                             << Verbose::Default);
      continue;
    }

    //THROW("Only integer and pointer constants could be substituted");
  }
}
