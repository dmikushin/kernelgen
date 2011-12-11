#include "CodeGeneration.h"
namespace kernelgen
{
  Value* BlockGenerator::makeVectorOperand(Value *operand, int vectorWidth) {
    if (operand->getType()->isVectorTy())
      return operand;

    VectorType *vectorType = VectorType::get(operand->getType(), vectorWidth);
    Value *vector = UndefValue::get(vectorType);
    vector = Builder.CreateInsertElement(vector, operand, Builder.getInt32(0));

    std::vector<Constant*> splat;

    for (int i = 0; i < vectorWidth; i++)
      splat.push_back (Builder.getInt32(0));

    Constant *splatVector = ConstantVector::get(splat);

    return Builder.CreateShuffleVector(vector, vector, splatVector);
  }

  Value* BlockGenerator::getOperand(const Value *oldOperand, ValueMapT &BBMap,
                    ValueMapT *VectorMap) {
    const Instruction *OpInst = dyn_cast<Instruction>(oldOperand);

    if (!OpInst)
      return const_cast<Value*>(oldOperand);

    if (VectorMap && VectorMap->count(oldOperand))
      return (*VectorMap)[oldOperand];

    // IVS and Parameters.
    if (VMap.count(oldOperand)) {
      Value *NewOperand = VMap[oldOperand];

      // Insert a cast if types are different
      if (oldOperand->getType()->getScalarSizeInBits()
          < NewOperand->getType()->getScalarSizeInBits())
        NewOperand = Builder.CreateTruncOrBitCast(NewOperand,
                                                   oldOperand->getType());

      return NewOperand;
    }

    // Instructions calculated in the current BB.
    if (BBMap.count(oldOperand)) {
      return BBMap[oldOperand];
    }

    // Ignore instructions that are referencing ops in the old BB. These
    // instructions are unused. They where replace by new ones during
    // createIndependentBlocks().
    if (getRegion().contains(OpInst->getParent()))
      return NULL;

    return const_cast<Value*>(oldOperand);
  }

  Type *BlockGenerator::getVectorPtrTy(const Value *V, int vectorWidth) {
    PointerType *pointerType = dyn_cast<PointerType>(V->getType());
    assert(pointerType && "PointerType expected");

    Type *scalarType = pointerType->getElementType();
    VectorType *vectorType = VectorType::get(scalarType, vectorWidth);

    return PointerType::getUnqual(vectorType);
  }

  /// @brief Load a vector from a set of adjacent scalars
  ///
  /// In case a set of scalars is known to be next to each other in memory,
  /// create a vector load that loads those scalars
  ///
  /// %vector_ptr= bitcast double* %p to <4 x double>*
  /// %vec_full = load <4 x double>* %vector_ptr
  ///
  Value *BlockGenerator::generateStrideOneLoad(const LoadInst *load, ValueMapT &BBMap,
                               int size) {
    const Value *pointer = load->getPointerOperand();
    Type *vectorPtrType = getVectorPtrTy(pointer, size);
    Value *newPointer = getOperand(pointer, BBMap);
    Value *VectorPtr = Builder.CreateBitCast(newPointer, vectorPtrType,
                                             "vector_ptr");
    LoadInst *VecLoad = Builder.CreateLoad(VectorPtr,
                                        load->getNameStr()
                                        + "_p_vec_full");
    if (!flags.Aligned)
      VecLoad->setAlignment(8);

    return VecLoad;
  }

  /// @brief Load a vector initialized from a single scalar in memory
  ///
  /// In case all elements of a vector are initialized to the same
  /// scalar value, this value is loaded and shuffeled into all elements
  /// of the vector.
  ///
  /// %splat_one = load <1 x double>* %p
  /// %splat = shufflevector <1 x double> %splat_one, <1 x
  ///       double> %splat_one, <4 x i32> zeroinitializer
  ///
  Value *BlockGenerator::generateStrideZeroLoad(const LoadInst *load, ValueMapT &BBMap,
                                int size) {
    const Value *pointer = load->getPointerOperand();
    Type *vectorPtrType = getVectorPtrTy(pointer, 1);
    Value *newPointer = getOperand(pointer, BBMap);
    Value *vectorPtr = Builder.CreateBitCast(newPointer, vectorPtrType,
                                             load->getNameStr() + "_p_vec_p");
    LoadInst *scalarLoad= Builder.CreateLoad(vectorPtr,
                                          load->getNameStr() + "_p_splat_one");

    if (!flags.Aligned)
      scalarLoad->setAlignment(8);

    std::vector<Constant*> splat;

    for (int i = 0; i < size; i++)
      splat.push_back (Builder.getInt32(0));

    Constant *splatVector = ConstantVector::get(splat);

    Value *vectorLoad = Builder.CreateShuffleVector(scalarLoad, scalarLoad,
                                                    splatVector,
                                                    load->getNameStr()
                                                    + "_p_splat");
    return vectorLoad;
  }

  /// @Load a vector from scalars distributed in memory
  ///
  /// In case some scalars a distributed randomly in memory. Create a vector
  /// by loading each scalar and by inserting one after the other into the
  /// vector.
  ///
  /// %scalar_1= load double* %p_1
  /// %vec_1 = insertelement <2 x double> undef, double %scalar_1, i32 0
  /// %scalar 2 = load double* %p_2
  /// %vec_2 = insertelement <2 x double> %vec_1, double %scalar_1, i32 1
  ///
  Value *BlockGenerator::generateUnknownStrideLoad(const LoadInst *load,
                                   VectorValueMapT &scalarMaps,
                                   int size) {
    const Value *pointer = load->getPointerOperand();
    VectorType *vectorType = VectorType::get(
      dyn_cast<PointerType>(pointer->getType())->getElementType(), size);

    Value *vector = UndefValue::get(vectorType);

    for (int i = 0; i < size; i++) {
      Value *newPointer = getOperand(pointer, scalarMaps[i]);
      Value *scalarLoad = Builder.CreateLoad(newPointer,
                                             load->getNameStr() + "_p_scalar_");
      vector = Builder.CreateInsertElement(vector, scalarLoad,
                                           Builder.getInt32(i),
                                           load->getNameStr() + "_p_vec_");
    }

    return vector;
  }

  /// @brief Get the new operand address according to the changed access in
  ///        JSCOP file.
  Value *BlockGenerator::getNewAccessOperand(isl_map *newAccessRelation, Value *baseAddr,
                             const Value *oldOperand, ValueMapT &BBMap) {
    unsigned accessIdx = 0;
    Value *newOperand = Builder.CreateStructGEP(baseAddr,
                                                accessIdx, "p_newarrayidx_");
    return newOperand;
  }

  /// @brief Generate the operand address
  Value *BlockGenerator::generateLocationAccessed(const Instruction *Inst,
                                  const Value *pointer, ValueMapT &BBMap ) {
    polly::MemoryAccess &access = statement.getAccessFor(Inst);
    isl_map *newAccessRelation = access.getNewAccessFunction();
    if (!newAccessRelation) {
      Value *newPointer = getOperand(pointer, BBMap);
      return newPointer;
    }

    Value *baseAddr = const_cast<Value*>(access.getBaseAddr());
    Value *newPointer = getNewAccessOperand(newAccessRelation, baseAddr,
                                            pointer, BBMap);
    return newPointer;
  }

  Value *BlockGenerator::generateScalarLoad(const LoadInst *load, ValueMapT &BBMap) {
    const Value *pointer = load->getPointerOperand();
    const Instruction *Inst = dyn_cast<Instruction>(load);
    Value *newPointer = generateLocationAccessed(Inst, pointer, BBMap);
    Value *scalarLoad = Builder.CreateLoad(newPointer,
                                           load->getNameStr() + "_p_scalar_");
    return scalarLoad;
  }

  /// @brief Load a value (or several values as a vector) from memory.
  void BlockGenerator::generateLoad(const LoadInst *load, ValueMapT &vectorMap,
                    VectorValueMapT &scalarMaps, int vectorWidth) {

    if (scalarMaps.size() == 1) {
      scalarMaps[0][load] = generateScalarLoad(load, scalarMaps[0]);
      return;
    }

    Value *newLoad;

    polly::MemoryAccess &Access = statement.getAccessFor(load);

    assert(scatteringDomain && "No scattering domain available");

    if (Access.isStrideZero(scatteringDomain))
      newLoad = generateStrideZeroLoad(load, scalarMaps[0], vectorWidth);
    else if (Access.isStrideOne(scatteringDomain))
      newLoad = generateStrideOneLoad(load, scalarMaps[0], vectorWidth);
    else
      newLoad = generateUnknownStrideLoad(load, scalarMaps, vectorWidth);

    vectorMap[load] = newLoad;
  }

  void BlockGenerator::copyInstruction(const Instruction *Inst, ValueMapT &BBMap,
                       ValueMapT &vectorMap, VectorValueMapT &scalarMaps,
                       int vectorDimension, int vectorWidth) {
    // If this instruction is already in the vectorMap, a vector instruction
    // was already issued, that calculates the values of all dimensions. No
    // need to create any more instructions.
    if (vectorMap.count(Inst))
      return;

    // Terminator instructions control the control flow. They are explicitally
    // expressed in the clast and do not need to be copied.
    if (Inst->isTerminator())
      return;

    if (const LoadInst *load = dyn_cast<LoadInst>(Inst)) {
      generateLoad(load, vectorMap, scalarMaps, vectorWidth);
      return;
    }

    if (const BinaryOperator *binaryInst = dyn_cast<BinaryOperator>(Inst)) {
      Value *opZero = Inst->getOperand(0);
      Value *opOne = Inst->getOperand(1);

      // This is an old instruction that can be ignored.
      if (!opZero && !opOne)
        return;

      bool isVectorOp = vectorMap.count(opZero) || vectorMap.count(opOne);

      if (isVectorOp && vectorDimension > 0)
        return;

      Value *newOpZero, *newOpOne;
      newOpZero = getOperand(opZero, BBMap, &vectorMap);
      newOpOne = getOperand(opOne, BBMap, &vectorMap);


      std::string name;
      if (isVectorOp) {
        newOpZero = makeVectorOperand(newOpZero, vectorWidth);
        newOpOne = makeVectorOperand(newOpOne, vectorWidth);
        name =  Inst->getNameStr() + "p_vec";
      } else
        name = Inst->getNameStr() + "p_sca";

      Value *newInst = Builder.CreateBinOp(binaryInst->getOpcode(), newOpZero,
                                           newOpOne, name);
      if (isVectorOp)
        vectorMap[Inst] = newInst;
      else
        BBMap[Inst] = newInst;

      return;
    }

    if (const StoreInst *store = dyn_cast<StoreInst>(Inst)) {
      if (vectorMap.count(store->getValueOperand()) > 0) {

        // We only need to generate one store if we are in vector mode.
        if (vectorDimension > 0)
          return;

        polly::MemoryAccess &Access = statement.getAccessFor(store);

        assert(scatteringDomain && "No scattering domain available");

        const Value *pointer = store->getPointerOperand();
        Value *vector = getOperand(store->getValueOperand(), BBMap, &vectorMap);

        if (Access.isStrideOne(scatteringDomain)) {
          Type *vectorPtrType = getVectorPtrTy(pointer, vectorWidth);
          Value *newPointer = getOperand(pointer, BBMap, &vectorMap);

          Value *VectorPtr = Builder.CreateBitCast(newPointer, vectorPtrType,
                                                   "vector_ptr");
          StoreInst *Store = Builder.CreateStore(vector, VectorPtr);

          if (!flags.Aligned)
            Store->setAlignment(8);
        } else {
          for (unsigned i = 0; i < scalarMaps.size(); i++) {
            Value *scalar = Builder.CreateExtractElement(vector,
                                                         Builder.getInt32(i));
            Value *newPointer = getOperand(pointer, scalarMaps[i]);
            Builder.CreateStore(scalar, newPointer);
          }
        }

        return;
      }
    }

    Instruction *NewInst = Inst->clone();

    // Copy the operands in temporary vector, as an in place update
    // fails if an instruction is referencing the same operand twice.
    std::vector<Value*> Operands(NewInst->op_begin(), NewInst->op_end());

    // Replace old operands with the new ones.
    for (std::vector<Value*>::iterator UI = Operands.begin(),
         UE = Operands.end(); UI != UE; ++UI) {
      Value *newOperand = getOperand(*UI, BBMap);

      if (!newOperand) {
        assert(!isa<StoreInst>(NewInst)
               && "Store instructions are always needed!");
        delete NewInst;
        return;
      }

      NewInst->replaceUsesOfWith(*UI, newOperand);
    }

    Builder.Insert(NewInst);
    BBMap[Inst] = NewInst;

    if (!NewInst->getType()->isVoidTy())
      NewInst->setName("p_" + Inst->getName());
  }


  // Insert a copy of a basic block in the newly generated code.
  //
  // @param Builder The builder used to insert the code. It also specifies
  //                where to insert the code.
  // @param BB      The basic block to copy
  // @param VMap    A map returning for any old value its new equivalent. This
  //                is used to update the operands of the statements.
  //                For new statements a relation old->new is inserted in this
  //                map.
  void BlockGenerator::copyBB(BasicBlock *BB, DominatorTree *DT) {
    Function *F = Builder.GetInsertBlock()->getParent();
    LLVMContext &Context = F->getContext();
    BasicBlock *CopyBB = BasicBlock::Create(Context,
                                            "polly.stmt_" + BB->getNameStr(),
                                            F);
    Builder.CreateBr(CopyBB);
    DT->addNewBlock(CopyBB, Builder.GetInsertBlock());
    Builder.SetInsertPoint(CopyBB);

    // Create two maps that store the mapping from the original instructions of
    // the old basic block to their copies in the new basic block. Those maps
    // are basic block local.
    //
    // As vector code generation is supported there is one map for scalar values
    // and one for vector values.
    //
    // In case we just do scalar code generation, the vectorMap is not used and
    // the scalarMap has just one dimension, which contains the mapping.
    //
    // In case vector code generation is done, an instruction may either appear
    // in the vector map once (as it is calculating >vectorwidth< values at a
    // time. Or (if the values are calculated using scalar operations), it
    // appears once in every dimension of the scalarMap.
    VectorValueMapT scalarBlockMap(getVectorSize());
    ValueMapT vectorBlockMap;

    for (BasicBlock::const_iterator II = BB->begin(), IE = BB->end();
         II != IE; ++II)
      for (int i = 0; i < getVectorSize(); i++) {
        if (isVectorBlock())
          VMap = ValueMaps[i];

        copyInstruction(II, scalarBlockMap[i], vectorBlockMap,
                        scalarBlockMap, i, getVectorSize());
      }
  }
};
