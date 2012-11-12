#include "Codegen.h"

namespace kernelgen
{
char KernelCodeGenerator::ID = 100;
std::vector<string>  dimensions;
std::vector<const char*> CudaFunctions;
DenseMap<const char*,const char *> CudaInricics;
bool CodeGenForCuda;
int ClastStmtCodeGen::totalNumberOfParallelLoops;
VectorizerChoice PollyVectorizerChoice;

Value *ClastExpCodeGen::codegen(const clast_name *e, Type *Ty)
{
	CharMapT::const_iterator I = IVS.find(e->name);

	assert(I != IVS.end() && "Clast name not found");

	return Builder.CreateSExtOrBitCast(I->second, Ty);
}

Value *ClastExpCodeGen::codegen(const clast_term *e, Type *Ty)
{
	APInt a = APInt_from_MPZ(e->val);

	Value *ConstOne = ConstantInt::get(Builder.getContext(), a);
	ConstOne = Builder.CreateSExtOrBitCast(ConstOne, Ty);

	if (!e->var)
		return ConstOne;

	Value *var = codegen(e->var, Ty);
	return Builder.CreateMul(ConstOne, var);
}

Value *ClastExpCodeGen::codegen(const clast_binary *e, Type *Ty)
{
	Value *LHS = codegen(e->LHS, Ty);

	APInt RHS_AP = APInt_from_MPZ(e->RHS);

	Value *RHS = ConstantInt::get(Builder.getContext(), RHS_AP);
	RHS = Builder.CreateSExtOrBitCast(RHS, Ty);

	switch (e->type) {
	case clast_bin_mod:
		return Builder.CreateSRem(LHS, RHS);
	case clast_bin_fdiv: {
		// floord(n,d) ((n < 0) ? (n - d + 1) : n) / d
		Value *One = ConstantInt::get(Ty, 1);
		Value *Zero = ConstantInt::get(Ty, 0);
		Value *Sum1 = Builder.CreateSub(LHS, RHS);
		Value *Sum2 = Builder.CreateAdd(Sum1, One);
		Value *isNegative = Builder.CreateICmpSLT(LHS, Zero);
		Value *Dividend = Builder.CreateSelect(isNegative, Sum2, LHS);
		return Builder.CreateSDiv(Dividend, RHS);
	}
	case clast_bin_cdiv: {
		// ceild(n,d) ((n < 0) ? n : (n + d - 1)) / d
		Value *One = ConstantInt::get(Ty, 1);
		Value *Zero = ConstantInt::get(Ty, 0);
		Value *Sum1 = Builder.CreateAdd(LHS, RHS);
		Value *Sum2 = Builder.CreateSub(Sum1, One);
		Value *isNegative = Builder.CreateICmpSLT(LHS, Zero);
		Value *Dividend = Builder.CreateSelect(isNegative, LHS, Sum2);
		return Builder.CreateSDiv(Dividend, RHS);
	}
	case clast_bin_div:
		return Builder.CreateSDiv(LHS, RHS);
	};

	llvm_unreachable("Unknown clast binary expression type");
}

Value *ClastExpCodeGen::codegen(const clast_reduction *r, Type *Ty)
{
	assert((   r->type == clast_red_min
	           || r->type == clast_red_max
	           || r->type == clast_red_sum)
	       && "Clast reduction type not supported");
	Value *old = codegen(r->elts[0], Ty);

	for (int i=1; i < r->n; ++i) {
		Value *exprValue = codegen(r->elts[i], Ty);

		switch (r->type) {
		case clast_red_min: {
			Value *cmp = Builder.CreateICmpSLT(old, exprValue);
			old = Builder.CreateSelect(cmp, old, exprValue);
			break;
		}
		case clast_red_max: {
			Value *cmp = Builder.CreateICmpSGT(old, exprValue);
			old = Builder.CreateSelect(cmp, old, exprValue);
			break;
		}
		case clast_red_sum:
			old = Builder.CreateAdd(old, exprValue);
			break;
		}
	}

	return old;
}

ClastExpCodeGen::ClastExpCodeGen(IRBuilder<> &B, CharMapT &IVMap)
	: Builder(B), IVS(IVMap) {}

Value *ClastExpCodeGen::codegen(const clast_expr *e, Type *Ty)
{
	switch(e->type) {
	case clast_expr_name:
		return codegen((const clast_name *)e, Ty);
	case clast_expr_term:
		return codegen((const clast_term *)e, Ty);
	case clast_expr_bin:
		return codegen((const clast_binary *)e, Ty);
	case clast_expr_red:
		return codegen((const clast_reduction *)e, Ty);
	}

	llvm_unreachable("Unknown clast expression!");
}

ClastStmtCodeGen::ClastStmtCodeGen(Scop *scop, IRBuilder<> &B, Pass *P) :
	S(scop), P(P), Builder(B), ExpGen(Builder, ClastVars) {}


void ClastStmtCodeGen::codegenForSequential(const clast_for *f) {
  Value *LowerBound, *UpperBound, *IV, *Stride;
  BasicBlock *AfterBB;
  Type *IntPtrTy = getIntPtrTy();

  LowerBound = ExpGen.codegen(f->LB, IntPtrTy);
  UpperBound = ExpGen.codegen(f->UB, IntPtrTy);
  Stride = Builder.getInt(APInt_from_MPZ(f->stride));

  IV = createLoop(LowerBound, UpperBound, Stride, Builder, P, AfterBB);

  // Add loop iv to symbols.
  ClastVars[f->iterator] = IV;

  if (f->body)
    codegen(f->body);

  // Loop is finished, so remove its iv from the live symbols.
  ClastVars.erase(f->iterator);
  Builder.SetInsertPoint(AfterBB->begin());
}

void ClastStmtCodeGen::codegenForCUDA(const clast_for *f)
{
	
	// At this point GridParamsBB and PosInGridBB BasicBlocks are already created.
	// The needed Value*-es are stored in positionInGrid, GridSize, BlockSize for each thread.
	int dimension = loopsGenerated-1;
	int inverseDimension = 3 - totalNumberOfParallelLoops + startDepth + dimension;
	const char * dimensionName = dimensions[inverseDimension].c_str();

	// In CountBoundsBB BasicBlock for each dimension compute:
	//   CountOfIterations
	//   ThreadUpperBound
	//   ThreadLowerBound
	//   Stride
	// These values are different between threads in runtime.
	BasicBlock * CountBoundsBB = Builder.GetInsertBlock();
	CountBoundsBB->setName(string("CUDA.CountBounds.") + dimensionName);

	IntegerType * IntType = Type::getInt32Ty(Builder.getContext());

	// Lower and Upper Bounds of Loop
	Value *lowerBound = ExpGen.codegen(f->LB,IntType);
	Value *upperBound = ExpGen.codegen(f->UB,IntType);

	// Stride of loop
	assert(kernelgen::APInt_from_MPZ(f->stride) != 0 && "what we must do in those situation?");
	assert(kernelgen::APInt_from_MPZ(f->stride).getSExtValue() > 0 && "TODO: support of negative stride");

	Value *LoopStride = ConstantInt::get(IntType,
	                                     kernelgen::APInt_from_MPZ(f->stride).zext(IntType->getBitWidth()));

	// The number of loop's iterations.
	// ((UpperBound - LowerBound) / stride + 1)
	// The number of iterations minus one = (UpperBound - LowerBound) / stride
	Value *UpperMinusLower = Builder.CreateSub(upperBound,lowerBound,
	                         string("UB.minus.LB.") + dimensionName);
	Value *NumOfIterationsMinusOne = Builder.CreateSDiv(UpperMinusLower,LoopStride,
	                                 string("NumOfIterationsMinusOne.") + dimensionName);

	// Compute number of Iterations per thread.
	// ((NumberOfIterations - 1) / GridSize) + 1)
	// ( NumOfIterationsMinusOne / GridSize + 1)
	Value *One = ConstantInt::get(lowerBound->getType(), 1);
	Value *IterationsPerThreadMinusOne = Builder.CreateSDiv(
	        NumOfIterationsMinusOne, GridSize[dimension],
	        string("IterationsPerThreadMinusOne.") + dimensionName);
	Value *IterationsPerThread = Builder.CreateAdd(
	                                 IterationsPerThreadMinusOne, One,
	                                 string("IterationsPerThread.") + dimensionName);

	// Compute Thread's Upper and Lower Bounds and Stride
	// ThreadLowerBound = LoopStride * (IterationsPerThread * BlockPosition + ThreadPositionInBlock)
	// ThreadUpperBound = ThreadLowerBound + ThreadStride * (IterationsPerThread - 1)
	// Stride = BlockSize (to increase probability of coalescing transactions to memory)
	Value *BlockLowerBound = Builder.CreateMul(
	                             BlockPositionInGrid[dimension], IterationsPerThread,
	                             string("BlockLowerBound.") + dimensionName);
	Value *BlockLBAddThreadPosInBlock = Builder.CreateAdd(
	                                        BlockLowerBound, ThreadPositionInBlock[dimension],
	                                        string("BlockLB.Add.ThreadPosInBlock.") + dimensionName);
	Value *ThreadLowerBound = Builder.CreateMul(
	                              BlockLBAddThreadPosInBlock, LoopStride,
	                              string("ThreadLowerBound.") + dimensionName);
	
	/*Value *ThreadStride = Builder.CreateMul(
	                          LoopStride, BlockSize[dimension],
	                          string("ThreadStride.") + dimensionName);
	Value *StrideMulIterPerThreadMinusOne = Builder.CreateMul(
	        IterationsPerThreadMinusOne, ThreadStride,
	        string("ThreadStride.Mul.IterPerThreadMinusOne.") + dimensionName);
	Value *ThreadUpperBound = Builder.CreateAdd(
	                              ThreadLowerBound, StrideMulIterPerThreadMinusOne,
	                              string("ThreadUpperBound.") + dimensionName);*/
	
	// Make block for truncation of threadUpperBound.
	BasicBlock *truncateThreadUB = SplitBlock(CountBoundsBB, CountBoundsBB->getTerminator(), P);
	truncateThreadUB->setName(string("CUDA.truncateThreadUB.") + dimensionName);

    Builder.SetInsertPoint(truncateThreadUB->getTerminator());
	
	Value *ThreadStride = Builder.CreateMul(
	                          LoopStride, BlockSize[dimension],
	                          string("ThreadStride.") + dimensionName);
	Value *StrideMulIterPerThreadMinusOne = Builder.CreateMul(
	        IterationsPerThreadMinusOne, ThreadStride,
	        string("ThreadStride.Mul.IterPerThreadMinusOne.") + dimensionName);
	Value *ThreadUpperBound = Builder.CreateAdd(
	                              ThreadLowerBound, StrideMulIterPerThreadMinusOne,
	                              string("ThreadUpperBound.") + dimensionName);
								  
	// if(threadUpperBound > loopUpperBound) threadUpperBound = loopUpperBound;
	Value *isThreadUBgtLoopUB = Builder.CreateICmpSGT(
	                                ThreadUpperBound, upperBound, string("isThreadUBgtLoopUB.") + dimensionName);
	Value *truncatedThreadUB = Builder.CreateSelect(
	                       isThreadUBgtLoopUB, upperBound, ThreadUpperBound,
	                      string("truncatedThreadUB.") + dimensionName);

	// Get terminator of CountBoundsBB.
	TerminatorInst * terminator = CountBoundsBB->getTerminator();
	Builder.SetInsertPoint(CountBoundsBB);
	// if(threadLowerBound > loopUpperBound) then no execute body et all
	Value *isThreadLBgtLoopUB = Builder.CreateICmpSGT(
	                                ThreadLowerBound, upperBound, string("isThreadLBgtLoopUB.") + dimensionName);
	Builder.CreateCondBr(isThreadLBgtLoopUB, *succ_begin(truncateThreadUB), truncateThreadUB);
	terminator->eraseFromParent();
	
    // Generate code for loop with computed bounds and stride
	// CountBoundsBB BasicBlock is a preheader of that loop
	BasicBlock *AfterBB;
	Builder.SetInsertPoint(truncateThreadUB->getTerminator());
	Value *IV = createLoopForCUDA(&Builder, lowerBound, upperBound,
	                              ThreadLowerBound, truncatedThreadUB, ThreadStride, dimensionName, P, &AfterBB);

    cast<BranchInst>(CountBoundsBB->getTerminator())->setSuccessor(0,AfterBB);
    
	ClastVars[f->iterator] = IV;
	if (f->body) codegen(f->body);

	// Loop is finished, so remove its iv from the live symbols.
	ClastVars.erase(f->iterator);
	AfterBB->moveAfter(Builder.GetInsertBlock());

	Builder.SetInsertPoint(AfterBB->begin());
}



void ClastStmtCodeGen::createCUDAGridParamsAndPosInGridBlocks()
{
	Module *M = Builder.GetInsertBlock()->getParent()->getParent();
	vector<Value *> GridParameters;

	// GridParams BasicBlock - load Grid parameters by calling CUDA functions
	BasicBlock *GridParamsBB = Builder.GetInsertBlock();
	GridParamsBB->setName("CUDA.getGridParams");

	// PosInGrid BasicBlock - compute thread positin in grid
	BasicBlock *PosInGridBB=SplitBlock(GridParamsBB,GridParamsBB->getTerminator(),P);
	PosInGridBB->setName("CUDA.getPositionInGrid");

	// Compute needed values separately for each dimension.
	for(int dimension = 0; dimension < loopCount; dimension++) {
		Builder.SetInsertPoint(GridParamsBB->getTerminator());

        int inverseDimension = 3 - totalNumberOfParallelLoops + startDepth + dimension;
		
		// Call CUDA functions and store values in vector GridParameters.
		for(int GridParameter = 0; GridParameter < 4; GridParameter ++) {
			GridParameters.push_back(Builder.CreateCall(
			                             M->getFunction(CudaFunctions[ inverseDimension * 4 + GridParameter]),
			                             CudaInricics[CudaFunctions[ inverseDimension * 4 + GridParameter]] ));
		}
		
        const char * dimensionName = dimensions[inverseDimension].c_str();

		Builder.SetInsertPoint(PosInGridBB->getTerminator());

		// Grid Parameters for current dimension
		Value *threadId = GridParameters[dimension * 4 + 0];
		Value *blockId = GridParameters[dimension * 4 + 1];
		Value *blockDim = GridParameters[dimension * 4 + 2];
		Value *gridDim =  GridParameters[dimension * 4 + 3];

		// Absolute position of block's first thread (position of block)
		// blockId.x * blockDim.x - "position of block's thread 0 for dimension x"
		Value* Position = Builder.CreateMul(blockId, blockDim,
		                                    string("PositionOfBlockInGrid.") + dimensionName);

		// GridDim.x * blockDim.x - size of grid in threads for dimension x
		Value * Size = Builder.CreateMul(gridDim, blockDim,
		                                 string("GridSize.") + dimensionName);

		// Store values.
		BlockPositionInGrid.push_back(Position);
		ThreadPositionInBlock.push_back(threadId);
		GridSize.push_back(Size);
		BlockSize.push_back(blockDim);
	}

	BasicBlock *LoopPreheader=SplitBlock(PosInGridBB,PosInGridBB->getTerminator(),P);
	Builder.SetInsertPoint(LoopPreheader->getTerminator());
}
void ClastStmtCodeGen::addParameters(const CloogNames *names) {
  llvm::SCEVExpander Rewriter(P->getAnalysis<llvm::ScalarEvolution>(), "polly");

  int i = 0;
  for (Scop::param_iterator PI = S->param_begin(), PE = S->param_end();
       PI != PE; ++PI) {
    assert(i < names->nb_parameters && "Not enough parameter names");

    const llvm::SCEV *Param = *PI;
    Type *Ty = Param->getType();
	assert(Param->getSCEVType() == llvm::scUnknown);
	
		const SCEVUnknown *unknownParam = cast<const SCEVUnknown>(Param);
		Value * newValue = ValueMap[unknownParam->getValue()];
		assert(newValue && dyn_cast<Argument>(newValue));
		const SCEV *newParam = P->getAnalysis<ScalarEvolution>().getUnknown(newValue);
		
	// заменить все Value * из SCEVUnknown на соответствующие из ValueMap
	// Думаю, всё же придётся разрешать участие вызовов функций в параметрах скопа
	// В этом случае новые адреса инструкций вызовов должны будут добавлены в ClastVars по мере генерации
	
    Instruction *insertLocation = --(Builder.GetInsertBlock()->end());
    Value *V = Rewriter.expandCodeFor(newParam, Ty, insertLocation);
    ClastVars[names->parameters[i]] = V;

    ++i;
  }
}


void ClastStmtCodeGen::codegenSubstitutions(const clast_stmt *Assignment,
                                             ScopStmt *Statement, int vectorDim, std::vector<ValueMapT> *VectorVMap) {
  int Dimension = 0;

  while (Assignment) {
    assert(CLAST_STMT_IS_A(Assignment, stmt_ass)
           && "Substitions are expected to be assignments");
    codegen((const clast_assignment *)Assignment, Statement, Dimension,
            vectorDim, VectorVMap);
    Assignment = Assignment->next;
    Dimension++;
  }
}
 Pass *createKernelCodeGeneratorPass()
 {
	 return new KernelCodeGenerator();
 }
}

using namespace kernelgen;
INITIALIZE_PASS_BEGIN(KernelCodeGenerator, "kernel-cloog",
                      "Execute kernel Cloog code generation", false, false)
INITIALIZE_PASS_DEPENDENCY(KernelCloogInfo)
INITIALIZE_PASS_DEPENDENCY(KernelScopInfo)
INITIALIZE_PASS_DEPENDENCY(RegionInfo)
INITIALIZE_PASS_DEPENDENCY(ScalarEvolution)
INITIALIZE_PASS_END(KernelCodeGenerator, "kernel-cloog",
                    "Execute kernel Cloog code generation", false, false)
					
static struct A {
	A() {
		PassRegistry &Registry = *PassRegistry::getPassRegistry();
		initializeKernelCodeGeneratorPass(Registry);
	}
} ARegister;
					
