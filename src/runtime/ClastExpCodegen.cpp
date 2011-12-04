
#include "CodeGeneration.h"
namespace kernelgen
{

Value *ClastExpCodeGen::codegen(const clast_name *e, Type *Ty)
{
	CharMapT::const_iterator I = IVS->find(e->name);

	if (I != IVS->end())
		return Builder.CreateSExtOrBitCast(I->second, Ty);
	else
		llvm_unreachable("Clast name not found");
}

Value *ClastExpCodeGen::codegen(const clast_term *e, Type *Ty)
{
	APInt a = polly::APInt_from_MPZ(e->val);

	Value *ConstOne = ConstantInt::get(Builder.getContext(), a);
	ConstOne = Builder.CreateSExtOrBitCast(ConstOne, Ty);

	if (e->var) {
		Value *var = codegen(e->var, Ty);
		return Builder.CreateMul(ConstOne, var);
	}

	return ConstOne;
}

Value *ClastExpCodeGen::codegen(const clast_binary *e, Type *Ty)
{
	Value *LHS = codegen(e->LHS, Ty);

	APInt RHS_AP = polly::APInt_from_MPZ(e->RHS);

	Value *RHS = ConstantInt::get(Builder.getContext(), RHS_AP);
	RHS = Builder.CreateSExtOrBitCast(RHS, Ty);

	switch (e->type) {
	case clast_bin_mod:
		return Builder.CreateSRem(LHS, RHS);
	case clast_bin_fdiv: {
		// floord(n,d) ((n < 0) ? (n - d + 1) : n) / d
		Value *One = ConstantInt::get(Builder.getInt1Ty(), 1);
		Value *Zero = ConstantInt::get(Builder.getInt1Ty(), 0);
		One = Builder.CreateZExtOrBitCast(One, Ty);
		Zero = Builder.CreateZExtOrBitCast(Zero, Ty);
		Value *Sum1 = Builder.CreateSub(LHS, RHS);
		Value *Sum2 = Builder.CreateAdd(Sum1, One);
		Value *isNegative = Builder.CreateICmpSLT(LHS, Zero);
		Value *Dividend = Builder.CreateSelect(isNegative, Sum2, LHS);
		return Builder.CreateSDiv(Dividend, RHS);
	}
	case clast_bin_cdiv: {
		// ceild(n,d) ((n < 0) ? n : (n + d - 1)) / d
		Value *One = ConstantInt::get(Builder.getInt1Ty(), 1);
		Value *Zero = ConstantInt::get(Builder.getInt1Ty(), 0);
		One = Builder.CreateZExtOrBitCast(One, Ty);
		Zero = Builder.CreateZExtOrBitCast(Zero, Ty);
		Value *Sum1 = Builder.CreateAdd(LHS, RHS);
		Value *Sum2 = Builder.CreateSub(Sum1, One);
		Value *isNegative = Builder.CreateICmpSLT(LHS, Zero);
		Value *Dividend = Builder.CreateSelect(isNegative, LHS, Sum2);
		return Builder.CreateSDiv(Dividend, RHS);
	}
	case clast_bin_div:
		return Builder.CreateSDiv(LHS, RHS);
	default:
		llvm_unreachable("Unknown clast binary expression type");
	};
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
		default:
			llvm_unreachable("Clast unknown reduction type");
		}
	}

	return old;
}

// Generates code to calculate a given clast expression.
//
// @param e The expression to calculate.
// @return The Value that holds the result.
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
	default:
		llvm_unreachable("Unknown clast expression!");
	}
}

void createLoop(IRBuilder<> *Builder, Value *LB, Value *UB, APInt Stride,
                       PHINode*& IV, BasicBlock*& AfterBB, Value*& IncrementedIV,
                       DominatorTree *DT)
{
	
	Function *F = Builder->GetInsertBlock()->getParent();
	
	LLVMContext &Context = F->getContext();

	BasicBlock *PreheaderBB = Builder->GetInsertBlock();
	BasicBlock *HeaderBB = BasicBlock::Create(Context, "polly.LoopHeader", F);
	BasicBlock *BodyBB = BasicBlock::Create(Context, "polly.LoopBody", F);
	
	AfterBB = BasicBlock::Create(Context, "polly.AfterLoop", F); //!!!!!!!!!!!!!
	Builder->CreateBr(HeaderBB);
	DT->addNewBlock(HeaderBB, PreheaderBB);

	Builder->SetInsertPoint(HeaderBB);

	// Use the type of upper and lower bound.
	assert(LB->getType() == UB->getType()
	       && "Different types for upper and lower bound.");

	IntegerType *LoopIVType = dyn_cast<IntegerType>(UB->getType());
	assert(LoopIVType && "UB is not integer?");

	// IV
	IV = Builder->CreatePHI(LoopIVType, 2, "polly.loopiv");
	IV->addIncoming(LB, PreheaderBB);

	// IV increment.
	Value *StrideValue = ConstantInt::get(LoopIVType,
	                                      Stride.zext(LoopIVType->getBitWidth()));
	
	IncrementedIV = Builder->CreateAdd(IV, StrideValue, "polly.next_loopiv");

	// Exit condition.
	if (flags.AtLeastOnce) { // At least on iteration.
		UB = Builder->CreateAdd(UB, Builder->getInt64(1));
		Value *CMP = Builder->CreateICmpEQ(IV, UB);
		Builder->CreateCondBr(CMP, AfterBB, BodyBB);
	} else { // Maybe not executed at all.
		Value *CMP = Builder->CreateICmpSLE(IV, UB);
		Builder->CreateCondBr(CMP, BodyBB, AfterBB);
	}
	DT->addNewBlock(BodyBB, HeaderBB);
	DT->addNewBlock(AfterBB, HeaderBB);

	Builder->SetInsertPoint(BodyBB);
}
void createLoopForCUDA(IRBuilder<> *Builder, Value *LB, Value *UB, 
                       Value * ThreadLB, Value * ThreadUB, Value * ThreadStride,
                       PHINode*& IV, BasicBlock*& AfterBB, Value*& IncrementedIV,
                       DominatorTree *DT, const char * dimension)
					   {
						   	Function *F = Builder->GetInsertBlock()->getParent();
	LLVMContext &Context = F->getContext();

	BasicBlock *PreheaderBB = Builder->GetInsertBlock();
	BasicBlock *HeaderBB = BasicBlock::Create(Context, (string)"CUDA.LoopHeader." + dimension, F);
	BasicBlock *BodyBB = BasicBlock::Create(Context, (string)"CUDA.LoopBody." + dimension, F);
	
	AfterBB = BasicBlock::Create(Context, (string)"CUDA.AfterLoop." + dimension, F); //!!!!!!!!!!!!!
	Builder->CreateBr(HeaderBB);
	DT->addNewBlock(HeaderBB, PreheaderBB);

	Builder->SetInsertPoint(HeaderBB);

	// Use the type of upper and lower bound.
	assert(LB->getType() == UB->getType()
	       && "Different types for upper and lower bound.");

	IntegerType *LoopIVType = dyn_cast<IntegerType>(UB->getType());
	assert(LoopIVType && "UB is not integer?");

	// IV
	IV = Builder->CreatePHI(LoopIVType, 2, (string)"CUDA.loopiv." + dimension);
	IV->addIncoming(LB, PreheaderBB);

	// IV increment.
	Value *StrideValue = ThreadStride;
	IncrementedIV = Builder->CreateAdd(IV, StrideValue, (string)"CUDA.next_loopiv." + dimension);

	// Exit condition.
	if (flags.AtLeastOnce) { // At least on iteration.
	    assert(false);
		UB = Builder->CreateAdd(UB, Builder->getInt64(1));
		Value *CMP = Builder->CreateICmpEQ(IV, UB);
		Builder->CreateCondBr(CMP, AfterBB, BodyBB);
	} else { // Maybe not executed at all.
	    ////////////////////////////////////////////////////////////////////////////////////////////////////
		// next iteration performed if in further condition is true                                       //
		// InductionVariable <= ThreadUppetBound && InductionVariable <= LoopUpperBound                   //
		////////////////////////////////////////////////////////////////////////////////////////////////////
		Value *ThreadCMP = Builder->CreateICmpSLE(IV, ThreadUB, (string)"isInThreadBounds." + dimension); //
		Value *LoopCMP = Builder->CreateICmpSLE(IV,UB,(string)"isInLoopBounds." + dimension);             //
		Value *ExitCond = Builder->CreateMul(ThreadCMP,LoopCMP,(string)"isInBounds." + dimension);        //
		Builder->CreateCondBr(ExitCond, BodyBB, AfterBB);                                                 //
		////////////////////////////////////////////////////////////////////////////////////////////////////
		
	}
	DT->addNewBlock(BodyBB, HeaderBB);
	DT->addNewBlock(AfterBB, HeaderBB);

	Builder->SetInsertPoint(BodyBB);
					   
					   }
};
