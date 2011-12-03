#include "CodeGeneration.h"

namespace kernelgen
{
int ClastStmtCodeGen::GoodNestedParallelLoops(const clast_stmt * stmt, int CurrentCount)
    {
		if (!stmt ||                                            // there is no statements on body
		    !CLAST_STMT_IS_A(stmt, stmt_for) ||                 // that statement is not clast_for
		    !DP->isParallelFor( (const clast_for *)stmt) ||     // that clast_for is not parallel
			stmt->next)                                         // that clast_for id not good neste in parent loop
			return CurrentCount;
		else
		{
			const clast_for *for_stmt = (const clast_for *)stmt;
			return GoodNestedParallelLoops(for_stmt->body, ++CurrentCount);
		}
	}
///////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////
void ClastStmtCodeGen::codegenForCUDA(const clast_for *f)
{

	Function *ParentFunction = Builder.GetInsertBlock()->getParent();
	
	Module *M = ParentFunction->getParent();
	LLVMContext &Context = ParentFunction->getContext();
	IntegerType *intPtrTy = TD->getIntPtrType(Context);

	char name[100];
    if(parallelLoops.size() == 1/*это первый параллельный цикл в скопе*/)
	{
		////////////////////////////////////////////////////
		// that is first call of that function            //
		// so, let's create some important blocks         //
		// first of all we need to load Grid parameters   //
		//    that performed in GridParamsBB BasicBlock   //
		// second, let's compute thread position in grid  //
		//    that performed if PosInGridBB BasicBlock    //
		////////////////////////////////////////////////////
		
        vector<Value *> GridParameters;
		// load Grid parameters by calling CUDA functions
		BasicBlock *GridParamsBB = BasicBlock::Create(Context, "CUDA.getGridParams", ParentFunction);
		// compute thread positin in grid
		BasicBlock *PosInGridBB = BasicBlock::Create(Context, "CUDA.getPositionInGrid", ParentFunction);
		
		BasicBlock *EntryBB = Builder.GetInsertBlock();
		Builder.CreateBr(GridParamsBB);
		DT->addNewBlock(GridParamsBB, EntryBB);
		
		//BasicBlock *EnteringBlock = S->getRegion().getEnteringBlock();
		//TerminatorInst *TI = EnteringBlock->getTerminator();
		//TI->replaceUsesOfWith(S->getRegion().getEntry(), GridParamsBB);
		//DT->addNewBlock(GridParamsBB, EnteringBlock); ///////////////////
		
		/////////////////////////////////////////////////////////
		// compute needed values separately for each dimension //
		/////////////////////////////////////////////////////////
		for(int dimension = 0; dimension < CudaFunctions.size() / 4 && 
		                        dimension< goodNestedParallelLoopsCount; dimension++)
		{
			Builder.SetInsertPoint(GridParamsBB);
			/////////////////////////////////////////////////////////////////////////////////////////
			// call CUDA functions                                                                 //
			// store values in vaector GridParameters                                              //          
			/////////////////////////////////////////////////////////////////////////////////////////
		    for(int GridParameter = 0; GridParameter < 4; GridParameter ++)                        //
		    {                                                                                      //
			    GridParameters.push_back(                                                          //
			      Builder.CreateCall( M->getFunction(CudaFunctions[dimension*4 + GridParameter]),  //    
				                      CudaInricics[CudaFunctions[dimension*4 + GridParameter]] )); //
		    }                                                                                      //
			/////////////////////////////////////////////////////////////////////////////////////////
   		    
			Builder.SetInsertPoint(PosInGridBB);
			
			//////////////////////////////////////////////////////
			// Grid Parameters for current dimension            //
			//////////////////////////////////////////////////////
			Value * threadId = GridParameters[dimension*4 + 0]; //
			Value * blockId =  GridParameters[dimension*4 + 1]; //
			Value * blockDim = GridParameters[dimension*4 + 2]; //
			Value * gridDim =  GridParameters[dimension*4 + 3]; //
			//////////////////////////////////////////////////////
			
			///////////////////////////////////////////////////////////////////////////
			// position in grid                                                      //
			// blockId.x * blockDim.x + threadId.x - index of thread for dimension X // 
		    ///////////////////////////////////////////////////////////////////////////
			Value* Position =                                                        //
			         Builder.CreateAdd(Builder.CreateMul(blockId,blockDim),          //
					 threadId, string("PositionInGrid.") + dimensions[dimension]);   //
			///////////////////////////////////////////////////////////////////////////
			
			//////////////////////////////////////////////////////////////////////
			//GridDim.x * blockDim.x - size of grid in threads for dimension  X //
			//////////////////////////////////////////////////////////////////////
			Value * Size =                                                      //
			         Builder.CreateMul(gridDim,blockDim,                        //
					      string("GridSize.") + dimensions[dimension]);         //
		    //////////////////////////////////////////////////////////////////////
		    
			//////////////////////////////////////
			// store value *                    //
			//////////////////////////////////////
			positionInGrid.push_back(Position); //
			GridSize.push_back(Size);           //
			BlockSize.push_back(blockDim);      //
			//////////////////////////////////////
		}
		    Builder.SetInsertPoint(GridParamsBB);
		    Builder.CreateBr(PosInGridBB);
			DT->addNewBlock(PosInGridBB, GridParamsBB); //////////////
			Builder.SetInsertPoint(PosInGridBB);			
			
	}
	
	////////////////////////////////////////////////////////////////////////////////
	// at that point GridParamsBB and PosInGridBB BasicBlocks have already crated //
	// needed Value* stored in positionInGrid,GridSize,BlockSize for each thread  //
	////////////////////////////////////////////////////////////////////////////////
	
	int dimension = goodNestedParallelLoopsCount - parallelLoops.size();
	const char * dimensionName = dimensions[dimension].c_str();
	BasicBlock *PrevBB = Builder.GetInsertBlock();
	
    ////////////////////////////////////////////////////////////
	// In CountBoundsBB BasicBlock for each dimension compute //
	//   CountOfIterations                                    //
	//   UpperBound                                           //
	//   LowerBound                                           //
    //   Stride                                               //
	// That Values are defferent between threads in runtime   //
	////////////////////////////////////////////////////////////
	BasicBlock * CountBoundsBB = BasicBlock::Create(Context, 
	     string("CUDA.CountBounds.") + dimensionName, ParentFunction);
	Builder.CreateBr(CountBoundsBB);
	DT->addNewBlock(CountBoundsBB, PrevBB); ///////////////////////////////////////
	
	Builder.SetInsertPoint(CountBoundsBB);
	
	//////////////////////////////////////////////////////////////////////////////////////
	//Lower and Upper Bounds of Loop                                                    //
	//////////////////////////////////////////////////////////////////////////////////////
	Value * lowerBound = ExpGen.codegen(f->LB,TD->getIntPtrType(Builder.getContext())); //
	Value * upperBound = ExpGen.codegen(f->UB,TD->getIntPtrType(Builder.getContext())); //
	//////////////////////////////////////////////////////////////////////////////////////
	
	//assert(число нитей по текущему измерению меньше числа итераций)
	
	//////////////////////////////////////////////////////////////////////////////////////////
	// Compute Iteration's count per thred                                                  //
	// Loop's IterationsCount = upperBound - lowerBound + 1                                 //
	// Iteration's Count per thread is ((IterationsCount-1) / GridSize + 1)                 //
	// (IterationsCount-1) = upperBound - loweBound                                         //
	//////////////////////////////////////////////////////////////////////////////////////////
	Value * IterationsCountMinusOne =                                                       //
	  Builder.CreateSub(upperBound,lowerBound, string("IterationsCount.") + dimensionName); //
	Value * Division =                                                                      //
	   Builder.CreateSDiv(IterationsCountMinusOne,GridSize[dimension]);                     //
	Value * IterationsPerThread =                                                           //
	   Builder.CreateAdd(Division,ConstantInt::get(Division->getType(), 1),                 // 
	           string("IterationsPerThread.") + dimensionName);                             //
	//////////////////////////////////////////////////////////////////////////////////////////
	
	///////////////////////////////////////////////////////////////////////////////////////
	// compute Thread's Upper Bound, Lower Bounds and Stride                             //
	// LowerBound = IterationsPerThread*ThreadPosition                                   //
	// UpperBound = LowerBound + Stride*IterationsPerThread                              //
	// Stride = BlockSize (to increase probability of coalescing transactions to memory) //
	///////////////////////////////////////////////////////////////////////////////////////
	Value * ThreadLowerBound =                                             //
	  Builder.CreateMul(positionInGrid[dimension], IterationsPerThread,    //
	       string("ThreadLoverBound.") + dimensionName);                   //
																		   //
	Value * Stride = BlockSize[dimension];                                 //
	Value * StrideMultIterPerThread =                                      //
	    Builder.CreateMul(Stride,IterationsPerThread);                     //
														                   //
	Value * ThreadUpperBound =                                             //
	   Builder.CreateAdd(ThreadLowerBound,StrideMultIterPerThread,         //
	        string("ThreadUpperBound.") + dimensionName);                  //
	/////////////////////////////////////////////////////////////////////////
	
	//////////////////////////////////////////////////////////////////////////////////
	// compute LoopStride - for general case, if OriginalLoopStride not equal to 1  //
	//////////////////////////////////////////////////////////////////////////////////
	IntegerType *LoopIVType = dyn_cast<IntegerType>(upperBound->getType());         //
	assert(LoopIVType && "UB is not integer?");                                     //
	                                                                                //
	APInt LoopStride = polly::APInt_from_MPZ(f->stride);                            //
    Value * StrideValue =                                                           //
	      Builder.CreateMul(Stride, ConstantInt::get(upperBound->getType(),         //
						               LoopStride.zext(LoopIVType->getBitWidth())), //
				string("ThreadStride.") + dimensionName );                          //
	//////////////////////////////////////////////////////////////////////////////////
	Builder.SetInsertPoint(CountBoundsBB);
	
	////////////////////////////////////////////////////////////////
	// Generate code for loop with computed bounds and stride     //
	// CountBoundsBB BasicBlock is a preheader of that loop       //
	///////////////////////////////////////////////////////////////////////////////////////////
    codegenForSequential(f, ThreadLowerBound, ThreadUpperBound, StrideValue, dimensionName); //
	///////////////////////////////////////////////////////////////////////////////////////////
} 

/// @brief Create a classical sequential loop.
void ClastStmtCodeGen::codegenForSequential(const clast_for *f, Value *lowerBound,
        Value *upperBound, Value * ThreadStride, const char * dimensionName)
{

	APInt LoopStride = polly::APInt_from_MPZ(f->stride);
	Function *F = Builder.GetInsertBlock()->getParent();
	
	PHINode *IV;
	Value *IncrementedIV;
	BasicBlock *AfterBB;

	Value *LoopLowerBound, *LoopUpperBound;
	
	assert(((lowerBound && upperBound) || (!lowerBound && !upperBound))
	       && "Either give both bounds or none");
	
	////////////////////////////////////////////////////////////////////////////
	//compute OriginalLoop's Lower and Upper Bounds                           //
	////////////////////////////////////////////////////////////////////////////
	if (lowerBound == 0 || upperBound == 0) {                                 //
		lowerBound = ExpGen.codegen(f->LB,                                    //
		                            TD->getIntPtrType(Builder.getContext())); //
		upperBound = ExpGen.codegen(f->UB,                                    //
		                            TD->getIntPtrType(Builder.getContext())); //
	}                                                                         //
	else {                                                                    //
		LoopLowerBound = ExpGen.codegen(f->LB,                                //
		                            TD->getIntPtrType(Builder.getContext())); //
		LoopUpperBound = ExpGen.codegen(f->UB,                                //
		                            TD->getIntPtrType(Builder.getContext())); //
	}                                                                         //
	////////////////////////////////////////////////////////////////////////////
	
	////////////////////////////////////////////////////////////////////////////
	// at that point Builder is at preheader of the loop                      //
	////////////////////////////////////////////////////////////////////////////
	// generate blocks for loop such as                                       //
	//     -LoopHeader - determ id Induction Var is in bounds.                //
	//	       if it is then goto LoopBody                                    //
	//         else goto AfterLoop                                            //
	//     -LoopBody - goto general body                                      //
	//     -AfterLoop                                                         //                                           
	////////////////////////////////////////////////////////////////////////////   
	if(ThreadStride)                                                          //
	   createLoopForCUDA(&Builder, LoopLowerBound, LoopUpperBound,            //
	             lowerBound, upperBound, ThreadStride, IV, AfterBB,           //
	                  IncrementedIV, DT, dimensionName );                     //
	else                                                                      //
		createLoop(&Builder, lowerBound, upperBound, LoopStride, IV, AfterBB, //
	              IncrementedIV, DT);                                         //
	////////////////////////////////////////////////////////////////////////////

	// Add loop iv to symbols.
    (*clastVars)[f->iterator] = IV;
    /////////////////////////////////////////////////////
	// generate general body of loop                   //
	// at that point Builder is at LoopBody BasicBlock //
	/////////////////////////////////////////////////////
    if (f->body)                                       //
		codegen(f->body);                              //
	////////////////////////////////////////////////////////////////////////
    // At that point Builder is at last block of body                     //
	//   if body is for_loop then Builder is at it's AfterLoop BasicBlock //
	//   else Builder is at statement's block                             //
	////////////////////////////////////////////////////////////////////////
	
	// Loop is finished, so remove its iv from the live symbols.
	clastVars->erase(f->iterator);
	
	BasicBlock *HeaderBB = *pred_begin(AfterBB);
	BasicBlock *LastBodyBB = Builder.GetInsertBlock();
	Builder.CreateBr(HeaderBB);
	//add incoming value from last block of body
	IV->addIncoming(IncrementedIV, LastBodyBB);
	
	// make AfterBB BasicBlock is the last BasicBlock of Loop
    AfterBB->moveAfter(&F->back());
	Builder.SetInsertPoint(AfterBB);
}

//////////////////////////////////////////////////////////////////

void ClastStmtCodeGen::addParameters(const CloogNames *names)
{
	SCEVExpander Rewriter(SE, "polly");

	// Create an instruction that specifies the location where the parameters
	// are expanded.
	CastInst::CreateIntegerCast(ConstantInt::getTrue(Builder.getContext()),
	                            Builder.getInt16Ty(), false, "insertInst",
	                            Builder.GetInsertBlock());

	int i = 0;
	for (polly::Scop::param_iterator PI = S->param_begin(), PE = S->param_end();
	     PI != PE; ++PI) {
		assert(i < names->nb_parameters && "Not enough parameter names");

		const SCEV *Param = *PI;
		Type *Ty = Param->getType();

		Instruction *insertLocation = --(Builder.GetInsertBlock()->end());
		Value *V = Rewriter.expandCodeFor(Param, Ty, insertLocation);
		(*clastVars)[names->parameters[i]] = V;

		++i;
	}
}

bool ClastStmtCodeGen::isInnermostLoop(const clast_for *f)
{
	const clast_stmt *stmt = f->body;

	while (stmt) {
		if (!CLAST_STMT_IS_A(stmt, stmt_user))
			return false;

		stmt = stmt->next;
	}

	return true;
}
};

/* 
    BasicBlock *sheduleBB =  BasicBlock::Create(Context, "CUDA.getStaticSchedule", ParentFunction);
	Builder.SetInsertPoint(sheduleBB);
    //рассчитать верхнюю и нижнюю границы, а так же stride
	//если по измерению нет шаред памяти, то stride = 1
	//если по измерению есть шаред память, то stride может быть не равен 1
	int goodNestedParallelLoopsCount; всего циклов будет обработано
	int MaxDimensionsCount;  максимальное количество циклов, которое может быть обработано
	assert(количество функций соответствует числу измерений)
	parallelLoops.size() номер обрабатываемого в данный момент цикла
	
	
	x 0 самый вложенный цикл
	y 1 если есть два тесновложенных параллельныйх цикла, то это самый внешний цикл
	z 2 если есть три тесновложенных цикла, то это самый внешний цик
	
	До вызова генерации кода
	    раcсчитываем рекурсивно goodNestedParallelLoopsCount
		если goodNestedParallelLoopsCount>MaxDimensionsCount, то
	        goodNestedParallelLoopsCount = MaxDimensionsCount
	
	Если parallelLoops.size > 3, то генерируем последовательный код
	
	для случая трёх циклов goodNestedParallelLoopsCount = 3
	    goodNestedParallelLoopsCount-parallelLoops.size == 2 при первой итерации, ось z
	    goodNestedParallelLoopsCount-parallelLoops.size == 1 при второй итерации, ось y
		goodNestedParallelLoopsCount-parallelLoops.size == 0 при второй итерации, ось x
	
	для случая двух циклов goodNestedParallelLoopsCount = 2
	    goodNestedParallelLoopsCount-parallelLoops.size == 1 при первой итерации, ось y
	    goodNestedParallelLoopsCount-parallelLoops.size == 0 при второй итерации, ось z

	для случая одного цикла goodNestedParallelLoopsCount = 1
	    goodNestedParallelLoopsCount-parallelLoops.size == 0 при первой итерации, ось x

	blockId.x * blockDim.x + threadId.x - индекс нити по оси X
	GridSize = GridDim.x * blockDim.x - всего нитей по оси X
	BlockDim.x - stride по оси X
	
	lowerBound = ExpGen.codegen(f->LB,TD->getIntPtrType(Builder.getContext())); 
	upperBound = ExpGen.codegen(f->UB,TD->getIntPtrType(Builder.getContext()));
	
	//assert(число нитей по текущему измерению меньше числа итераций)
	
	IterationsCount = UpperBound-LowerBound - число итераций по текущей оси
	(IterationsCount-1) / GridSize + 1 = 
	(IterationsCount + (GridSize-1)) / GridSize =           stride
	*/
