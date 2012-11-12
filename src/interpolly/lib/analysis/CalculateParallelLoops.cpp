#include "CalculateParallelLoops.h"
#include "KernelCloog.h"
#include "KernelScopInfo.h"
#include "GICHelper.h"

using namespace llvm;

namespace kernelgen
{

	int64_t ClastExpCalculator::calculate(const clast_expr *e)
	{
		switch(e->type) {
			case clast_expr_name:
				return calculate((const clast_name *)e);
			case clast_expr_term:
				return calculate((const clast_term *)e);
			case clast_expr_bin:
				return calculate((const clast_binary *)e);
			case clast_expr_red:
				return calculate((const clast_reduction *)e);
			default:
				assert(false && "Unknown clast expression!");
		}
	}
	int64_t ClastExpCalculator::calculate(const clast_name *e)
	{
		NameIterator I = clastNames.find(e->name);

		if (I != clastNames.end())
			return (I->second);
		else
			assert(false && "Clast name not found!");
	}
	int64_t ClastExpCalculator::calculate(const clast_term *e)
	{
		int64_t a = APInt_from_MPZ(e->val).getSExtValue();
		if (e->var)
			return a * calculate(e->var);
		else
			return a;
	}
	int64_t ClastExpCalculator::calculate(const clast_binary *e)
	{
		int64_t a = calculate(e->LHS);
		int64_t b = APInt_from_MPZ(e->RHS).getSExtValue();

		switch (e->type) {
			case clast_bin_fdiv:
				// floord(n,d) ((n < 0) ? (n - d + 1) : n) / d
				return ((a < 0) ? (a - b + 1) : a) / b;
			case clast_bin_cdiv:
				// ceild(n,d) ((n < 0) ? n : (n + d - 1)) / d
				return ((a < 0) ? a : (a + b - 1)) / b;
			case clast_bin_mod:
				return a % b;
			case clast_bin_div:
				return a / b;
			default:
				assert(false && "Unknown clast binary expression type");
		};
	}
	int64_t ClastExpCalculator::calculate(const clast_reduction *r)
	{

		int64_t old = calculate(r->elts[0]);

		for (int i=1; i < r->n; ++i) {
			int64_t exprValue = calculate(r->elts[i]);

			switch (r->type) {
				case clast_red_min:
					old = ((exprValue < old)? exprValue : old);
					break;
				case clast_red_max:
					old = ((exprValue > old)? exprValue : old);
					break;
				case clast_red_sum:
					old += exprValue;
					break;
				default:
					assert(false && "Clast unknown reduction expression type");
			}

		}
		return old;
	}
	void CalculateParallelLoops::calculateSizes(std::list<const clast_for *> loops, int nest)
	{
		if(!loops.empty()) {

			const clast_for *loop = loops.front();
			loops.pop_front();

			int64_t upperBound = calculator.calculate(loop->UB);
			int64_t lowerBound = calculator.calculate(loop->LB);

			int64_t iterations = upperBound - lowerBound + 1;
			iterations = ((iterations > 0)?iterations:0);
			iterations = iterations /  APInt_from_MPZ(loop->stride).getSExtValue();

			(*memForLoopsSizes)[nest] += iterations;

			clastNames[loop->iterator] = lowerBound;
			calculateSizes(loops, nest + 1);
			clastNames[loop->iterator] = upperBound;
			calculateSizes(loops, nest + 1);

			clastNames.erase(loop->iterator);
		}
	}

	bool CalculateParallelLoops::runOnScop ( Scop &S )
	{
		KCI = &getAnalysis<KernelCloogInfo>();
		D = &getAnalysis<Dependences>();

		context = &(S.rootFunction->getParent()->getContext());
		int64Ty = Type::getInt64Ty( *context );

		findFunctionsForLoops();
		calculateParallelLoops();

		memForLoopsSizes->assign(parallelLoops.size(),0);
		calculateSizes(parallelLoops,0);

		int pow = 1;
		for(unsigned i = 0; i < parallelLoops.size(); i++, pow*=2)
			(*memForLoopsSizes)[i] /= pow ;

		if(parallelLoopExists) {
			*parallelLoopExists = !parallelLoops.empty();
			if(!*parallelLoopExists) {
				LoopsMapType::iterator loopsIterator = loopsToFunctions.begin();
				while(loopsIterator != loopsToFunctions.end() && !*parallelLoopExists) {
					if(D->isParallelFor(loopsIterator->first))
						*parallelLoopExists = true;
					loopsIterator++;
				}
			}
		}
		return false;
	}

	char CalculateParallelLoops::ID = 0;
	void CalculateParallelLoops::getAnalysisUsage ( AnalysisUsage &AU ) const
	{
		KernelScopPass::getAnalysisUsage ( AU );
		AU.addRequiredTransitive<Dependences>();
		AU.addRequiredTransitive<KernelCloogInfo>();
		AU.setPreservesAll();
	}

	void CalculateParallelLoops::processStmtUser ( const clast_user_stmt *stmt, std::list<const clast_for *> &currentLoopsNest )
	{

		ScopStmt *Statement = ( ScopStmt * ) stmt->statement->usr;
		assert ( Statement->getNumIterators() == currentLoopsNest.size() );
		SCEVTreeNode *SCEVTreeLeaf = Statement->SCEVTreeLeaf;
		int depth = 0;
		for ( std::list<const clast_for *>::iterator iter = currentLoopsNest.begin(), iterEnd = currentLoopsNest.end();
		      iter!=iterEnd; iter++ ) {
			const clast_for *Loop = *iter;
			const Function *funcForLoop = Statement->getInductionVariableForDimension ( depth )->getParent()->getParent();
			SCEVTreeNode * nodeForFunc = SCEVTreeLeaf;
			while ( nodeForFunc->f != funcForLoop )
				nodeForFunc=nodeForFunc->parentNode;
			LoopsMapType::iterator iterForLoop = loopsToFunctions.find ( Loop );
			if ( iterForLoop != loopsToFunctions.end() )
				assert ( iterForLoop->second == nodeForFunc );
			else
				loopsToFunctions[Loop] = nodeForFunc;

			depth++;
		}

	}
	void CalculateParallelLoops::findStmtUser ( const clast_stmt *stmt, std::list<const clast_for *> &currentLoopsNest )
	{
		if	    ( CLAST_STMT_IS_A ( stmt, stmt_root ) )
			assert ( false && "No second root statement expected" );
		else if ( CLAST_STMT_IS_A ( stmt, stmt_user ) )
			processStmtUser ( ( const clast_user_stmt * ) stmt, currentLoopsNest );
		else if ( CLAST_STMT_IS_A ( stmt, stmt_block ) )
			findStmtUser ( ( ( const clast_block * ) stmt )->body,currentLoopsNest );
		else if ( CLAST_STMT_IS_A ( stmt, stmt_for ) ) {
			currentLoopsNest.push_back ( ( const clast_for * ) stmt );
			findStmtUser ( ( ( const clast_for * ) stmt )->body,currentLoopsNest );
			currentLoopsNest.pop_back();
		} else  if ( CLAST_STMT_IS_A ( stmt, stmt_guard ) )
			findStmtUser ( ( ( const clast_guard * ) stmt )->then,currentLoopsNest );
		else if ( CLAST_STMT_IS_A ( stmt, stmt_ass ) )
			assert ( false ); //findStmtUser((const clast_assignment *)stmt, currentLoopsNest);

		if ( stmt->next )
			findStmtUser ( stmt->next,currentLoopsNest );
	}

	const clast_for * CalculateParallelLoops::oneGoodParalelLoopExistOnThatLevel ( const clast_stmt * startOfLevel )
	{
		const clast_stmt * currentStmt = startOfLevel;
		bool allGood = true;
		const clast_for * loop = NULL;
		std::list<const clast_stmt *> notProcessed;
		while( currentStmt && allGood) {
			if(CLAST_STMT_IS_A(currentStmt, stmt_user)) {
				allGood = false;
				continue;
				currentStmt = currentStmt->next;
			} else if( CLAST_STMT_IS_A(currentStmt, stmt_for)) {
				if ( loop || !D->isParallelFor((const clast_for *)currentStmt)  ) {
					allGood = false;
					continue;
				}
				loop = ( const clast_for * ) currentStmt;
				currentStmt = currentStmt->next;
			} else if(CLAST_STMT_IS_A(currentStmt, stmt_guard)) {
				notProcessed.push_back(currentStmt->next);
				currentStmt = ((const clast_guard *)currentStmt)->then;
			} else if(CLAST_STMT_IS_A(currentStmt, stmt_block)) {
				notProcessed.push_back(currentStmt->next);
				currentStmt = ((const clast_block *)currentStmt)->body;
			} else	if(CLAST_STMT_IS_A(currentStmt, stmt_ass)) {
				assert(false);
				currentStmt = currentStmt->next;
			}

			while(!currentStmt && !notProcessed.empty()) {
				currentStmt = notProcessed.back();
				notProcessed.pop_back();
			}
		}

		if ( allGood )
			return loop;
		else return NULL;

	}

	void CalculateParallelLoops::makeMetadataForFunctionAndCallInst(
	    unsigned startDepth, unsigned loopCount, Function *f, CallInst *callInst)
	{
		unsigned MDKindID = f->getParent()->getMDKindID("depthOfCallInst");

		Value * pair[2] =  { ConstantInt::get(int64Ty,startDepth),ConstantInt::get(int64Ty,loopCount) };
		MDNode *mdNode = MDNode::get( *context, pair  );
		if(callInst)
			callInst->setMetadata(MDKindID, mdNode);

		NamedMDNode *nodeForFunction = f->getParent()->getOrInsertNamedMetadata(f->getName());
		bool isPairAlreadyInOperands = false;
		for(unsigned i = 0; i < nodeForFunction->getNumOperands() && !isPairAlreadyInOperands; i++) {
			uint64_t existingStartDepth = cast<ConstantInt>(nodeForFunction->getOperand(i)->getOperand(0))->getZExtValue();
			uint64_t existingLoopCount = cast<ConstantInt>(nodeForFunction->getOperand(i)->getOperand(1))->getZExtValue();
			if(existingStartDepth == startDepth && loopCount == existingLoopCount)
				isPairAlreadyInOperands = true;
		}
		if(!isPairAlreadyInOperands)
			nodeForFunction->addOperand(mdNode);
	}

	void CalculateParallelLoops::calculateParallelLoops()
	{
		const clast_root *root = KCI->getClast();

		if(((const clast_stmt *)root)->next ) {

			SCEVTreeNode *previousNode = NULL, *currentNode = NULL;

			int loopCount = 0;
			int depth = 0;
			int startDepth = 0;

			const clast_stmt * startOfLevel = ((const clast_stmt *)root)->next;
			while (const clast_for * loopAtLevel = oneGoodParalelLoopExistOnThatLevel(startOfLevel) ) {

				currentNode = loopsToFunctions[loopAtLevel];
				parallelLoops.push_back(loopAtLevel);
				if(depth == 0)
					previousNode = currentNode;

				if(currentNode != previousNode) {
					makeMetadataForFunctionAndCallInst(startDepth, loopCount, previousNode->f, previousNode->invokeCallInst);

					previousNode = currentNode;
					startDepth = depth;
					loopCount = 0;
				}

				loopCount++;
				depth++;
				startOfLevel = loopAtLevel->body;

				if(depth == 3)
					break;

			}
			if(currentNode) {

				makeMetadataForFunctionAndCallInst(startDepth, loopCount, currentNode->f, currentNode->invokeCallInst );

				NamedMDNode *numberOfLoopsNode = currentNode->f->getParent()->getOrInsertNamedMetadata("NumberOfParallelLoops");
				MDNode *mdNode = MDNode::get( *context, ConstantInt::get(int64Ty,depth) );
				numberOfLoopsNode->addOperand(mdNode);
			}
		}
	}

	void CalculateParallelLoops::findFunctionsForLoops()
	{
		const clast_root *root = KCI->getClast();
		std::list<const clast_for *> currentLoopsNest;
		if ( ( ( const clast_stmt* ) root )->next )
			findStmtUser ( ( ( const clast_stmt* ) root )->next, currentLoopsNest );
		assert ( currentLoopsNest.size() == 0 );
	}
	Pass *createCalculateParallelLoopsPass(std::vector<unsigned> * memForLoopsSizes, bool *parallelLoopExists)
	{
		return new CalculateParallelLoops(memForLoopsSizes, parallelLoopExists);
	}
}

using namespace kernelgen;

INITIALIZE_PASS_BEGIN ( CalculateParallelLoops, "polly-dependences",
                        "Polly - Calculate dependences", false, false )
INITIALIZE_PASS_DEPENDENCY ( KernelScopInfo )
INITIALIZE_PASS_DEPENDENCY ( KernelCloogInfo )
INITIALIZE_PASS_DEPENDENCY ( Dependences )
INITIALIZE_PASS_END ( CalculateParallelLoops, "polly-dependences",
                      "Polly - Calculate dependences", false, false )
