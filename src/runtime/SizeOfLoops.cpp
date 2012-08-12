#include "polly/Cloog.h"
#include "polly/Dependences.h"
#include "polly/ScopInfo.h"
#include "polly/ScopPass.h"
#include "polly/Support/GICHelper.h"
#include "llvm/Support/raw_os_ostream.h"
#include "cloog/cloog.h"
#include "runtime.h"
#include <stdio.h>

#include <iostream>
#include <fstream>
#include <vector>

using namespace kernelgen;
using namespace polly;
using namespace std;

typedef std::map<const char*, int64_t>::iterator NameIterator;

class ClastExpCalculator
{
	std::map<const char*, int64_t> *availableClastNames;
public:
	void setClastNames(std::map<const char*, int64_t> *clastNames) {
		availableClastNames = clastNames;
	}

	inline int64_t calculate(const clast_expr *e) {
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
	inline int64_t calculate(const clast_name *e) {
		assert(availableClastNames && "ClastNames not set!");
		NameIterator I = availableClastNames->find(e->name);

		if (I != availableClastNames->end())
			return (I->second);
		else
			assert(false && "Clast name not found!");
	}
	inline int64_t calculate(const clast_term *e) {
		int64_t a = APInt_from_MPZ(e->val).getSExtValue();
		if (e->var)
			return a * calculate(e->var);
		else
			return a;
	}
	inline int64_t calculate(const clast_binary *e) {
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
	inline int64_t calculate(const clast_reduction *r) {

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
};

class ClastStmtInterpreter
{
private:
	std::map<const char*, int64_t> *availableClastNames;
	ClastExpCalculator clastExpCalculator;
	int currentNestingLevel;//starts from one
	int maxNestingLevel;
public:
	int64_t numberOfIterations[3];

	ClastStmtInterpreter(int _maxNestingLevel=3)
		:maxNestingLevel(_maxNestingLevel) {}

	inline void interpret(const clast_for *e) {

		assert( e->stride && "zero stride not supported yet");
		currentNestingLevel++;  ///////////////////
		if(currentNestingLevel <=maxNestingLevel) {

			int64_t upperBound = clastExpCalculator.calculate(e->UB);
			int64_t lowerBound = clastExpCalculator.calculate(e->LB);

			int64_t iterations = upperBound - lowerBound + 1;
			iterations = ((iterations > 0)?iterations:0);
			iterations = iterations /  APInt_from_MPZ(e->stride).getSExtValue();
			//if(iterations) {
			(*availableClastNames)[e->iterator] = lowerBound;
			interpret(e->body);
			(*availableClastNames)[e->iterator] = upperBound;
			interpret(e->body);
			availableClastNames->erase(e->iterator);
			//}
			numberOfIterations[currentNestingLevel-1] += iterations;
		}
		currentNestingLevel--;  //////////////////
	}
	inline void interpret(const clast_stmt *stmt) {
		if	    (CLAST_STMT_IS_A(stmt, stmt_root))
			assert(false && "No second root statement expected");
		else if (CLAST_STMT_IS_A(stmt, stmt_ass))
			interpret((const clast_assignment *)stmt);
		else if (CLAST_STMT_IS_A(stmt, stmt_user))
			interpret((const clast_user_stmt *)stmt);
		else if (CLAST_STMT_IS_A(stmt, stmt_block))
			interpret((const clast_block *)stmt);
		else if (CLAST_STMT_IS_A(stmt, stmt_for))
			interpret((const clast_for *)stmt);
		else if (CLAST_STMT_IS_A(stmt, stmt_guard))
			interpret((const clast_guard *)stmt);

		if (stmt->next)
			interpret(stmt->next);
	}
	inline void interpret(const clast_root *r) {

		availableClastNames = new std::map<const char*, int64_t>();
		clastExpCalculator.setClastNames(availableClastNames);
		
		for(int i = 0; i < maxNestingLevel; i++)
			numberOfIterations[i] = 0;
		for(int i = maxNestingLevel; i < 3; i++)
			numberOfIterations[i] = -1;
		
		currentNestingLevel = 0;

		const clast_stmt *stmt = (const clast_stmt*) r;
		if (stmt->next)
			interpret(stmt->next);

		int pow = 1;
		for(int i = 0; i < maxNestingLevel; i++, pow*=2)
			numberOfIterations[i] /= pow ;
		
		delete availableClastNames;
		return;
	}
	inline void interpret(const clast_assignment * e) {
		(*availableClastNames)[e->LHS] = clastExpCalculator.calculate(e->RHS);
	}
	inline void interpret(const clast_block * e) {
		if(e->body) {
			std::map<const char*, int64_t> tmpClastNames(availableClastNames->begin(),availableClastNames->end());

			interpret(e->body);

			if(tmpClastNames.size() != availableClastNames->size()) {
				vector<const char *> unavailableNames;
				for(NameIterator name = availableClastNames->begin(), end = availableClastNames->end();
				    name !=  end; name++)
					if( tmpClastNames.find(name->first) != end )
						unavailableNames.push_back(name->first);

				for(vector<const char *>::iterator unavailableName=unavailableNames.begin(), end=unavailableNames.end();
				    unavailableName!=end;  unavailableName++)
					availableClastNames->erase(*unavailableName);
			}
		}
	}
	inline void interpret(const clast_user_stmt *e) {
		return;
	}
	inline void interpret(const clast_guard *e) {
		interpret(e->then);
		return;
	}
};

class SizeOfLoops : public ScopPass
{
	vector<Size3> * sizeOfLoops;
	bool *isThereAtLeastOneParallelLoop;
	Dependences *DP;
public:
	static char ID;
	void printCloogAST(CloogInfo &C) {
		/*CloogState *state = cloog_state_malloc();;
		CloogOptions * options = cloog_options_malloc(state);
		FILE * f = fopen("cloog.txt","w");
		clast_pprint(f, (clast_stmt*) root, 0, options);
		fclose(f);*/
		cout << "<------------------- Cloog AST of Scop ------------------->" << endl;
		raw_os_ostream OS1(std::cout);
		C.pprint(OS1);
		OS1.flush();
		cout << "<--------------------------------------------------------->" << endl;
	}
	void printSizeOfLoops(Size3 &size3, int numberOfLoops) {
		if (verbose & KERNELGEN_VERBOSE_POLLYGEN)
		{
			outs().changeColor(raw_ostream::GREEN);
			cout << "\n    Number of good nested parallel loops: " << numberOfLoops << endl;
			if(numberOfLoops) {
				cout << "    Average size of loops: " << size3.x;
				if(numberOfLoops >=2) cout << " " << size3.y;
				if(numberOfLoops >=3) cout << " " << size3.z;
				cout << endl;
			}
			outs().resetColor();
		}
	}
	void setMemoryForSizes(vector<Size3> *memForSizes) {
		sizeOfLoops = memForSizes;
		return;
	}
	SizeOfLoops(vector<Size3> *memForSizes=0,bool *_isThereAtLeastOneParallelLoop = NULL)
		:sizeOfLoops(memForSizes), ScopPass(ID), isThereAtLeastOneParallelLoop(_isThereAtLeastOneParallelLoop) {}
	bool runOnScop(Scop &scop);
	int GoodNestedParallelLoops(const clast_stmt * stmt, int CurrentCount);
	int GoodNestedParallelLoops(const clast_stmt * stmt);
	Size3 retrieveSize3FromCloogLoopAST(const clast_root * CloogAST, int numLoops);
	void getAnalysisUsage(AnalysisUsage &AU) const {
		ScopPass::getAnalysisUsage(AU);
		AU.addRequired<CloogInfo>();
		AU.addRequired<Dependences>();
        AU.setPreservesAll();
	}
	void findParallelLoop(const clast_stmt * stmt);
};
bool isaGoodListOfStatements(const clast_stmt * stmt, const clast_for * &nested_for, bool & user_or_assignment)
{
	// возвращает true, если
	//     в теле нет вложенных циклов
	//     в теле есть всего один вложенный цикл и нет assignment или user_stmt, не вложенных в него
	// через nested_for возвращает указатель на вложенный цикл или NULL если его нет

	if(!stmt) return true;
	bool good = true;

	if(CLAST_STMT_IS_A(stmt, stmt_user) || CLAST_STMT_IS_A(stmt, stmt_ass))
		if(nested_for) return false; // уже где-то есть цикл ---------------
		else user_or_assignment = true; // Нашли user/assignment !!!!!!!!!!!
	if(CLAST_STMT_IS_A(stmt, stmt_guard))
		good = isaGoodListOfStatements(((const clast_guard *)stmt)->then, nested_for, user_or_assignment);
	if(CLAST_STMT_IS_A(stmt, stmt_block))
		good = isaGoodListOfStatements(((const clast_block *)stmt)->body, nested_for, user_or_assignment);
	if(CLAST_STMT_IS_A(stmt, stmt_for))
		if(nested_for || user_or_assignment) return false; // уже где-то есть user/assignment или цикл --------
		else nested_for = (const clast_for *)stmt; //нашли for !!!!!!!!

	return good && isaGoodListOfStatements( stmt->next, nested_for,user_or_assignment);
}
int SizeOfLoops::GoodNestedParallelLoops(const clast_stmt * stmt)
{
	
	int goodLoops = 0;
	while(goodLoops < 3) {
		const clast_for * nested_for = NULL;
		bool user_or_assignment = false;
		if(isaGoodListOfStatements(stmt, nested_for, user_or_assignment)) {
			if(nested_for) // если есть цикл
				if(DP->isParallelFor(nested_for)) { // и он параллельный
					goodLoops++; // то увеличиваем счётчик
					stmt = nested_for->body; //и проверяем тело обнаруженного цикла
				} else break; //если встретился не параллельный цикл
			else break; // если нет больше циклов
		} else break; // если список не удовлетворяет требованиям
	}
	return goodLoops;
}
int SizeOfLoops::GoodNestedParallelLoops(const clast_stmt * stmt, int CurrentCount)
{
	Dependences *DP = &getAnalysis<Dependences>();
	if (!stmt ||                                            // there is no statements on body    //
	    !CLAST_STMT_IS_A(stmt, stmt_for) ||                 // that statement is not clast_for   //
	    !DP->isParallelFor( (const clast_for *)stmt) ||     // that clast_for is not parallel    //
	    stmt->next)                                         // that clast_for is not good nested //
		return CurrentCount;                                                                     //
	else {                                                                                       //
		const clast_for *for_stmt = (const clast_for *)stmt;                                     //
		return GoodNestedParallelLoops(for_stmt->body, ++CurrentCount);                          //
	}                                                                                            //
}
Size3 SizeOfLoops::retrieveSize3FromCloogLoopAST(const clast_root * CloogAST, int numLoops)
{
	ClastStmtInterpreter interpreter(numLoops);
	interpreter.interpret(CloogAST);
	return Size3(interpreter.numberOfIterations);
}
void SizeOfLoops::findParallelLoop(const clast_stmt * stmt)
{
	while(stmt!=NULL && !*isThereAtLeastOneParallelLoop) {
		if(CLAST_STMT_IS_A(stmt, stmt_for)) {
		    const clast_for *for_loop = (const clast_for *)stmt;
			if(DP->isParallelFor(for_loop)) {
				*isThereAtLeastOneParallelLoop = true;
				return;
			} else
				findParallelLoop(for_loop->body);
		}
		stmt=stmt->next;
	}
}
bool SizeOfLoops::runOnScop(Scop &scop)
{
	assert(sizeOfLoops && "memory for vector<Size3> not set!");
	assert(sizeOfLoops->size() <= 1 &&
	       "only one scop allowed!");

	CloogInfo &C = getAnalysis<CloogInfo>();
	DP = &getAnalysis<Dependences>();
	const clast_root *root = C.getClast();
		
	//if there are some not substituted parameters then we can not compute size of loops 
	assert(scop.getNumParams() == 0 &&
	       "FIXME: "
	       "After Constant Substitution number of scop's global parameters must be zero");
	///   Those parameters are scalar integer values, which are constant during
	///   execution.
	assert(root->names->nb_parameters == 0 &&
	       "FIXME: "
	       "After Constant Substitution number of cloog's parameters must be zero");

	const clast_stmt *stmt = (const clast_stmt*) root;

	int goodLoopsCount = GoodNestedParallelLoops(stmt->next);
	goodLoopsCount = goodLoopsCount > 3 ? 3 : goodLoopsCount;

	if(goodLoopsCount != 0)
		sizeOfLoops->push_back(
		    retrieveSize3FromCloogLoopAST(root, goodLoopsCount));
	else
		sizeOfLoops->push_back(Size3());
		
	if(isThereAtLeastOneParallelLoop) {	
	   *isThereAtLeastOneParallelLoop = false;
	   findParallelLoop(stmt->next);
    }
	printSizeOfLoops( (*sizeOfLoops)[0], goodLoopsCount);
    //printCloogAST(C);
        if (verbose & KERNELGEN_VERBOSE_POLLYGEN)
		outs() << "\n<------------------------------ Scop: end ----------------------------------->\n";
	return false;
}

char SizeOfLoops::ID = 0;
static RegisterPass<SizeOfLoops>
Z("Size3-scop", "Compute Size3 structure for scops");

Pass* createSizeOfLoopsPass(vector<Size3> *memForSize3 = NULL, bool *isThereAtLeastOneParallelLoop = NULL)
{
	return new SizeOfLoops(memForSize3 , isThereAtLeastOneParallelLoop);
}
