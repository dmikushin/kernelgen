#include "Dependences.h"
#include "llvm/ADT/DenseMap.h"
#include <list>
#include <vector>
#include "llvm/LLVMContext.h"

class clast_for;
class clast_user_stmt;
class clast_stmt;
class clast_expr;
class clast_name;
class clast_term;
class clast_binary;
class clast_reduction;

namespace llvm
{
	class CallInst;
	class Function;
	class Type;
}
namespace kernelgen
{
	class KernelCloogInfo;
	class KernelScopInfo;
	class SCEVTreeNode;


	class ClastExpCalculator
	{
		std::map<const char*, int64_t> &clastNames;
		typedef std::map<const char*, int64_t>::iterator NameIterator;
	public:
		ClastExpCalculator(std::map<const char*, int64_t> &clastNames)
			: clastNames(clastNames) {}
		int64_t calculate(const clast_expr *e);
		int64_t calculate(const clast_name *e);
		int64_t calculate(const clast_term *e);
		int64_t calculate(const clast_binary *e);
		int64_t calculate(const clast_reduction *r);
	};

	class CalculateParallelLoops : public KernelScopPass
	{
	public:

		typedef llvm::DenseMap<const clast_for *, SCEVTreeNode *> LoopsMapType;
		LoopsMapType loopsToFunctions;

		KernelCloogInfo * KCI;
		KernelScopInfo * KSI;
		Dependences *D;

		LLVMContext *context;
		Type *int64Ty;

		static char ID;
		std::vector<unsigned> *memForLoopsSizes;
		bool *parallelLoopExists;
		
		std::list<const clast_for *> parallelLoops;
		std::map<const char*, int64_t> clastNames;
		ClastExpCalculator calculator;
		
		CalculateParallelLoops(std::vector<unsigned> * memForLoopsSizes = NULL, bool *parallelLoopExists = NULL)
			: KernelScopPass(ID),memForLoopsSizes(memForLoopsSizes) ,parallelLoopExists(parallelLoopExists), calculator(clastNames) {}

		void getAnalysisUsage(AnalysisUsage &AU) const;
		virtual bool runOnScop(Scop &S);
		void findFunctionsForLoops();

		void processStmtUser(const clast_user_stmt *stmt, std::list<const clast_for *> &currentLoopsNest);
		void findStmtUser(const clast_stmt *stmt, std::list<const clast_for *> &currentLoopsNest);

		void calculateParallelLoops();
		void calculateSizes(std::list<const clast_for *> loops, int nest);
		const clast_for * oneGoodParalelLoopExistOnThatLevel ( const clast_stmt * startOfLevel );
		void makeMetadataForFunctionAndCallInst(unsigned startDepth, unsigned loopCount, Function *f, CallInst *callInst);
	};
    Pass *createCalculateParallelLoopsPass(std::vector<unsigned> * memForLoopsSizes = NULL, bool *parallelLoopExists = NULL);
}

namespace llvm
{
	class PassRegistry;
	void initializeCalculateParallelLoopsPass(llvm::PassRegistry&);
}
