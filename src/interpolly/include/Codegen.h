#include "KernelCloog.h"
#include "KernelScopInfo.h"
#include "llvm/Module.h"
#include "llvm/Function.h"
#include "llvm/ADT/Twine.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/ValueMap.h"
#include "llvm/Support/IRBuilder.h"
#include "llvm/Analysis/Verifier.h"
#include "llvm/Analysis/ScalarEvolution.h"
#include "llvm/Analysis/ScalarEvolutionExpander.h"
#include "llvm/Analysis/ScalarEvolutionExpressions.h"
#include "GICHelper.h"
#include "LoopGenerators.h"

#include "BlockGenerator.h"

#include <string>
using namespace std;



#ifndef KERNEL_CODEGENERATION
#define KERNEL_CODEGENERATION
namespace kernelgen
{
extern bool CodeGenForCuda;
class KernelCodeGenerator;
typedef DenseMap<const char*, Value*> CharMapT;
typedef DenseMap<const Value*, Value*> ValueMapT;


extern std::vector<string> dimensions;
extern std::vector<const char*> CudaFunctions;
extern DenseMap<const char*,const char *> CudaInricics;

enum VectorizerChoice {
    VECTORIZER_NONE,
    VECTORIZER_POLLY,
    VECTORIZER_UNROLL_ONLY,
    VECTORIZER_FIRST_NEED_GROUPED_UNROLL = VECTORIZER_UNROLL_ONLY,
    VECTORIZER_BB
};

extern VectorizerChoice PollyVectorizerChoice;

class ClastExpCodeGen
{
	IRBuilder<> &Builder;
	const CharMapT &IVS;

	Value *codegen(const clast_name *e, Type *Ty);
	Value *codegen(const clast_term *e, Type *Ty);
	Value *codegen(const clast_binary *e, Type *Ty);
	Value *codegen(const clast_reduction *r, Type *Ty);
public:

	// A generator for clast expressions.
	//
	// @param B The IRBuilder that defines where the code to calculate the
	//          clast expressions should be inserted.
	// @param IVMAP A Map that translates strings describing the induction
	//              variables to the Values* that represent these variables
	//              on the LLVM side.
	ClastExpCodeGen(IRBuilder<> &B, CharMapT &IVMap);

	// Generates code to calculate a given clast expression.
	//
	// @param e The expression to calculate.
	// @return The Value that holds the result.
	Value *codegen(const clast_expr *e, Type *Ty);
};

class ClastStmtCodeGen
{
public:
	const std::vector<std::string> &getParallelLoops();
    static int totalNumberOfParallelLoops;
	// Each thread has it's own position in Grid
	// That position is computed in runtime for each dimension of grid
	vector<Value*> BlockPositionInGrid;
	vector<Value*> ThreadPositionInBlock;

	// For each dimension of grid computes it's size (count of threads)
	// GridSize contains respectively Value*
	vector<Value*> GridSize;

	// For each dimension of block it's size obtained by call to one of the CUDA Functions
	// BlockSize contains respectively Value*
	vector<Value*> BlockSize;

	// Maximal count of good nested parallel loops, which can be parallelized
	int MaxDimensionsCount;

    uint64_t startDepth;
	uint64_t loopCount;
	int loopsGenerated;
public:
	// The Scop we code generate.
	Scop *S;
	Pass *P;
	Function *f;

	// The Builder specifies the current location to code generate at.
	IRBuilder<> &Builder;

	// Map the Values from the old code to their counterparts in the new code.
	ValueMapT ValueMap;

	// clastVars maps from the textual representation of a clast variable to its
	// current *Value. clast variables are scheduling variables, original
	// induction variables or parameters. They are used either in loop bounds or
	// to define the statement instance that is executed.
	//
	//   for (s = 0; s < n + 3; ++i)
	//     for (t = s; t < m; ++j)
	//       Stmt(i = s + 3 * m, j = t);
	//
	// {s,t,i,j,n,m} is the set of clast variables in this clast.
	CharMapT ClastVars;

	// Codegenerator for clast expressions.
	ClastExpCodeGen ExpGen;

	void codegenSubstitutions(const clast_stmt *Assignment,
	                          ScopStmt *Statement, int vectorDim = 0,
	                          std::vector<ValueMapT> *VectorVMap = 0);

	/// @brief Create a classical sequential loop.
	void codegenForSequential(const clast_for *f);

	void createCUDAGridParamsAndPosInGridBlocks();

	void codegenForCUDA(const clast_for *f);

	void addParameters(const CloogNames *names);

	IntegerType *getIntPtrTy() {
		return P->getAnalysis<TargetData>().getIntPtrType(Builder.getContext());
	}

	Value *codegen(const clast_equation *eq) {
		Value *LHS = ExpGen.codegen(eq->LHS, getIntPtrTy());
		Value *RHS = ExpGen.codegen(eq->RHS, getIntPtrTy());
		CmpInst::Predicate P;

		if (eq->sign == 0)
			P = ICmpInst::ICMP_EQ;
		else if (eq->sign > 0)
			P = ICmpInst::ICMP_SGE;
		else
			P = ICmpInst::ICMP_SLE;

		return Builder.CreateICmp(P, LHS, RHS);
	}

public:

	void codegen(const clast_root *r) {
		addParameters(r->names);

		const clast_stmt *stmt = (const clast_stmt*) r;
		loopsGenerated = 0;
		
		if(loopCount >= 1 && CodeGenForCuda)
		    createCUDAGridParamsAndPosInGridBlocks();

		if (stmt->next)
			codegen(stmt->next);
	}

	void codegen(const clast_stmt *stmt) {
		if	    (CLAST_STMT_IS_A(stmt, stmt_root))
			assert(false && "No second root statement expected");
		else if (CLAST_STMT_IS_A(stmt, stmt_ass))
			codegen((const clast_assignment *)stmt);
		else if (CLAST_STMT_IS_A(stmt, stmt_user))
			codegen((const clast_user_stmt *)stmt);
		else if (CLAST_STMT_IS_A(stmt, stmt_block))
			codegen((const clast_block *)stmt);
		else if (CLAST_STMT_IS_A(stmt, stmt_for))
			codegen((const clast_for *)stmt);
		else if (CLAST_STMT_IS_A(stmt, stmt_guard))
			codegen((const clast_guard *)stmt);

		if (stmt->next)
			codegen(stmt->next);
	}

	void codegen(const clast_assignment *a) {
		Value *V= ExpGen.codegen(a->RHS, getIntPtrTy());
		ClastVars[a->LHS] = V;
	}

	void codegen(const clast_assignment *A, ScopStmt *Stmt,
	             unsigned Dim, int VectorDim = 0,
	             std::vector<ValueMapT> *VectorVMap = 0) {

		const PHINode *PN;
		Value *RHS;

		assert(!A->LHS && "Statement assignments do not have left hand side");

		PN = Stmt->getInductionVariableForDimension(Dim);
		RHS = ExpGen.codegen(A->RHS, Builder.getInt64Ty());
		RHS = Builder.CreateTruncOrBitCast(RHS, PN->getType());

		if (VectorVMap)
			(*VectorVMap)[VectorDim][PN] = RHS;

		ValueMap[PN] = RHS;
	}

	void codegen(const clast_user_stmt *u, std::vector<Value*> *IVS = NULL,
	             const char *iterator = NULL, isl_set *Domain = 0) {
		ScopStmt *Statement = (ScopStmt *)u->statement->usr;

		if (u->substitutions)
			codegenSubstitutions(u->substitutions, Statement);

		int VectorDimensions = IVS ? IVS->size() : 1;

		if (VectorDimensions == 1) {
			BlockGenerator::generate(Builder, *Statement, ValueMap, P);
			return;
		}
		assert(false);
		/*VectorValueMapT VectorMap(VectorDimensions);

		if (IVS) {
			assert (u->substitutions && "Substitutions expected!");
			int i = 0;
			for (std::vector<Value*>::iterator II = IVS->begin(), IE = IVS->end();
			     II != IE; ++II) {
				ClastVars[iterator] = *II;
				codegenSubstitutions(u->substitutions, Statement, i, &VectorMap);
				i++;
			}
		}

		VectorBlockGenerator::generate(Builder, *Statement, VectorMap, Domain, P);*/
	}

	void codegen(const clast_block *b) {
		if (b->body)
			codegen(b->body);
	}

	void codegen(const clast_for *f) {
		loopsGenerated++;
		if(loopsGenerated - 1< loopCount && CodeGenForCuda)
		    codegenForCUDA(f);
		else
			codegenForSequential(f);
		
	}

	void codegen(const clast_guard *g) {
		Function *F = Builder.GetInsertBlock()->getParent();
		LLVMContext &Context = F->getContext();

		BasicBlock *CondBB = SplitBlock(Builder.GetInsertBlock(),
		                                Builder.GetInsertPoint(), P);
		CondBB->setName("polly.cond");
		BasicBlock *MergeBB = SplitBlock(CondBB, CondBB->begin(), P);
		MergeBB->setName("polly.merge");
		BasicBlock *ThenBB = BasicBlock::Create(Context, "polly.then", F);

		//DominatorTree &DT = P->getAnalysis<DominatorTree>();
		//DT.addNewBlock(ThenBB, CondBB);
		//DT.changeImmediateDominator(MergeBB, CondBB);

		CondBB->getTerminator()->eraseFromParent();

		Builder.SetInsertPoint(CondBB);

		Value *Predicate = codegen(&(g->eq[0]));

		for (int i = 1; i < g->n; ++i) {
			Value *TmpPredicate = codegen(&(g->eq[i]));
			Predicate = Builder.CreateAnd(Predicate, TmpPredicate);
		}

		Builder.CreateCondBr(Predicate, ThenBB, MergeBB);
		Builder.SetInsertPoint(ThenBB);
		Builder.CreateBr(MergeBB);
		Builder.SetInsertPoint(ThenBB->begin());

		codegen(g->then);

		MergeBB->moveAfter(Builder.GetInsertBlock());

		Builder.SetInsertPoint(MergeBB->begin());
	}


	ClastStmtCodeGen(Scop *scop, IRBuilder<> &B, Pass *P);
};

class KernelCodeGenerator : public KernelScopPass
{
public:
	static char ID;
	
	KernelScopInfo *KSI;
	KernelCloogInfo *KCI;
	DominatorTree *DT;
	RegionInfo *RI;
	Region *region;

	Scop *scop;
	Function *f;
	int startLoopDepth;

	static void addCUDADeclarations(Module *M) {
		IRBuilder<> Builder(M->getContext());
		LLVMContext &Context = Builder.getContext();
		IntegerType *intType = Type::getInt32Ty(Context);

		if (!M->getFunction("llvm.nvvm.read.ptx.sreg.tid.x")) {
			//  Define all dimensions, that can be used while code generation.
			dimensions.push_back("z");
			dimensions.push_back("y");
			dimensions.push_back("x");

			// Define parameters of dimensions.
			vector<string> parameters;
			parameters.push_back("tid");
			parameters.push_back("ctaid");
			parameters.push_back("ntid");
			parameters.push_back("nctaid");

			string prefix1("llvm.nvvm.read.ptx.sreg.");
			string prefix2(".");
			string prefix3(".");

			for(unsigned int i = 0; i < dimensions.size(); i++)
				for(unsigned int j =0; j < parameters.size(); j++) {
					CudaFunctions.push_back((new string(prefix1 + parameters[j] +
					                                    prefix2 + dimensions[i]))->c_str());
					CudaInricics[CudaFunctions.back()] = (new string(parameters[j] +
					                                      prefix3 + dimensions[i]))->c_str();
				}

			for(unsigned int i = 0; i < CudaFunctions.size(); i++) {
				FunctionType *FT = FunctionType::get(intType, std::vector<Type*>(), false);
				Function::Create(FT, Function::ExternalLinkage,(CudaFunctions)[i], M);
			}
		}
	}

	KernelCodeGenerator() : KernelScopPass(ID) {}


	BasicBlock* getReturnBlock() {
		Instruction *returnInst = NULL;
		for(llvm::Function::iterator basicBlock=f->begin(), basicBlockEnd=f->end();
		    basicBlock != basicBlockEnd; basicBlock ++) {

			if(isa<ReturnInst>(*basicBlock->getTerminator())) {
				if(returnInst != NULL)
					// function has more than one instrunction
					assert(false);
				//store return instruction in current leaf of ScalarEvolution-tree
				returnInst = cast<ReturnInst>(basicBlock->getTerminator());
			}
		}
		assert(returnInst != NULL);
		return returnInst->getParent();
	}

	virtual bool runOnScop(Scop &S) {
		KSI = &getAnalysis<KernelScopInfo>();
		KCI = &getAnalysis<KernelCloogInfo>();
		DT = &getAnalysis<DominatorTree>();
 		RI =  &getAnalysis<RegionInfo>();
        
		scop = &S;
		f = scop -> rootFunction;
		startLoopDepth = 1;// передавать через метаданные

		/*BasicBlock *exitBlock = getReturnBlock();
		BasicBlock *exitingBlock = exitBlock->getSinglePredecessor();
		assert(exitingBlock);
		assert( (++succ_begin(&f->getEntryBlock())) == succ_end(&f->getEntryBlock()));
		BasicBlock *entryBlock = *succ_begin(&f->getEntryBlock());

		region = new Region(entryBlock, exitBlock, RI, DT, RI->getTopLevelRegion());

		assert(region);

		// The builder will be set to startBlock.

		IRBuilder<> builder(entryBlock);

		BasicBlock *splitBlock = addSplitAndStartBlock(&builder);
		BasicBlock *StartBlock = builder.GetInsertBlock();

		mergeControlFlow(splitBlock, &builder);
		builder.SetInsertPoint(StartBlock->begin());*/

		LLVMContext & context = f->getParent()->getContext();

		NamedMDNode *nodeForFunction = f->getParent()->getNamedMetadata(f->getName());
		unsigned MDKindID = f->getParent()->getMDKindID("depthOfCallInst");
		//const StringRef &functionName = f->getName();
		//f->setName(functionName + "_old");
		
		for(int i =0; i < nodeForFunction->getNumOperands(); i++) {

			uint64_t startDepth = cast<ConstantInt>(nodeForFunction->getOperand(i)->getOperand(0))->getZExtValue();
			uint64_t loopCount = cast<ConstantInt>(nodeForFunction->getOperand(i)->getOperand(1))->getZExtValue();

			Function *newFunction = Function::Create(f->getFunctionType(), f->getLinkage(),
			                        /*functionName*/f->getName() + "_generated",f->getParent());
			BasicBlock * newEntryBlock = BasicBlock::Create(context, "Entry",
			                             newFunction);

			BasicBlock * newExitBlock = BasicBlock::Create(context, "tmpExit",
			                            newFunction);
			BranchInst::Create(newExitBlock, newEntryBlock);
			ReturnInst::Create(context,UndefValue::get(f->getReturnType()),newExitBlock);


			IRBuilder<> builder(newEntryBlock);
			ClastStmtCodeGen CodeGen(&S, builder, this);

			Function::arg_iterator newArg = newFunction->arg_begin();
			for(Function::arg_iterator arg = f->arg_begin(), argEnd = f->arg_end();
			    arg != argEnd; arg++) {
				newArg->setName("new_" + arg->getName());
				CodeGen.ValueMap[&(*arg)] = &(*newArg);
				newArg++;
			}
			CodeGen.f = newFunction;
			CodeGen.startDepth = startDepth;
			CodeGen.loopCount = loopCount;
			CodeGen.codegen(KCI->getClast());

			BasicBlock *returnBlock = newExitBlock->getSinglePredecessor();
			assert(returnBlock);
			returnBlock->getTerminator()->eraseFromParent();
			newExitBlock->eraseFromParent();

			assert(isa<ReturnInst>(*returnBlock->getTerminator()));

			assert(!verifyFunction(*f) && "code generation failed : function was broken");

			std::list<CallInst *> callInsts;
			callInsts.clear();
			
			for(Value::use_iterator user = f->use_begin(), userEnd = f->use_end();
			    user != userEnd; user++) {
				if(CallInst *callInst = dyn_cast<CallInst>(*user)) {
					MDNode *node = callInst->getMetadata(MDKindID);
					uint64_t callStartDepth = cast<ConstantInt>(node->getOperand(0))->getZExtValue();
					uint64_t callLoopCount = cast<ConstantInt>(node->getOperand(1))->getZExtValue();
					if(callStartDepth == startDepth && loopCount == callLoopCount)
						callInsts.push_back(callInst);
				}
			}
			
			for(std::list<CallInst *>::iterator iter = callInsts.begin(), iterEnd = callInsts.end();
			iter != iterEnd; iter++)
				(*iter)->replaceUsesOfWith(f, newFunction);
				
			/*newArg = newFunction->arg_begin();
			for(Function::arg_iterator arg = f->arg_begin(), argEnd = f->arg_end();
			    arg != argEnd; arg++) {
				newArg->setName(arg->getName());
				newArg++;
			}*/
		}
		return false;

	}

	void getAnalysisUsage(AnalysisUsage &AU) const {
		/// AU.addRequired<KernelPrepare>();
		KernelScopPass::getAnalysisUsage(AU);
		AU.addRequired<KernelCloogInfo>();
		AU.addRequired<DominatorTree>();
		AU.addRequired<TargetData>();
		AU.addRequired<RegionInfo>();
		AU.addRequired<ScalarEvolution>();
		AU.setPreservesAll();
	}
};

  Pass *createKernelCodeGeneratorPass();
};

namespace llvm
{
class PassRegistry;
void initializeKernelCodeGeneratorPass(llvm::PassRegistry&);
}

#endif
