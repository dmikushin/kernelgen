//#include <polly/LinkAllPasses.h>
//#include "polly/Config/config.h"
#include <polly/Support/GICHelper.h>
#include <polly/Support/ScopHelper.h>
#include <polly/Cloog.h>
#include <polly/Dependences.h>
#include <polly/ScopInfo.h>
#include <polly/ScopPass.h>
#include <polly/TempScopInfo.h>

#include <llvm/Support/CommandLine.h>
#include <llvm/Support/Debug.h>
#include <llvm/Support/IRBuilder.h>
#include <llvm/Analysis/LoopInfo.h>
#include <llvm/Analysis/ScalarEvolutionExpander.h>
#include <llvm/Transforms/Utils/BasicBlockUtils.h>
#include <llvm/Target/TargetData.h>
#include <llvm/Module.h>
#include <llvm/ADT/SetVector.h>
#include <llvm/Support/raw_os_ostream.h>

#include <cloog/cloog.h>
#include <cloog/isl/cloog.h>


#include <vector>
#include <utility>
#include <string.h>
#include <fstream>

using namespace std;
namespace llvm
{
class Pass;
class PassInfo;
class RegionPass;
}
using namespace llvm;

struct isl_set;

typedef DenseMap<const Value*, Value*> ValueMapT;
typedef DenseMap<const char*, Value*> CharMapT;
typedef std::vector<ValueMapT> VectorValueMapT;


namespace polly
{

extern char &IndependentBlocksID;
extern char &CodePreperationID;
};

namespace kernelgen
{
//CudaFunction is some extern function, which returns some information about thread position in grid
//CudaFunction returns one of the intricics
//CudaFunctions defined and filled in CodeGeneration.cpp
extern std::vector<const char*> CudaFunctions;

//CudaIntricic is some information about thread position in grid
//CudaInricics defined and filled in CodeGeneration.cpp
extern DenseMap<const char*,const char *> CudaInricics;

//dimensions contains names of dimensions
//dimensions defined and filled in CodeGeneration.cpp
extern vector<string> dimensions;

void createLoop(IRBuilder<> *Builder, Value *LB, Value *UB, APInt Stride,
                PHINode*& IV, BasicBlock*& AfterBB, Value*& IncrementedIV,
                DominatorTree *DT);

void createLoopForCUDA(IRBuilder<> *Builder, Value *LB, Value *UB,
                       Value * ThreadLB, Value * ThreadUB, Value * ThreadStride,
                       PHINode*& IV, BasicBlock*& AfterBB, Value*& IncrementedIV,
                       DominatorTree *DT, const char * dimension = "");

/// Class to generate LLVM-IR that calculates the value of a clast_expr.
class ClastExpCodeGen
{
	IRBuilder<> &Builder;
	const CharMapT *IVS;

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
	ClastExpCodeGen(IRBuilder<> &B, CharMapT *IVMap) : Builder(B), IVS(IVMap) {}

	// Generates code to calculate a given clast expression.
	//
	// @param e The expression to calculate.
	// @return The Value that holds the result.
	Value *codegen(const clast_expr *e, Type *Ty);
	// @brief Reset the CharMap.
	//
	// This function is called to reset the CharMap to new one, while generating
	// OpenMP code.
	void setIVS(CharMapT *IVSNew) {
		IVS = IVSNew;
	}

};

class BlockGenerator
{
	IRBuilder<> &Builder;
	ValueMapT &VMap;
	VectorValueMapT &ValueMaps;
	polly::Scop &S;
	polly::ScopStmt &statement;
	isl_set *scatteringDomain;

public:
	BlockGenerator(IRBuilder<> &B, ValueMapT &vmap, VectorValueMapT &vmaps,
	               polly::ScopStmt &Stmt, isl_set *domain)
		: Builder(B), VMap(vmap), ValueMaps(vmaps), S(*Stmt.getParent()),
		  statement(Stmt), scatteringDomain(domain) {}

	const Region &getRegion() {
		return S.getRegion();
	}

	Value* makeVectorOperand(Value *operand, int vectorWidth);

	Value* getOperand(const Value *oldOperand, ValueMapT &BBMap,
	                  ValueMapT *VectorMap = 0);

	Type *getVectorPtrTy(const Value *V, int vectorWidth);

	/// @brief Load a vector from a set of adjacent scalars
	///
	/// In case a set of scalars is known to be next to each other in memory,
	/// create a vector load that loads those scalars
	///
	/// %vector_ptr= bitcast double* %p to <4 x double>*
	/// %vec_full = load <4 x double>* %vector_ptr
	///
	Value *generateStrideOneLoad(const LoadInst *load, ValueMapT &BBMap,
	                             int size);

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
	Value *generateStrideZeroLoad(const LoadInst *load, ValueMapT &BBMap,
	                              int size);

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
	Value *generateUnknownStrideLoad(const LoadInst *load,
	                                 VectorValueMapT &scalarMaps,
	                                 int size);

	/// @brief Get the new operand address according to the changed access in
	///        JSCOP file.
	Value *getNewAccessOperand(isl_map *newAccessRelation, Value *baseAddr,
	                           const Value *oldOperand, ValueMapT &BBMap);

	/// @brief Generate the operand address
	Value *generateLocationAccessed(const Instruction *Inst,
	                                const Value *pointer, ValueMapT &BBMap );

	Value *generateScalarLoad(const LoadInst *load, ValueMapT &BBMap) ;

	/// @brief Load a value (or several values as a vector) from memory.
	void generateLoad(const LoadInst *load, ValueMapT &vectorMap,
	                  VectorValueMapT &scalarMaps, int vectorWidth);

	void copyInstruction(const Instruction *Inst, ValueMapT &BBMap,
	                     ValueMapT &vectorMap, VectorValueMapT &scalarMaps,
	                     int vectorDimension, int vectorWidth) ;

	int getVectorSize() {
		return ValueMaps.size();
	}

	bool isVectorBlock() {
		return getVectorSize() > 1;
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
	void copyBB(BasicBlock *BB, DominatorTree *DT);
};
};

namespace kernelgen
{

Pass* createCodeGenerationPass();
//runtime flags
//It is not importamt at all
struct Flags {
public:
	bool Vector, OpenMP, AtLeastOnce, Aligned, CUDA;
	Flags() {
		Vector = OpenMP = AtLeastOnce = Aligned = false;
		CUDA = true;
	}
};
//flags structure
extern Flags flags;
enum Flag {
	VECTOR = 1 << 0,
	OPEN_MP = 1 << 1,
	AT_LEAST_ONCE = 1 << 2,
	ALIGNED = 1 << 3,
	CUDA = 1 << 4
};

void set_flags(unsigned int _flags);
Flags & get_flags();

class CodeGeneration : public polly::ScopPass
{

	Region *region;
	polly::Scop *S;
	DominatorTree *DT;
	ScalarEvolution *SE;
	polly::ScopDetection *SD;
	TargetData *TD;
	RegionInfo *RI;

	std::vector<std::string> parallelLoops;

public:

	static char ID;

	CodeGeneration() : ScopPass(ID) {}

	//Add definitions of functions, which returns CUDA intricics
	void addCUDADefinitions(IRBuilder<> &Builder);

	bool runOnScop(polly::Scop &scop);

	virtual void printScop(raw_ostream &OS)  {
		for (std::vector<std::string>::const_iterator PI = parallelLoops.begin(),
		     PE = parallelLoops.end(); PI != PE; ++PI)
			OS << "Parallel loop with iterator '" << *PI << "' generated\n";
	}

	virtual void getAnalysisUsage(AnalysisUsage &AU) const;
};

class ClastStmtCodeGen
{
	// The Scop we code generate.
	polly::Scop *S;
	ScalarEvolution &SE;
	DominatorTree *DT;
	polly::ScopDetection *SD;
	polly::Dependences *DP;
	TargetData *TD;

	//Each thread has it's own position in Grid
	//That position is computes in runtime for each dimension of grid
	//PositionInGrid contains respectively Value *
	vector<Value* > positionInGrid;

	vector<Value* > BlockPositionInGrid;
	vector<Value* > ThreadPositionInBlock;

	//For each dimension of grid compute it's size (count of threads)
	//GridSize contains respectively Value *
	vector<Value*> GridSize;

	//For each dimension of block it's size obtained by call ti one of the CUDA Functions
	//BlockSize contains respectively Value *
	vector<Value*> BlockSize;

	int goodNestedParallelLoopsCount;

	//Maximal count of good nested parallel loops, which can be parallelized
	int MaxDimensionsCount;

	//vector<Dimension> dimensionsBounds[]
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
	CharMapT *clastVars;

	// Codegenerator for clast expressions.
	ClastExpCodeGen ExpGen;

	// Do we currently generate parallel code?
	bool parallelCodeGeneration;

	std::vector<std::string> parallelLoops;

public:

	const std::vector<std::string> &getParallelLoops() {
		return parallelLoops;
	}

protected:


	void codegenSubstitutions(const clast_stmt *Assignment,
	                          polly::ScopStmt *Statement, int vectorDim = 0,
	                          std::vector<ValueMapT> *VectorVMap = 0) {
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

/////////////////////////////////////////////////////////////////////////////////////
	//
	/// @brief Create a classical sequential loop.                                 //
	void codegenForSequential(const clast_for *f, Value *lowerBound =0,
	                          Value *upperBound =0, Value * ThreadStride = 0,const char * dimensionName = "");            //

	/// @brief Generate code for Loop on terms of CUDA kernel
	void codegenForCUDA(const clast_for *f);
/////////////////////////////////////////////////////////////////////////////////////

//////////////////////////////////////////////////////////////////////////////////////////
	//
	bool isInnermostLoop(const clast_for *f);                                           //
	//
	//
	void addParameters(const CloogNames *names);                                        //
	//
//////////////////////////////////////////////////////////////////////////////////////////

	void codegen(const clast_for *f) {
		if (flags.CUDA && parallelCodeGeneration && parallelLoops.size() < goodNestedParallelLoopsCount/* && DP->isParallelFor(f)*/) {
			parallelLoops.push_back(f->iterator);
			codegenForCUDA(f);
		} else
			codegenForSequential(f);
	}

	Value *codegen(const clast_equation *eq) {
		Value *LHS = ExpGen.codegen(eq->LHS,
		                            TD->getIntPtrType(Builder.getContext()));
		Value *RHS = ExpGen.codegen(eq->RHS,
		                            TD->getIntPtrType(Builder.getContext()));
		CmpInst::Predicate P;

		if (eq->sign == 0)
			P = ICmpInst::ICMP_EQ;
		else if (eq->sign > 0)
			P = ICmpInst::ICMP_SGE;
		else
			P = ICmpInst::ICMP_SLE;

		return Builder.CreateICmp(P, LHS, RHS);
	}

	void codegen(const clast_guard *g) {
		Function *F = Builder.GetInsertBlock()->getParent();
		LLVMContext &Context = F->getContext();
		BasicBlock *ThenBB = BasicBlock::Create(Context, "polly.then", F);
		BasicBlock *MergeBB = BasicBlock::Create(Context, "polly.merge", F);
		DT->addNewBlock(ThenBB, Builder.GetInsertBlock());
		DT->addNewBlock(MergeBB, Builder.GetInsertBlock());

		Value *Predicate = codegen(&(g->eq[0]));

		for (int i = 1; i < g->n; ++i) {
			Value *TmpPredicate = codegen(&(g->eq[i]));
			Predicate = Builder.CreateAnd(Predicate, TmpPredicate);
		}

		Builder.CreateCondBr(Predicate, ThenBB, MergeBB);
		Builder.SetInsertPoint(ThenBB);

		codegen(g->then);

		Builder.CreateBr(MergeBB);
		Builder.SetInsertPoint(MergeBB);
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
		(*clastVars)[a->LHS] = ExpGen.codegen(a->RHS,
		                                      TD->getIntPtrType(Builder.getContext()));
	}

	void codegen(const clast_assignment *a, polly::ScopStmt *Statement,
	             unsigned Dimension, int vectorDim,
	             std::vector<ValueMapT> *VectorVMap = 0) {
		Value *RHS = ExpGen.codegen(a->RHS,
		                            TD->getIntPtrType(Builder.getContext()));

		assert(!a->LHS && "Statement assignments do not have left hand side");
		const PHINode *PN;
		PN = Statement->getInductionVariableForDimension(Dimension);
		const Value *V = PN;

		if (VectorVMap)
			(*VectorVMap)[vectorDim][V] = RHS;

		ValueMap[V] = RHS;
	}

	void codegen(const clast_user_stmt *u, std::vector<Value*> *IVS = NULL,
	             const char *iterator = NULL, isl_set *scatteringDomain = 0) {

		polly::ScopStmt *Statement = (polly::ScopStmt *)u->statement->usr;
		BasicBlock *BB = Statement->getBasicBlock();

		if (u->substitutions)
			codegenSubstitutions(u->substitutions, Statement);

		int vectorDimensions = IVS ? IVS->size() : 1;

		VectorValueMapT VectorValueMap(vectorDimensions);

		if (IVS) {
			assert (u->substitutions && "Substitutions expected!");
			int i = 0;
			for (std::vector<Value*>::iterator II = IVS->begin(), IE = IVS->end();
			     II != IE; ++II) {
				(*clastVars)[iterator] = *II;
				codegenSubstitutions(u->substitutions, Statement, i, &VectorValueMap);
				i++;
			}
		}

		BlockGenerator Generator(Builder, ValueMap, VectorValueMap, *Statement,
		                         scatteringDomain);
		Generator.copyBB(BB, DT);
	}

	void codegen(const clast_block *b) {
		if (b->body)
			codegen(b->body);
	}
	int GoodNestedParallelLoops(const clast_stmt * stmt, int CurrentCount);

	void computeLaunchParameters(std::vector<Value*> & LaunchParameters, const  clast_root *r);

public:

	std::vector<Value*> LaunchParameters;

	void codegen(const clast_root *r) {
		clastVars = new CharMapT();
		addParameters(r->names);
		ExpGen.setIVS(clastVars);
		const clast_stmt *stmt = (const clast_stmt*) r;
/////////////////////////////////////////////////////////////////////////////////
// Determ if there is possibility to parallel Code Generation                  //
/////////////////////////////////////////////////////////////////////////////////
		goodNestedParallelLoopsCount = GoodNestedParallelLoops(stmt->next,0);  //
		if(MaxDimensionsCount == 0 && goodNestedParallelLoopsCount > 0) {
			LaunchParameters.clear();
			computeLaunchParameters(LaunchParameters, r);
		}
		if(goodNestedParallelLoopsCount > MaxDimensionsCount)                  //
			goodNestedParallelLoopsCount = MaxDimensionsCount;                 //
		//
		parallelCodeGeneration = goodNestedParallelLoopsCount > 0;             //
/////////////////////////////////////////////////////////////////////////////////

		if (stmt->next) {
			codegen(stmt->next);
		}

		delete clastVars;
	}

	ClastStmtCodeGen(polly::Scop *scop, ScalarEvolution &se, DominatorTree *dt,
	                 polly::ScopDetection *sd, polly::Dependences *dp, TargetData *td,
	                 IRBuilder<> &B, int MaxDimCount) :
		S(scop), SE(se), DT(dt), SD(sd), DP(dp), TD(td), Builder(B),
		ExpGen(Builder, NULL), MaxDimensionsCount(MaxDimCount) {}

};

}
