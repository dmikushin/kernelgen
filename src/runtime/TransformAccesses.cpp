//===- TransformAccesses.cpp - LLVM pass for transforming memory accesses -===//
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

#include "polly/CodeGen/Cloog.h"
#include "polly/ScopInfo.h"
#include "polly/ScopPass.h"
#include "polly/Support/GICHelper.h"
#include "llvm/Support/raw_os_ostream.h"
#include "llvm/Assembly/Writer.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/Instructions.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/Support/Debug.h"

#include "isl/set.h"
#include "isl/map.h"
#include "isl/printer.h"
#include "isl/ilp.h"
#include "isl/int.h"
#include <isl/aff.h>

#include <iostream>
#include <fstream>
#include <vector>
#include "polly/Dependences.h"

#include "KernelGen.h"

#undef DEBUG_TYPE
#define DEBUG_TYPE "access-general-form"

//!!!STATISTIC(Readings,  "Number of readings");

using namespace kernelgen;
using namespace polly;
using namespace std;

class TransformAccesses : public polly::ScopPass
{
	Scop *S;
	DataLayout * DL;
public:
	static char ID;
	TransformAccesses()
		: ScopPass(ID) {
		/*Readings=0;
		Writings=0;
		ReadWrite=0;
		WriteWrite=0;
		NoAAScops=0;*/
	}
	bool runOnScop(Scop &scop);
	void getAnalysisUsage(AnalysisUsage &AU) const {
		ScopPass::getAnalysisUsage(AU);
		AU.addRequired<DataLayout>();
		AU.setPreservesAll();
	}
	static int getAccessFunction(__isl_take isl_set *Set, __isl_take isl_aff *Aff, void *User) {
		assert((  *((isl_aff **)User) == NULL) && "Result is already set."
		       "Currently only single isl_aff is supported");
		assert(isl_set_plain_is_universe(Set)
		       && "Code generation failed because the set is not universe");

		*((isl_aff **)User)=isl_aff_copy(Aff);
		isl_set_free(Set);
		isl_aff_free(Aff);
		return 0;
	}
};

void copyIslAffToConstraint(isl_aff *affToCopy, isl_constraint *constraint)
{
	assert(affToCopy && constraint);
	assert(isl_aff_dim(affToCopy,isl_dim_in) == (int)isl_constraint_dim(constraint,isl_dim_in));

	int nIn = isl_aff_dim(affToCopy,isl_dim_in);
	isl_int tmp;
	isl_int_init(tmp);

	for (int i =0 ; i < nIn; i++) {
		isl_aff_get_coefficient(affToCopy,isl_dim_in,i,&tmp);
		constraint = isl_constraint_set_coefficient(constraint,isl_dim_in,i,tmp);
	}
	isl_aff_get_constant(affToCopy,&tmp);
	constraint = isl_constraint_set_constant(constraint,tmp);
	isl_int_clear(tmp);
}
bool TransformAccesses::runOnScop(Scop &scop)
{
	S = &getCurScop();//&scop;
	DL = &getAnalysis<DataLayout>();

	/*assert(scop.getNumParams() == 0 &&
	       "FIXME: "
	       "After Constant Substitution number of scop's global parameters must be zero"
	       "if there are parameters then outer loop does not parsed");*/

        if(scop.getNumParams() != 0)
	{
		VERBOSE(Verbose::Polly << Verbose::Red <<
			"\n    FAIL: Scop has parameters, maybe not all kernel detected as scop!!!\n" <<
			Verbose::Reset << Verbose::Default);
		getAnalysis<ScopInfo>().releaseMemory();
		return false;
	}


	//<foreach statement in scop>


	for (Scop::iterator stmt_iterator = S->begin(), stmt_iterator_end = S->end();
	    stmt_iterator != stmt_iterator_end; stmt_iterator++) {

		ScopStmt * stmt = *stmt_iterator;
		assert(strcmp(stmt -> getBaseName(),"FinalRead"));

		for (ScopStmt::memacc_iterator access_iterator=stmt->memacc_begin(), access_iterator_end = stmt->memacc_end();
		    access_iterator != access_iterator_end; access_iterator++) {
		    
			MemoryAccess* memoryAccess=*access_iterator;
			int allocSize = memoryAccess->getElemTypeSize();/// !!!!!!
			int storeSize = memoryAccess->getElemTypeSize();
			isl_map * accessRelation = memoryAccess->getAccessRelation();
			isl_space * space = isl_map_get_space(accessRelation);
			const Value * baseAddressValue = memoryAccess -> getBaseAddr();

			assert((isl_map_dim(accessRelation, isl_dim_out) == 1)
			       && "Only single dimensional access functions supported");

			if (isa<AllocaInst>(*baseAddressValue)) {
				VERBOSE(Verbose::Polly << "MemoryAccess to alloca: " << *baseAddressValue <<
						"\n    " << memoryAccess->getAccessRelationStr() <<
						"\n        " << "allocSize: "<< allocSize << " storeSize: " << storeSize << "\n" <<
						Verbose::Default);
				memoryAccess->setGeneralAccessRelation(accessRelation);
				memoryAccess->setCurrentRelationType(MemoryAccess::RelationType_general);
				isl_space_free(space);
				isl_map_free(accessRelation);
			} else {
				VERBOSE(Verbose::Polly << "MemoryAccess to pointer: " << *baseAddressValue <<
						"\n    " << memoryAccess->getAccessRelationStr() <<
						"\n        " << "allocSize: "<< allocSize << " storeSize: " << storeSize << "\n" <<
						Verbose::Default);

				//assert(isa<ConstantExpr>(*baseAddressValue)&&
				//       "that must be substituted constant expression");
				if(!isa<ConstantExpr>(*baseAddressValue))
				{
					VERBOSE(Verbose::Polly << Verbose::Red <<
						"\n    FAIL: Scop contains indirect addressing, can not compute access conflicts!!!\n" <<
						Verbose::Reset << Verbose::Default);
					isl_space_free(space);
					isl_map_free(accessRelation);
					getAnalysis<ScopInfo>().releaseMemory();
					return false;
				}

                                const ConstantExpr * expr = cast<ConstantExpr>(baseAddressValue);
				assert(expr->getOpcode() == Instruction::IntToPtr &&
				       "constant expression must be IntToPtr");
				assert(isa<ConstantInt>(*expr -> getOperand(0)) &&
				       "the parameter must be integer constant");

				// get actual base address
				ConstantInt * baseAddressConstant = cast<ConstantInt>(expr -> getOperand(0));
				uint64_t baseAddress = baseAddressConstant->getZExtValue();

				isl_pw_aff * access = isl_map_dim_max(accessRelation, 0);
				isl_aff *accessFunction=NULL;
				isl_pw_aff_foreach_piece(access,getAccessFunction,&accessFunction);
				assert(accessFunction &&
				       "isl_pw_aff is empty?!");

				VERBOSE(Verbose::Polly << Verbose::Red);

				isl_local_space *affSpace = isl_aff_get_local_space(accessFunction);
				isl_local_space * localSpace = isl_local_space_from_space(isl_space_copy(space));

				assert(isl_local_space_dim(affSpace,isl_dim_in) == isl_local_space_dim(localSpace, isl_dim_in) );
				assert(isl_local_space_dim(localSpace,isl_dim_out)==1);

				isl_aff * multiplier = isl_aff_zero_on_domain(isl_aff_get_domain_local_space(accessFunction));

				multiplier = isl_aff_set_constant_si(multiplier,allocSize);
				isl_aff *result = isl_aff_mul(accessFunction,multiplier);

				isl_int address;
				isl_int_init(address);
				isl_int_set_ui(address, baseAddress);
				result = isl_aff_add_constant(result, address);
				isl_int_clear(address);

				isl_constraint *one = NULL,*two = NULL;
				isl_map * newMap = NULL;

				newMap = isl_map_universe(space);
				{
					result = isl_aff_neg(result);
					one = isl_inequality_alloc(isl_local_space_copy(localSpace));
					copyIslAffToConstraint(result,one);
					one = isl_constraint_set_coefficient_si(one, isl_dim_out,0,1);
				}
				newMap = isl_map_add_constraint(newMap,one);
				{
					result = isl_aff_neg(result);
					result = isl_aff_add_constant_si(result,storeSize-1);
					two = isl_inequality_alloc(isl_local_space_copy(localSpace));
					copyIslAffToConstraint(result,two);
					two = isl_constraint_set_coefficient_si(two, isl_dim_out,0,-1);
				}
				newMap = isl_map_add_constraint(newMap,two);
				newMap = isl_map_set_tuple_name(newMap, isl_dim_out,"NULL");
				memoryAccess->setGeneralAccessRelation(newMap);
				memoryAccess->setCurrentRelationType(MemoryAccess::RelationType_general);

				VERBOSE(Verbose::Polly << "    replacedBy: "  << stringFromIslObj(newMap) << "\n" <<
						Verbose::Reset << Verbose::Default);

				isl_pw_aff_free(access);
				isl_local_space_free(affSpace);
				isl_local_space_free(localSpace);
				isl_aff_free(result);
				isl_map_free(newMap);

			}
		}

	}

	/*// Iterate through all SCoP statements and find independent
	// memory accesses.
	list<MemoryAccess*> IndepMA;
	for (Scop::iterator StmtI1 = S->begin(), StmtE1 = S->end(); StmtI1 != StmtE1; StmtI1++)
	{
		ScopStmt* Stmt = *StmtI1;

		for (ScopStmt::memacc_iterator MAI1 = Stmt->memacc_begin(),
			MAE1 = Stmt->memacc_end(); (MAI1 != MAE1) && IndependentMA; MAI1++)
		{
			MemoryAccess* MA1 = *MAI1;
			const Value* BA1 = memoryAccess->getBaseAddr();
			ConstantInt* BAC1 = cast<ConstantInt>(expr->getOperand(0));
			uint64_t Addr1 = baseAddressConstant->getZExtValue();					

			for (Scop::iterator StmtI2 = S->begin(), StmtE2 = S->end(); StmtI2 != StmtE2; StmtI2++)
			{
				ScopStmt* Stmt = *StmtI2;

				for (ScopStmt::memacc_iterator MAI2 = Stmt->memacc_begin(),
					MAE2 = Stmt->memacc_end(); (MAI2 != MAE2) && IndependentMA; MAI2++)
				{
					MemoryAccess* MA2 = *MAI2;
					const Value* BA2 = memoryAccess->getBaseAddr();
					ConstantInt* BAC2 = cast<ConstantInt>(expr->getOperand(0));
					uint64_t Addr2 = baseAddressConstant->getZExtValue();					
				}
			}
		}


		// Iterate through all instructions in basic block.
		for (BasicBlock::Iterator II = BB->begin(), IE = BB->end(); II != IE; II++)
		{
			Instruction* I = II;
			MemoryAccess* MA1 = Stmt->lookupAccessFor(I);
			if (!MA1) continue;
			
			// Check if there are no memory accesses, that fall into
			// MA's address range and have different read/write mode.
			bool IndependentMA = true;
			for (Scop::iterator StmtI2 = S->begin(), StmtE2 = S->end();
				(StmtI2 != StmtE2) && IndependentMA; StmtI2++)
			{
				ScopStmt * stmt = *stmt_iterator;

				for (ScopStmt::memacc_iterator MAI = Stmt->memacc_begin(),
					MAE = Stmt->memacc_end(); (MAI != MAE) && IndependentMA; MAI++)
				{
					MemoryAccess* MA2 = *AI;
					
					// TODO: if MA1 intersects with MA2 and has different
					// read/write mode, then MA1 is not independent.
					IndependentMA = false;
				}
			}
			
			if (IndependentMA)
		}
	}*/

	return false;
}

char TransformAccesses::ID = 0;
Pass* createTransformAccessesPass()
{
	return new TransformAccesses();
}
namespace llvm
{
class PassRegistry;
void initializeTransformAccessesPass(llvm::PassRegistry&);
}
INITIALIZE_PASS_BEGIN(TransformAccesses, "transform-accesses",
                      "kernelgen's runtime trnasform accesses to general form", false,
                      false)
INITIALIZE_PASS_DEPENDENCY(ScopInfo)
INITIALIZE_PASS_DEPENDENCY(DataLayout)
INITIALIZE_PASS_END(TransformAccesses, "transform-accesses",
                    "kernelgen's runtime trnasform accesses to general form", false,
                    false)
static RegisterPass<TransformAccesses>
D("TransformAccesses", "Kernelgen - Transformation of accesses to general form");
