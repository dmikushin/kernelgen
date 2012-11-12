#include "KernelCloog.h"
#include "KernelScopInfo.h"
#include "KernelScopPass.h"
#include "GICHelper.h"
#include "Dependences.h"

#include "llvm/Support/raw_os_ostream.h"
#include "llvm/Assembly/Writer.h"
#include "llvm/Constants.h"
#include "llvm/Target/TargetData.h"
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


#include "runtime.h"

#undef DEBUG_TYPE
#define DEBUG_TYPE "access-general-form"

//!!!STATISTIC(Readings,  "Number of readings");

using namespace kernelgen;
using namespace std;

class TransformAccesses : public KernelScopPass
{
	Scop *S;
	TargetData * TD;
public:
	static char ID;
	TransformAccesses()
		:KernelScopPass(ID) {
		/*Readings=0;
		Writings=0;
		ReadWrite=0;
		WriteWrite=0;
		NoAAScops=0;*/
	}
	bool runOnScop(Scop &scop);
	void getAnalysisUsage(AnalysisUsage &AU) const {
		KernelScopPass::getAnalysisUsage(AU);
		AU.addRequired<TargetData>();
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

	for(int i =0 ; i < nIn; i++) {
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
	TD = &getAnalysis<TargetData>();

	assert(scop.getNumParams() == 0 &&
	       "FIXME: "
	       "After Constant Substitution number of scop's global parameters must be zero"
	       "if there are parameters then outer loop does not parsed");
	//<foreach statement in scop>


	for(Scop::iterator stmt_iterator = S->begin(), stmt_iterator_end = S->end();
	    stmt_iterator != stmt_iterator_end; stmt_iterator++) {

		ScopStmt * stmt = *stmt_iterator;
		assert(strcmp(stmt -> getBaseName(),"FinalRead"));


		for(ScopStmt::memacc_iterator access_iterator=stmt->memacc_begin(), access_iterator_end = stmt->memacc_end();
		    access_iterator != access_iterator_end; access_iterator++) {

			MemoryAccess* memoryAccess=*access_iterator;
			int allocSize = memoryAccess->getElemTypeSize();/// !!!!!!
			int storeSize = memoryAccess->getElemTypeSize();
			isl_map * accessRelation = memoryAccess->getAccessRelation();
			isl_space * space = isl_map_get_space(accessRelation);
			const Value * baseAddressValue = memoryAccess -> getBaseAddr();


			assert((isl_map_dim(accessRelation, isl_dim_out) == 1)
			       && "Only single dimensional access functions supported");

			if(isa<AllocaInst>(*baseAddressValue)) {
				if (verbose & KERNELGEN_VERBOSE_POLLYGEN)
				{
					outs() << "MemoryAccess to alloca: " << *baseAddressValue <<"\n";
					outs().indent(4) << memoryAccess->getAccessRelationStr() << "\n";
					outs().indent(8) << "allocSize: "<< allocSize << " storeSize: " << storeSize << "\n";
					//continue;
				}
				memoryAccess->setGeneralAccessRelation(accessRelation);
				memoryAccess->setCurrentRelationType(MemoryAccess::RelationType_general);
				isl_space_free(space);
				isl_map_free(accessRelation);
			} else {
				if (verbose & KERNELGEN_VERBOSE_POLLYGEN)
				{
					outs() << "MemoryAccess to pointer: " << *baseAddressValue <<"\n";
					outs().indent(4) << memoryAccess->getAccessRelationStr() << "\n";
					outs().indent(8) << "allocSize: "<< allocSize << " storeSize: " << storeSize << "\n";
				}

				assert(isa<ConstantExpr>(*baseAddressValue)&&
				       "that must be substituted constant expression");
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

				if (verbose & KERNELGEN_VERBOSE_POLLYGEN)
				{
					outs().changeColor(raw_ostream::RED);
				}

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

				if (verbose & KERNELGEN_VERBOSE_POLLYGEN)
				{
					outs().indent(4) << "replacedBy: "  << stringFromIslObj(newMap) << "\n";
					outs().resetColor();
				}

				isl_pw_aff_free(access);
				isl_local_space_free(affSpace);
				isl_local_space_free(localSpace);
				isl_aff_free(result);
				isl_map_free(newMap);

			}
		}

	}

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
INITIALIZE_PASS_DEPENDENCY(KernelScopInfo)
INITIALIZE_PASS_DEPENDENCY(TargetData)
INITIALIZE_PASS_END(TransformAccesses, "transform-accesses",
                    "kernelgen's runtime trnasform accesses to general form", false,
                    false)
static RegisterPass<TransformAccesses>
D("TransformAccesses", "Kernelgen - Transformation of accesses to general form");
