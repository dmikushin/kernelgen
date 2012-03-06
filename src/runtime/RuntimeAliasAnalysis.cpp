#include "polly/Cloog.h"
#include "polly/ScopInfo.h"
#include "polly/ScopPass.h"
#include "polly/Support/GICHelper.h"
#include "llvm/Support/raw_os_ostream.h"
#include "llvm/Assembly/Writer.h"
#include "llvm/Constants.h"
#include "llvm/Target/TargetData.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/Support/Debug.h"

#include "isl/set.h"
#include "isl/map.h"
#include "isl/aff.h"
#include "isl/printer.h"
#include "isl/ilp.h"
#include "isl/int.h"

#include <iostream>
#include <fstream>
#include <vector>

#define DEBUG_TYPE "runtime-AA"

STATISTIC(Readings,  "Number of readings");
STATISTIC(Writings,   "Number of writings");
STATISTIC(ReadWrite,   "FAIL: Number of read-write conflicts");
STATISTIC(WriteWrite,  "FAIL: Number of write-write conflicts");
STATISTIC(NoAAScops,   "RESULT: Number of good scops (with no memory aliasing)");

using namespace std;

using namespace polly;
using namespace std;


typedef pair<uint64_t, uint64_t> MemoryRange;
bool isIntersect(MemoryRange &range1, MemoryRange &range2)
{
	return !(range2.first > range1.second ||
	         range1.first > range2.second );
}
void printRange(MemoryRange &range, raw_ostream &OS)
{
	OS << "[" << range.first << ", " << range.second << "]";
}


class RuntimeAliasAnalysis : public ScopPass
{
		Scop *S;
		TargetData * TD;
		CloogInfo *C;

		vector<MemoryRange> readingMemoryRanges;
		vector<MemoryRange> writingMemoryRanges;

	public:
		static char ID;
		RuntimeAliasAnalysis()
			:ScopPass(ID) {
			Readings=0;
			Writings=0;
			ReadWrite=0;
			WriteWrite=0;
			NoAAScops=0;
		}
		bool runOnScop(Scop &scop);
		static int getAccessFunction(__isl_take isl_set *Set, __isl_take isl_aff *Aff, void *User);
		void getAnalysisUsage(AnalysisUsage &AU) const {
			ScopPass::getAnalysisUsage(AU);
			AU.addRequired<TargetData>();
			AU.addRequired<CloogInfo>();
			AU.setPreservesAll();
		}
};

int RuntimeAliasAnalysis::getAccessFunction(__isl_take isl_set *Set, __isl_take isl_aff *Aff, void *User)
{
	assert((  *((isl_aff **)User) == NULL) && "Result is already set."
	       "Currently only single isl_aff is supported");
	assert(isl_set_plain_is_universe(Set)
	       && "Code generation failed because the set is not universe");

	*((isl_aff **)User)=Aff;
	isl_set_free(Set);
	return 0;
}
static int print_flag = 1;
bool RuntimeAliasAnalysis::runOnScop(Scop &scop)
{
	S = &scop;
	TD = &getAnalysis<TargetData>();
	
	raw_os_ostream OS(cout);
	if(print_flag)
	OS << "\n<------------------------------ One Another Scop ----------------------------->\n\n";

    assert(scop.getNumParams() == 0 &&
	       "FIXME: "
	       "After Constant Substitution number of scop's global parameters must be zero");
	//<foreach statement in scop>
	for(Scop::iterator stmt_iterator = S->begin(), stmt_iterator_end = S->end();
	    stmt_iterator != stmt_iterator_end; stmt_iterator++) {

		ScopStmt * stmt = *stmt_iterator;
		if(strcmp(stmt -> getBaseName(),"FinalRead")) {
			isl_set *stmtDomain = stmt->getDomain();
			if(print_flag) {

				OS.indent(8) << stmt -> getBaseName() << " domain := "<<"\n";// print name of statement
				OS.indent(12) << stmt -> getDomainStr() << "\n\n";           // print domain of statement
			}
			//<foreach memory access in scop>
			for(ScopStmt::memacc_iterator access_iterator=stmt->memacc_begin(), access_iterator_end = stmt->memacc_end();
			    access_iterator != access_iterator_end; access_iterator++) {

				MemoryAccess* memoryAccess=*access_iterator;
				const Value * baseAddressValue = memoryAccess -> getBaseAddr();

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

				// get access function from memory access
				isl_pw_aff * access = isl_map_dim_max(memoryAccess->getAccessRelation(), 0);
				isl_aff *accessFunction=NULL;
				isl_pw_aff_foreach_piece(access,getAccessFunction,&accessFunction);
				assert(accessFunction &&
				       "isl_pw_aff is empty?!");

				//isl_int to keep max and min
				isl_int islMax, islMin;
				isl_int_init(islMax);
				isl_int_init(islMin);

				//enum isl_lp_result {
				//	isl_lp_error = -1,
				//	isl_lp_ok = 0,
				//	isl_lp_unbounded,
				//	isl_lp_empty
				//};

				// get max and min of access function within domain constraints
				isl_lp_result lpResult;
				lpResult = isl_set_max(stmtDomain,accessFunction, &islMax);
				assert(lpResult == isl_lp_ok);
				lpResult = isl_set_min(stmtDomain,accessFunction, &islMin);
				assert(lpResult == isl_lp_ok);

				// get intereg values from mpz
				int64_t max = APInt_from_MPZ (islMax).getSExtValue();
				int64_t min = APInt_from_MPZ (islMin).getSExtValue();

				// get size of element, which is accessed
				int storeSize = memoryAccess -> getElemTypeSize();//TD->getTypeStoreSize(type);

				// print information
				if(print_flag) {
					memoryAccess -> print(OS);
					OS.indent(16) << "Base address  =  " << baseAddress << "   SizeOfElement = " << storeSize <<"\n";
					OS.indent(20) << "Minimum offset = " << min << "  Maximum offset = " << max << "\n";
					OS.indent(20) << "Byte min offset = " << storeSize*min << "  Byte max offset = " << storeSize*max << "\n";
					OS.indent(16) << "Total access range = [" << baseAddress + storeSize*min << ", " <<  baseAddress + storeSize*max << "]\n\n";
				}
				// save range
				if(memoryAccess->isRead())
					readingMemoryRanges.push_back(make_pair(baseAddress + storeSize*min, baseAddress + storeSize*max));
				else writingMemoryRanges.push_back(make_pair(baseAddress + storeSize*min, baseAddress + storeSize*max));

				// free isl data structures
				isl_int_clear(islMax);
				isl_int_clear(islMin);
				isl_aff_free(accessFunction);
				isl_pw_aff_free(access);

			}//</ foreach memory access in scop>
			isl_set_free(stmtDomain);
		}

	}//</ foreach statement in scop>

	//statistics
	Readings += readingMemoryRanges.size();//,  "Number of readings");
	Writings += writingMemoryRanges.size();//,   "Number of writings");

#define PRINT_TESTING_PAIR(first, second, stream, flag) \
	if(print_flag) { \
		(stream).indent(8) << "Testing pair : "; \
		printRange((first),(stream)); \
		(stream) << " vs ";	\
		printRange((second),(stream)); \
		if( !( flag = isIntersect( (first), ( second)) )) (stream) << " .... OK!\n"; \
		else   (stream) << " .... FAIL!\n"; \
	}

	bool flag = true;
	bool localFlag = true;

	if(writingMemoryRanges.size()) {
		//<test if there are read-write conflicts>
		for(unsigned i = 0; i < readingMemoryRanges.size(); i++)
			for(unsigned j = 0; j < writingMemoryRanges.size(); j++) {
				PRINT_TESTING_PAIR(readingMemoryRanges[i], writingMemoryRanges[j], OS, localFlag)
				flag = flag && !localFlag;
				if(localFlag) ReadWrite++; //,   "Number of read-write conflicts"
			}
		// </ test if there are read-write conflicts>
		//<test if there are write-write conflicts>
		for(unsigned i = 0; i < writingMemoryRanges.size(); i++)
			for(unsigned j = i+1; j < writingMemoryRanges.size(); j++)
				{
					PRINT_TESTING_PAIR(writingMemoryRanges[i], writingMemoryRanges[j], OS, localFlag)
					flag = flag && !localFlag;
					if(localFlag) WriteWrite++; //,   "Number of write-write conflicts"
				}
		//</ test if there are write-write conflicts>
	}
	if(!flag ) getAnalysis<ScopInfo>().releaseMemory(); // delete scop
	if(flag) NoAAScops++;// "Number of scops with no memory aliasing"
	OS.flush();
	return false;
}

char RuntimeAliasAnalysis::ID = 0;
Pass* createRuntimeAliasAnalysisPass()
{
	return new RuntimeAliasAnalysis();
}
namespace llvm
{
	class PassRegistry;
	void initializeRuntimeAliasAnalysisPass(llvm::PassRegistry&);
}
INITIALIZE_PASS_BEGIN(RuntimeAliasAnalysis, "runtime-alias-analysis",
                      "kernelgen's runtime alias analysis of concrete pointer's values", false,
                      false)
INITIALIZE_PASS_DEPENDENCY(ScopInfo)
INITIALIZE_PASS_DEPENDENCY(CloogInfo)
INITIALIZE_PASS_DEPENDENCY(TargetData)
INITIALIZE_PASS_END(RuntimeAliasAnalysis, "runtime-alias-analysis",
                    "kernelgen's runtime alias analysis of concrete pointer's values", false,
                    false)
static RegisterPass<RuntimeAliasAnalysis>
Z("RuntimeAliasAnalysis", "Kernelgen - Runtime Alias Analysis");