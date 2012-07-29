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
#include "isl/printer.h"
#include "isl/ilp.h"
#include "isl/int.h"
#include <isl/aff.h>

#include <iostream>
#include <fstream>
#include <vector>
#include "polly/Dependences.h"

//#include "runtime.h"
#undef DEBUG_TYPE
#define DEBUG_TYPE "access-general-form"

//!!!STATISTIC(Readings,  "Number of readings");

using namespace polly;
using namespace std;
namespace llvm
{
class PassRegistry;
void initializeInspectLoopsPass(llvm::PassRegistry&);
void initializeScopDescriptionPass(llvm::PassRegistry&);
}

class InspectLoops : public polly::ScopPass
{
	Scop *S;
	polly::CloogInfo *C;
	Dependences *DP;
public:
	static char ID;
	InspectLoops()
		:ScopPass(ID) {
		/*Readings=0;
		Writings=0;
		ReadWrite=0;
		WriteWrite=0;
		NoAAScops=0;*/
	}
	bool runOnScop(Scop &scop);
	void getAnalysisUsage(AnalysisUsage &AU) const {
		ScopPass::getAnalysisUsage(AU);
		AU.addRequired<CloogInfo>();
		AU.addRequired<Dependences>();
		AU.setPreservesAll();
	}
	void checkLoopBodyes(const clast_for *for_loop, int indent);
};

void InspectLoops::checkLoopBodyes(const clast_for *for_loop,int indent)
{
	if(DP->isParallelFor(for_loop)) {
		outs().changeColor(raw_ostream::GREEN);
		outs().indent(indent) << "loop is parallel \n" ;
	} else {
		outs().changeColor(raw_ostream::RED);
		outs().indent(indent) << "loop is not parallel \n" ;
	}
	outs().resetColor();
	const clast_stmt *stmt=for_loop->body;
	while(stmt!=NULL) {
		if(CLAST_STMT_IS_A(stmt, stmt_for))
			checkLoopBodyes((const clast_for *)stmt,indent+4);
		stmt=stmt->next;
	}
}
bool InspectLoops::runOnScop(Scop &scop)
{
	S = &getCurScop();//&scop;
	C = &getAnalysis<CloogInfo>();
	DP = &getAnalysis<Dependences>();
	assert(S && DP && C);

	assert(scop.getNumParams() == 0 &&
	       "FIXME: "
	       "After Constant Substitution number of scop's global parameters must be zero"
	       "if there are parameters then outer loop does not parsed");
	//<foreach statement in scop>

	const clast_root *root = C->getClast();
	assert(CLAST_STMT_IS_A(((clast_stmt *)root)->next,stmt_for));
	const clast_for *for_loop =  (clast_for *)((clast_stmt *)root)->next;
	checkLoopBodyes(for_loop,4);

	return false;
}

char InspectLoops::ID = 0;
Pass* createInspectLoopsPass()
{
	return new InspectLoops();
}

INITIALIZE_PASS_BEGIN(InspectLoops, "inspect-loops",
                      "kernelgen's inspect-loops", false,
                      false)
INITIALIZE_PASS_DEPENDENCY(ScopInfo)
INITIALIZE_PASS_DEPENDENCY(CloogInfo)
INITIALIZE_PASS_DEPENDENCY(Dependences)
INITIALIZE_PASS_END(InspectLoops, "inspect-loops",
                    "kernelgen's inspect-loops", false,
                    false)
					
static RegisterPass<InspectLoops>
Z("InspectLoops", "Kernelgen - inspect-loops");


class ScopDescription : public polly::ScopPass
{
	Scop *S;
	polly::CloogInfo *C;
public:
	static char ID;
	ScopDescription()
		:ScopPass(ID) {
		/*Readings=0;
		Writings=0;
		ReadWrite=0;
		WriteWrite=0;
		NoAAScops=0;*/
	}
	bool runOnScop(Scop &scop);
	void getAnalysisUsage(AnalysisUsage &AU) const {
		ScopPass::getAnalysisUsage(AU);
		AU.addRequired<CloogInfo>();
		AU.setPreservesAll();
	}
	static void printCloogAST(CloogInfo &C) {
		outs() << "<------------------- Cloog AST of Scop ------------------->\n";
		C.pprint(outs());
		outs() << "<--------------------------------------------------------->\n";
	}
};

bool ScopDescription::runOnScop(Scop &scop)
{
	S = &getCurScop();//&scop;
	C = &getAnalysis<CloogInfo>();
	assert(S && C);
	outs() << "\n";
	outs() << "<------------------------------ Scop: start --------------------------------->\n";
	printCloogAST(*C);
	scop.print(outs());
	outs() << "<------------------------------ Scop: end ----------------------------------->\n";
	return false;
}

char ScopDescription::ID = 0;
Pass* createScopDescriptionPass()
{
	return new ScopDescription();
}

INITIALIZE_PASS_BEGIN(ScopDescription, "print-scop",
                      "Print cloog and polly's scop definitions", false,
                      false)
INITIALIZE_PASS_DEPENDENCY(ScopInfo)
INITIALIZE_PASS_DEPENDENCY(CloogInfo)
INITIALIZE_PASS_END(ScopDescription, "print-scop",
                    "Print cloog and polly's scop definitions", false,
                    false)
					
static RegisterPass<ScopDescription>
M("ScopDescription", "Print cloog and polly's scop definitions");
