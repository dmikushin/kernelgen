//===- ScopInspection.cpp - KernelGen SCoP walker -------------------------===//
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

#include "Runtime.h"

#undef DEBUG_TYPE
#define DEBUG_TYPE "access-general-form"

//!!!STATISTIC(Readings,  "Number of readings");

using namespace kernelgen;
using namespace polly;
using namespace std;

namespace llvm
{
class PassRegistry;
void initializeInspectDependencesPass(llvm::PassRegistry&);
void initializeScopDescriptionPass(llvm::PassRegistry&);
}

class InspectDependences : public polly::ScopPass
{
	Scop *S;
	polly::CloogInfo *C;
	Dependences *DP;
public:
	static char ID;
	InspectDependences()
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
	void checkLoopBodies(const clast_for *for_loop, int indent);
	void printDependences(const char * description, Dependences::Type type);
};

void InspectDependences::checkLoopBodies(const clast_for *for_loop,int indent)
{
	if(DP->isParallelFor(for_loop)) {
		if (settings.getVerboseMode() & Verbose::Polly)
		{
			VERBOSE(Verbose::Polly << Verbose::Green);
			outs().indent(indent);
			VERBOSE("loop is parallel\n" << Verbose::Reset << Verbose::Default);
		}
	} else {
		if (settings.getVerboseMode() & Verbose::Polly)
		{
			VERBOSE(Verbose::Polly << Verbose::Red);
			outs().indent(indent);
			VERBOSE("loop is not parallel\n" << Verbose::Reset << Verbose::Default);
		}
	}
	const clast_stmt *stmt=for_loop->body;
	while(stmt!=NULL) {
		if(CLAST_STMT_IS_A(stmt, stmt_for))
			checkLoopBodies((const clast_for *)stmt,indent+4);
		stmt=stmt->next;
	}
}
void InspectDependences::printDependences(const char * description, Dependences::Type type)
{
	isl_union_map *tmp = NULL;
	outs().changeColor(raw_ostream::RED);
	outs()<< description << "\n";
	outs().resetColor();
	outs().indent(4) << stringFromIslObj(tmp = DP->getDependences(type)) << "\n";
	isl_union_map_free(tmp);
}
bool InspectDependences::runOnScop(Scop &scop)
{
	S = &getCurScop();//&scop;
	C = &getAnalysis<CloogInfo>();
	DP = &getAnalysis<Dependences>();
	assert(S && DP && C);

	assert(scop.getNumParams() == 0 &&
	       "FIXME: "
	       "After Constant Substitution number of scop's global parameters must be zero"
	       "if there are parameters then outer loop does not parsed");

	if (settings.getVerboseMode() & Verbose::Polly)
	{
		outs() << "<------------------------------ Scop: dependences --------------------------->\n";

		printDependences("Write after read dependences: ", Dependences::TYPE_WAR);
		printDependences("Read after write dependences: ", Dependences::TYPE_RAW);
		printDependences("Write after write dependences: ", Dependences::TYPE_WAW);
	}

	const clast_root *root = C->getClast();
	assert(root);
	if(((clast_stmt *)root)->next)
	{
	    if(CLAST_STMT_IS_A(((clast_stmt *)root)->next,stmt_for))
		{
	       const clast_for *for_loop =  (clast_for *)((clast_stmt *)root)->next;
	       checkLoopBodies(for_loop,4);
		}
    } else 
	{
		VERBOSE(Verbose::Polly << Verbose::Red <<
				"WARNING: There is useless Scop ( i.e. scop without statements )!!!\n" <<
				Verbose::Reset << Verbose::Default);
		assert(scop.begin() == scop.end());
	}
	
	VERBOSE(Verbose::Polly <<
			"<------------------------------ Scop: dependences end ----------------------->\n" <<
			Verbose::Default);

	return false;
}

char InspectDependences::ID = 0;
Pass* createInspectDependencesPass()
{
	return new InspectDependences();
}

INITIALIZE_PASS_BEGIN(InspectDependences, "inspect-dependences",
                      "kernelgen's inspect-dependences", false,
                      false)
INITIALIZE_PASS_DEPENDENCY(ScopInfo)
INITIALIZE_PASS_DEPENDENCY(CloogInfo)
INITIALIZE_PASS_DEPENDENCY(Dependences)
INITIALIZE_PASS_END(InspectDependences, "inspect-dependences",
                    "kernelgen's inspect dependences", false,
                    false)

static RegisterPass<InspectDependences>
Z("InspectDependences", "Kernelgen - inspect dependences");


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
		if (settings.getVerboseMode() & Verbose::Polly)
		{
			outs() << "<------------------- Cloog AST of Scop ------------------->\n";
			C.pprint(outs());
			outs() << "<--------------------------------------------------------->\n";
		}
	}
};

bool ScopDescription::runOnScop(Scop &scop)
{
	S = &getCurScop();//&scop;
	C = &getAnalysis<CloogInfo>();
	assert(S && C);
	if (settings.getVerboseMode() & Verbose::Polly)
	{
		outs() << "\n";
		outs() << "<------------------------------ Scop: start --------------------------------->\n";
		printCloogAST(*C);
		scop.print(outs());
		outs() << "<------------------------------ Scop: end ----------------------------------->\n";
	}
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

class SetRelationType : public polly::ScopPass
{
	Scop *S;
public:
	static char ID;
	MemoryAccess::RelationType relationType;
	SetRelationType(MemoryAccess::RelationType _relationType = MemoryAccess::RelationType_polly)
		:ScopPass(ID),relationType(_relationType) {
		/*Readings=0;
		Writings=0;
		ReadWrite=0;
		WriteWrite=0;
		NoAAScops=0;*/
	}
	bool runOnScop(Scop &scop);
	void getAnalysisUsage(AnalysisUsage &AU) const {
		ScopPass::getAnalysisUsage(AU);
		AU.setPreservesAll();
	}
};
bool SetRelationType::runOnScop(Scop &scop)
{
	S = &getCurScop();//&scop;

	for(Scop::iterator stmt_iterator = S->begin(), stmt_iterator_end = S->end();
	    stmt_iterator != stmt_iterator_end; stmt_iterator++) {

		ScopStmt * stmt = *stmt_iterator;
		assert(strcmp(stmt -> getBaseName(),"FinalRead"));

		for(ScopStmt::memacc_iterator access_iterator=stmt->memacc_begin(), access_iterator_end = stmt->memacc_end();
		    access_iterator != access_iterator_end; access_iterator++) {

			MemoryAccess* memoryAccess=*access_iterator;
			memoryAccess->setCurrentRelationType(relationType);
		}
	}

	return false;
}

char SetRelationType::ID = 0;
Pass* createSetRelationTypePass(MemoryAccess::RelationType relationType = MemoryAccess::RelationType_polly)
{
	return new SetRelationType(relationType);
}
namespace llvm
{
    class PassRegistry;
    void initializeSetRelationTypePass(llvm::PassRegistry&);
}

INITIALIZE_PASS_BEGIN(SetRelationType, "set-accesses-type",
                      "kernelgen's set current acceses type", false,
                      false)
INITIALIZE_PASS_DEPENDENCY(ScopInfo)
INITIALIZE_PASS_END(SetRelationType, "set-accesses-type",
                    "kernelgen's set current relation type", false,
                    false)
static RegisterPass<SetRelationType>
E("SetRelationType", "Kernelgen -  set current relation type");
