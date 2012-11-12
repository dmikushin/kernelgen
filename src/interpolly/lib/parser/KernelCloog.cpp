//===- Cloog.cpp - Cloog interface ----------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Cloog[1] interface.
//
// The Cloog interface takes a Scop and generates a Cloog AST (clast). This
// clast can either be returned directly or it can be pretty printed to stdout.
//
// A typical clast output looks like this:
//
// for (c2 = max(0, ceild(n + m, 2); c2 <= min(511, floord(5 * n, 3)); c2++) {
//   bb2(c2);
// }
//
// [1] http://www.cloog.org/ - The Chunky Loop Generator
//
//===----------------------------------------------------------------------===//

#include "KernelCloog.h"
#include "KernelScopInfo.h"

#define DEBUG_TYPE "polly-cloog"
#include "llvm/Assembly/Writer.h"
#include "llvm/Module.h"
#include "llvm/Support/Debug.h"

#include "cloog/isl/domain.h"
#include "cloog/isl/cloog.h"

#include <unistd.h>

using namespace llvm;

namespace kernelgen
{
class KernelCloog
{
	Scop *S;
	CloogOptions *Options;
	CloogState *State;
	clast_stmt *ClastRoot;

	void buildCloogOptions();
	CloogUnionDomain *buildCloogUnionDomain();
	CloogInput *buildCloogInput();

public:
	KernelCloog(Scop *Scop);

	~KernelCloog();

	/// Write a .cloog input file
	void dump(FILE *F);

	/// Print a source code representation of the program.
	void pprint(llvm::raw_ostream &OS);

	/// Create the Cloog AST from this program.
	struct clast_root *getClast();
};

KernelCloog::KernelCloog(Scop *Scop) : S(Scop)
{
	State = cloog_isl_state_malloc(Scop->getIslCtx());
	buildCloogOptions();
	ClastRoot = cloog_clast_create_from_input(buildCloogInput(), Options);
}

KernelCloog::~KernelCloog()
{
	cloog_clast_free(ClastRoot);
	cloog_options_free(Options);
	cloog_state_free(State);
}

// Create a FILE* write stream and get the output to it written
// to a std::string.
class FileToString
{
	int FD[2];
	FILE *input;
	static const int BUFFERSIZE = 20;

	char buf[BUFFERSIZE + 1];


public:
	FileToString() {
		pipe(FD);
		input = fdopen(FD[1], "w");
	}
	~FileToString() {
		close(FD[0]);
		//close(FD[1]);
	}

	FILE *getInputFile() {
		return input;
	}

	void closeInput() {
		fclose(input);
		close(FD[1]);
	}

	std::string getOutput() {
		std::string output;
		int readSize;

		while (true) {
			readSize = read(FD[0], &buf, BUFFERSIZE);

			if (readSize <= 0)
				break;

			output += std::string(buf, readSize);
		}


		return output;
	}

};

/// Write .cloog input file.
void KernelCloog::dump(FILE *F)
{
	CloogInput *Input = buildCloogInput();
	cloog_input_dump_cloog(F, Input, Options);
	cloog_input_free(Input);
}

/// Print a source code representation of the program.
void KernelCloog::pprint(raw_ostream &OS)
{
	FileToString *Output = new FileToString();
	clast_pprint(Output->getInputFile(), ClastRoot, 0, Options);
	Output->closeInput();
	OS << Output->getOutput();
	delete (Output);
}

/// Create the Cloog AST from this program.
struct clast_root *KernelCloog::getClast() {
	return (clast_root*)ClastRoot;
}

void KernelCloog::buildCloogOptions()
{
	Options = cloog_options_malloc(State);
	Options->quiet = 1;
	Options->strides = 1;
	Options->save_domains = 1;
	Options->noscalars = 1;

	//Options->otl = 1;

	Options->esp=1;
	Options->sh=1;

	// The last loop depth to optimize should be the last scattering dimension.
	// CLooG by default will continue to split the loops even after the last
	// scattering dimension. This splitting is problematic for the schedules
	// calculated by the PoCC/isl/Pluto optimizer. Such schedules contain may
	// not be fully defined, but statements without dependences may be mapped
	// to the same exeuction time. For such schedules, continuing to split
	// may lead to a larger set of if-conditions in the innermost loop.
	Options->l = -1;
	Options->f = -1;

}

CloogUnionDomain *KernelCloog::buildCloogUnionDomain()
{
	CloogUnionDomain *DU = cloog_union_domain_alloc(S->getNumParams());

	for (Scop::iterator SI = S->begin(), SE = S->end(); SI != SE; ++SI) {
		ScopStmt *Stmt = *SI;
		CloogScattering *Scattering;
		CloogDomain *Domain;

		Scattering = cloog_scattering_from_isl_map(Stmt->getScattering());
		Domain  = cloog_domain_from_isl_set(Stmt->getDomain());

		std::string entryName = Stmt->getBaseName();

		DU = cloog_union_domain_add_domain(DU, entryName.c_str(), Domain,
		                                   Scattering, Stmt);
	}

	return DU;
}

CloogInput *KernelCloog::buildCloogInput()
{
	CloogDomain *Context = cloog_domain_from_isl_set(S->getContext());
	CloogUnionDomain *Statements = buildCloogUnionDomain();

	isl_set *ScopContext = S->getContext();

	for (unsigned i = 0; i < isl_set_dim(ScopContext, isl_dim_param); i++) {
		isl_id *id = isl_set_get_dim_id(ScopContext, isl_dim_param, i);
		Statements = cloog_union_domain_set_name(Statements, CLOOG_PARAM, i,
		             isl_id_get_name(id));
		isl_id_free(id);
	}

	isl_set_free(ScopContext);

	CloogInput *Input = cloog_input_alloc(Context, Statements);
	return Input;
}

struct CloogExporter : public KernelScopPass {
	static char ID;
	Scop *S;
	explicit CloogExporter() : KernelScopPass(ID) {}

	std::string getFileName(Function *F) const;
	virtual bool runOnScop(Scop &S);
	void getAnalysisUsage(AnalysisUsage &AU) const;
};

std::string CloogExporter::getFileName(Function *F) const
{
	std::string FunctionName = F->getName();
	std::string ExitName, EntryName;

	raw_string_ostream ExitStr(ExitName);
	raw_string_ostream EntryStr(EntryName);

	WriteAsOperand(EntryStr, &F->getEntryBlock(), false);
	EntryStr.str();

	//if (R->getExit()) {
	//  WriteAsOperand(ExitStr, R->getExit(), false);
	//  ExitStr.str();
	//} else
	ExitName = "FunctionExit";

	std::string RegionName = EntryName + "---" + ExitName;
	std::string FileName = FunctionName + "___" + RegionName + ".cloog";

	return FileName;
}

char CloogExporter::ID = 0;
bool CloogExporter::runOnScop(Scop &S)
{
	KernelCloogInfo &C = getAnalysis<KernelCloogInfo>();

	std::string FunctionName = S.rootFunction->getName();
	std::string Filename = getFileName(S.rootFunction);

	errs() << "Writing Scop '" << S.nameStr << "' in function '"
	       << FunctionName << "' to '" << Filename << "'...\n";

	FILE *F = fopen(Filename.c_str(), "w");
	C.dump(F);
	fclose(F);

	return false;
}

void CloogExporter::getAnalysisUsage(AnalysisUsage &AU) const
{
	// Get the Common analysis usage of ScopPasses.
	KernelScopPass::getAnalysisUsage(AU);
	AU.addRequired<KernelCloogInfo>();
}



/// Write a .cloog input file
void KernelCloogInfo::dump(FILE *F)
{
	C->dump(F);
}

/// Print a source code representation of the program.
void KernelCloogInfo::pprint(llvm::raw_ostream &OS)
{
	C->pprint(OS);
}

/// Create the Cloog AST from this program.
const struct clast_root *KernelCloogInfo::getClast() {
	return C->getClast();
}

void KernelCloogInfo::releaseMemory()
{
	if (C) {
		delete C;
		C = 0;
	}
}

bool KernelCloogInfo::runOnScop(Scop &S)
{
	if (C)
		delete C;

	scop = &S;

	C = new KernelCloog(&S);

	Function *F = S.rootFunction;

	DEBUG(dbgs() << ":: " << F->getName());
	DEBUG(dbgs() << " : " << S.nameStr << "\n");;
	DEBUG(C->pprint(dbgs()));

	return false;
}

void KernelCloogInfo::printScop(raw_ostream &OS) const
{
	Function *function = scop->rootFunction;

	OS << function->getName() << "():\n";

	C->pprint(OS);
}

void KernelCloogInfo::getAnalysisUsage(AnalysisUsage &AU) const
{
	// Get the Common analysis usage of ScopPasses.
	KernelScopPass::getAnalysisUsage(AU);
}
void KernelCloogInfo::dump()

	{
		pprint(dbgs());
	}

char KernelCloogInfo::ID = 0;
Pass *createKernelCloogInfoPass()
{
	return new KernelCloogInfo();
}

llvm::Pass *createCloogExporterPass()
{
	return new CloogExporter();
}

}

using namespace kernelgen;
INITIALIZE_PASS_BEGIN(KernelCloogInfo, "kernel-cloog",
                      "Execute kernel Cloog code generation", false, true)
INITIALIZE_PASS_DEPENDENCY(KernelScopInfo)
INITIALIZE_PASS_END(KernelCloogInfo, "kernel-cloog",
                    "Execute kernel Cloog code generation", false, true)


static struct A {
	A() {
		PassRegistry &Registry = *PassRegistry::getPassRegistry();
		initializeKernelCloogInfoPass(Registry);
		
	}
} ARegister;

static RegisterPass<CloogExporter> A("kernel-export-cloog",
                                     "kernel - Export the Cloog input file"
                                     " (Writes a .cloog file for each Scop)"
                                    );
