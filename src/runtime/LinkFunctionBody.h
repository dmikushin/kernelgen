#include "llvm/Function.h"
#include "llvm/Transforms/Utils/ValueMapper.h"

using namespace llvm;


namespace kernelgen {

/// LinkFunctionBody - Copy the source function over into the dest function and
/// fix up references to values.  Dest is an external function, and Src is not.
/// Also insert the used functions declarations.
void LinkFunctionBody(llvm::Function *Dst, llvm::Function *Src);
void LinkFunctionBody(Function *Dst, Function *Src, ValueToValueMapTy & ValueMap );
void linkFunctionWithAllDependendes(Function *srcFunction, Function * dstFunction);

typedef std::list<llvm::GlobalVariable *>::iterator variable_iter;
typedef std::list<llvm::Function *>::iterator function_iter;
typedef std::list<llvm::GlobalAlias *>::iterator alias_iter;

struct DepsByType {
	std::list<llvm::GlobalVariable *> variables;
	std::list<llvm::Function *> functions;
	std::list<llvm::GlobalAlias *> aliases;
	void push_back(llvm::GlobalValue* value) {

		switch(value->getValueID()) {
		case llvm::Value::FunctionVal:
			functions.push_back(llvm::cast<llvm::Function>(value));
			break;
		case llvm::Value::GlobalVariableVal:
			variables.push_back(llvm::cast<llvm::GlobalVariable>(value));
			break;
		case llvm::Value::GlobalAliasVal:
			aliases.push_back(llvm::cast<llvm::GlobalAlias>(value));
			break;
		}

	}
};

};


