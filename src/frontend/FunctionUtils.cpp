#include <llvm/Function.h>
#include <llvm/LLVMContext.h>
#include <llvm/DerivedTypes.h>
#include <llvm/ADT/ArrayRef.h>
#include <string>
#include <vector>
#include <llvm/Instructions.h>

using namespace llvm;
using namespace std;
namespace kernelgen
{
Function * transformToVoidFunction(Function * &functionToTranform)
{
	LLVMContext & context = getGlobalContext();
	Type *oldReturnType = functionToTranform->getReturnType();
	if(oldReturnType == Type::getVoidTy(context)) return functionToTranform;

	FunctionType *oldFunctionType = functionToTranform->getFunctionType();
	std::vector<Type*> argTypes;
	int numParams = oldFunctionType -> getNumParams();
	for(int i = 0; i< numParams; i++)
		argTypes.push_back(oldFunctionType->getParamType(i));

	FunctionType * newFunctionType = FunctionType::get(Type::getVoidTy(context), argTypes, false);
	Function * newFunction = Function::Create(newFunctionType,GlobalValue::InternalLinkage,
	                         "", functionToTranform->getParent());
	newFunction->takeName(functionToTranform);
	vector<BasicBlock*> Blocks;
	Function::BasicBlockListType & newBBList = newFunction -> getBasicBlockList();

	{
		for(Function::iterator BB = functionToTranform->begin(), BB_End = functionToTranform->end();
		    BB != BB_End; BB++) {
			Blocks.push_back(BB);
		}
		for(int i = 0; i < Blocks.size(); i++) {
			Blocks[i]->removeFromParent();
			newBBList.push_back(Blocks[i]);
			ReturnInst * TI ;
			if(TI = dyn_cast<ReturnInst>(Blocks[i]->getTerminator()) )
			{
			     TI->eraseFromParent();
				 ReturnInst *newRetInst = ReturnInst::Create(context,Blocks[i]);
			}
		}
	}
	
	for(Function::arg_iterator AI = functionToTranform->arg_begin(),
							   NewAI = newFunction->arg_begin(),
	                           AI_End = functionToTranform->arg_end();
	    AI != AI_End; AI++,NewAI++) {
		NewAI-> setName(AI->getName());
		AI -> replaceAllUsesWith(NewAI);
	}
	
	functionToTranform->dropAllReferences();
	functionToTranform->removeFromParent();
	//if(functionToTranform -> use_empty()) functionToTranform->eraseFromParent();
	//   else functionToTranform->removeFromParent();
	functionToTranform = newFunction;
	return newFunction;
}
};
