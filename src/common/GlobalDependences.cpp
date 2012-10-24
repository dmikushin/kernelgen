#include <map>
#include <list>
#include <llvm/Function.h>
#include <llvm/GlobalVariable.h>
#include <llvm/GlobalAlias.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/Instructions.h>
#include <llvm/Module.h>
#include <llvm/Transforms/Utils/Cloning.h>

using namespace llvm;
using namespace kernelgen;
//struct globalDepsMap { typedef std::map<T *, globalDeps> Type;};

class GlobalDependences;
//void GlobalDependences::getAllDependencesForValue(llvm::GlobalValue * value, DepsByType & dependencesByType);

class GlobalValueInfo
{
public:
	static GlobalDependences *globalDependences;
	static void initializeWithGlobalDependences(GlobalDependences *_globalDependences) {
		assert(globalDependences == NULL);
		globalDependences = _globalDependences;
	}
	static void finalizeWithGlobalDependences(GlobalDependences *_globalDependences) {
		assert(globalDependences == _globalDependences);
		globalDependences = NULL;
	}

	bool handled;
	llvm::GlobalValue * value;
	bool collected;
	int numOfReferences;

	// maybe replace be set
	std::list<GlobalValueInfo *> dependences;
	typedef std::list<GlobalValueInfo *>::iterator deps_iterator;
	int getNumOfReferences() {
		return numOfReferences;
	}
	void setGlobalValue(llvm::GlobalValue * _value) {
		value = _value;
	}
	llvm::GlobalValue * getGlobalValue() {
		return value;
	}


	template<typename UserType>
	void processUser(UserType *user);

//template<typename ConstantType>
//void processConstant(ConstantType *constant);
	void processConstant(Constant *constant);

	GlobalValueInfo()
		:handled(false), value(NULL), collected(false),numOfReferences(0) { }


	void handleDependences();

	void recursiveCollectAllDependences(std::list<GlobalValueInfo *> & dependencesList) {
		assert(value);
		if(!collected) {
			if(!handled)
				handleDependences();

			dependencesList.push_back(this);
			collected=true;

			for(deps_iterator iter = dependences.begin(), iter_end = dependences.end();
			    iter!=iter_end; iter++)
				(*iter)->recursiveCollectAllDependences(dependencesList);
		}
	}

	~GlobalValueInfo() {
		dropAllReferences();
		assert(numOfReferences==0);
	}

	GlobalValueInfo *obtain() {
		numOfReferences++;
		return this;
	}

	void release() {
		numOfReferences--;
		assert(numOfReferences >= 0);
	}

	void dropAllReferences() {
		for(deps_iterator iter = dependences.begin(), iter_end = dependences.end();
		    iter!= iter_end; iter++)
			(*iter)->release();
		dependences.clear();
		handled=false;
	}
};
GlobalDependences *GlobalValueInfo::globalDependences;

template <typename T> struct GlobalDependencesType {
	typedef std::map<T *, GlobalValueInfo> type;
};

class GlobalDependences
{
	friend class GlobalValueInfo;

	typedef GlobalDependencesType<llvm::GlobalValue>::type FunctionsDeps;
	typedef GlobalDependencesType<llvm::GlobalValue>::type VariablesDeps;
	typedef GlobalDependencesType<llvm::GlobalValue>::type AliasesDeps;

	FunctionsDeps functionsDeps;
	VariablesDeps variablesDeps;
	AliasesDeps aliasesDeps;

private:
	GlobalValueInfo *getInfoForGlobalValue(llvm::GlobalValue *value) {
		GlobalValueInfo *tmp;
		switch(value->getValueID()) {
		case llvm::Value::FunctionVal:
			tmp = &functionsDeps[value];
			break;
		case llvm::Value::GlobalVariableVal:
			tmp = &variablesDeps[value];
			break;
		case llvm::Value::GlobalAliasVal:
			tmp = &aliasesDeps[value];
			break;
		}
		//if new object
		if(tmp->getGlobalValue() == NULL)
			tmp->setGlobalValue(value);
		return tmp->obtain();
	}

public:
	void getAllDependencesForValue(llvm::GlobalValue * value, DepsByType & dependencesByType) {
		std::list<GlobalValueInfo *> dependencesList;
		GlobalValueInfo *info= getInfoForGlobalValue(value);
		info->recursiveCollectAllDependences(dependencesList);
		for(std::list<GlobalValueInfo *>::iterator iter = dependencesList.begin(),
		    iter_end = dependencesList.end(); iter != iter_end; iter++) {
			(*iter)->collected = false;
			dependencesByType.push_back((*iter)->value);
		}
		info->release();
		switch(value->getValueID()) {
		case llvm::Value::FunctionVal:
			functionsDeps.erase(value);
			break;
		case llvm::Value::GlobalVariableVal:
			variablesDeps.erase(value);
			break;
		case llvm::Value::GlobalAliasVal:
			aliasesDeps.erase(value);
			break;
		}

	}
	void dropAllReferences() {
		for(FunctionsDeps::iterator iter = functionsDeps.begin(),iter_end = functionsDeps.end();
		    iter!=iter_end; iter++)
			iter->second.dropAllReferences();

		for(VariablesDeps::iterator iter = variablesDeps.begin(),iter_end = variablesDeps.end();
		    iter!=iter_end; iter++)
			iter->second.dropAllReferences();

		for(AliasesDeps::iterator iter = aliasesDeps.begin(),iter_end = aliasesDeps.end();
		    iter!=iter_end; iter++)
			iter->second.dropAllReferences();
	}
	GlobalDependences() {
		GlobalValueInfo::initializeWithGlobalDependences(this);
	}
	~GlobalDependences() {
		dropAllReferences();
		GlobalValueInfo::finalizeWithGlobalDependences(this);
		// all maps will be deleted automatically
	}
	void eraseAllDependences() {
		dropAllReferences();
		functionsDeps.clear();
		variablesDeps.clear();
		aliasesDeps.clear();
	}
	//удаление элементов не дописал
};


template<typename UserType>
void GlobalValueInfo::processUser(UserType *user)
{
	int numberOfOperands = user->getNumOperands();
	for(int i =0; i < numberOfOperands; i++) {
		User *operandValue = dyn_cast<User>(user->getOperand(i));
		if(!operandValue) continue;

		if(isa<GlobalValue>(*operandValue))
			dependences.push_back(globalDependences->getInfoForGlobalValue(cast<GlobalValue>(operandValue)));
		else if(isa<Constant>(*operandValue)) {
			std::list<User *> stack;
			stack.push_back(operandValue);
			while(!stack.empty()) {
				User *current = stack.back();
				stack.pop_back();

				int numOfCurrentOperands = current->getNumOperands();
				for(int operandIndex = 0; operandIndex < numOfCurrentOperands; operandIndex++) {

					operandValue = dyn_cast<User>(current->getOperand(operandIndex));
					if(!operandValue) continue;

					if(isa<GlobalValue>(*operandValue)) {
						dependences.push_back(globalDependences->getInfoForGlobalValue(cast<GlobalValue>(operandValue)));
					} else if(isa<User>(*operandValue)) {
						if(std::find(stack.begin(),stack.end(), operandValue) == stack.end() &&
						   operandValue != current)
							stack.push_back(cast<User>(operandValue));
					}
				}
			}
		}
	}
}

void GlobalValueInfo::handleDependences()
{
	switch(value->getValueID()) {
	case llvm::Value::FunctionVal: {
		Function *f = cast<Function>(value);
		for (Function::iterator bb = f->begin(), be = f->end(); bb != be; bb++)
			for (BasicBlock::iterator ii = bb->begin(), ie = bb->end(); ii != ie; ii++)
				processUser(cast<Instruction>(ii));
	}
	break;
	case llvm::Value::GlobalVariableVal:
		processUser(cast<GlobalVariable>(value));
		break;
	case llvm::Value::GlobalAliasVal:
		assert(false);
		break;
	}
	handled = true;
}


static GlobalDependences globalDependences;
#define CLEAR_DEPENDENCES
namespace kernelgen
{
void getAllDependencesForValue(llvm::GlobalValue * value, DepsByType & dependencesByType)
{
	//прописать сюда периодическую очистку зависимостей
	globalDependences.getAllDependencesForValue(value,dependencesByType);
#ifdef CLEAR_DEPENDENCES
	globalDependences.eraseAllDependences();
#endif
}

void linkFunctionWithAllDependendes(Function *srcFunction, Function * dstFunction)
{
	Module *dstModule = dstFunction->getParent();

	DepsByType dependences;
	getAllDependencesForValue(srcFunction, dependences);

	// Map values from composite to new module
	ValueToValueMapTy VMap;

	// Loop over all of the global variables, making corresponding
	// globals in the new module.  Here we add them to the VMap and
	// to the new Module.  We don't worry about attributes or initializers,
	// they will come later.
	for (variable_iter iter = dependences.variables.begin(),
	     iter_end = dependences.variables.end(); iter != iter_end; ++iter) {
		GlobalVariable *I = *iter;

		GlobalVariable *GV = dstModule->getNamedGlobal(I->getName());

		if(!GV)
			GV = new GlobalVariable(*dstModule,
			                        I->getType()->getElementType(),
			                        I->isConstant(), I->getLinkage(),
			                        (Constant*) 0, I->getName(),
			                        (GlobalVariable*) 0,
			                        I->isThreadLocal(),
			                        I->getType()->getAddressSpace());

		GV->copyAttributesFrom(I);
		VMap[I] = GV;
	}

	// Loop over the functions in the module, making external functions as before.
	for (function_iter iter = dependences.functions.begin(),
	     iter_end = dependences.functions.end(); iter != iter_end; ++iter) {

		Function *I = *iter;
		Function *NF = NULL;
		if(I != srcFunction) {

			Function *existedFunction = dstModule->getFunction(I->getName());
			if(existedFunction && existedFunction->isIntrinsic())
				NF = existedFunction;
			else
				NF = Function::Create(
				         cast<FunctionType>(I->getType()->getElementType()),
				         I->getLinkage(), I->getName(), dstModule);

		} else
			NF = dstFunction;
		NF->copyAttributesFrom(I);
		VMap[I] = NF;
	}

	// Loop over the aliases in the module.
	for (alias_iter iter = dependences.aliases.begin(),
	     iter_end = dependences.aliases.end(); iter != iter_end; ++iter) {
		// It is wery interesting what is alias
		assert(false);
	}

	// Now that all of the things that global variable initializer can refer to
	// have been created, loop through and copy the global variable referrers
	// over...  We also set the attributes on the global now.
	for (variable_iter iter = dependences.variables.begin(),
	     iter_end = dependences.variables.end(); iter != iter_end; ++iter) {
		GlobalVariable *I = *iter;

		GlobalVariable *GV = cast<GlobalVariable>(VMap[I]);
		if (I->hasInitializer())
			GV->setInitializer(MapValue(I->getInitializer(), VMap));
	}

	// Similarly, copy over required function bodies now...
	for (function_iter iter = dependences.functions.begin(),
	     iter_end = dependences.functions.end(); iter != iter_end; ++iter) {
		Function *I = *iter;
		Function *F = cast<Function>(VMap[I]);

		if (!I->isDeclaration()) {
			Function::arg_iterator DestI = F->arg_begin();
			for (Function::const_arg_iterator J = I->arg_begin();
			     J != I->arg_end(); ++J) {
				DestI->setName(J->getName());
				VMap[J] = DestI++;
			}

			SmallVector<ReturnInst*, 8> Returns;  // Ignore returns cloned.
			CloneFunctionInto(F, I, VMap, /*ModuleLevelChanges=*/true, Returns);

			for (Function::arg_iterator argI = I->arg_begin(),
			     argE = I->arg_end(); argI != argE; ++argI)
				VMap.erase(argI);
		}
	}
}
}
