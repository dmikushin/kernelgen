#include <map>
#include <list>
#include <llvm/Function.h>
#include <llvm/GlobalVariable.h>
#include <llvm/GlobalAlias.h>
#include "LinkFunctionBody.h"
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
		   iter_end = dependencesList.end(); iter != iter_end; iter++)
		   {
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
    void dropAllReferences()
	{
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
	void eraseAllDependences()
	{
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
    void getAllDependencesForValue(llvm::GlobalValue * value, DepsByType & dependencesByType) {
		//прописать суда периодическую очистку зависимостей
		globalDependences.getAllDependencesForValue(value,dependencesByType);
		#ifdef CLEAR_DEPENDENCES 
		globalDependences.eraseAllDependences();
		#endif
	}
}


