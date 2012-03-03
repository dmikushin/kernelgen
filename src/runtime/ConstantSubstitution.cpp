#include "llvm/Target/TargetData.h"
#include "llvm/Instructions.h"
#include "llvm/Type.h"
#include "llvm/Function.h"
#include "llvm/Constants.h"
#include <set>
#include <map>
#include <iostream>

#include "runtime.h"

using namespace kernelgen;
using namespace llvm;

std::map<LoadInst *, int> LoadInstOffsets;

struct UseTreeNode {
	const Value * sourceValue;
	int currentOffset;
};

typedef std::set<UseTreeNode *>::iterator UseIterator;
typedef std::map<LoadInst *, int>::iterator MapIterator;

enum AllowedInstructions {
	GEP = 1, Load, Cast, Store
};

int isAllowedInstruction(const Value * const &someValue)
{
	const Instruction * const instruction = dyn_cast<Instruction>(someValue);
	if(!instruction) return false;
	if(dyn_cast<GetElementPtrInst>(instruction)) return GEP;
	if(dyn_cast<LoadInst>(instruction)) return Load;
	if(dyn_cast<StoreInst>(instruction)) return Store;
	if(dyn_cast<CastInst>(instruction)) return Cast;
	return 0;
}

void makeUseTreeForIntegerArgs(UseTreeNode * const &useNode, const TargetData * const &targetData)
{
	const Value * sourceValue = useNode->sourceValue;
	for(Value::const_use_iterator useIterator = sourceValue->use_begin(), useEnd = sourceValue->use_end();
	    useIterator != useEnd; useIterator++) {
		const Value * User = *useIterator;
		int inst;
		
		// FIXME :
		// assume that before using of arg it must be loaded by LoadInst from some GEP or BitCast
		// before LoadInst in each branch of UseTree must be only GEPs and BitCasts
		assert(inst = isAllowedInstruction(User));

		UseTreeNode * newUseNode;
		if(inst == GEP || inst == Cast) {
			newUseNode = new UseTreeNode();
			newUseNode->sourceValue = User;
			newUseNode->currentOffset = useNode->currentOffset;
		}
		switch(inst) {
		case GEP: {
			// GEP increase offset and change type
			// new Type is newUseNode->sourceValue->getType()
			const GetElementPtrInst * GEPInst = dyn_cast<GetElementPtrInst>(User);
			std::vector<Value *> Indices(GEPInst->idx_begin(),GEPInst->idx_end());
			newUseNode->currentOffset += targetData->getIndexedOffset(
			                                 useNode->sourceValue->getType(), // old type from which we make GEP
			                                 ArrayRef<Value *>(Indices)       // Indices of GEP
			                             );
			makeUseTreeForIntegerArgs(newUseNode, targetData);
		}
		break;
		case Cast:
		    // Cast change only Type
			// new Type is newUseNode->sourceValue->getType()
			makeUseTreeForIntegerArgs(newUseNode, targetData);
			break;
		case Load:
		    LoadInstOffsets[(LoadInst *)User] = useNode->currentOffset;
			//alignment?
			break;
		case Store:
			break;
		}
		if(inst == GEP || inst == Cast)
			delete newUseNode;
	}

}

void computeLoadInstOffsets(const Value * sourceArg,const TargetData * targetData)
{
	UseTreeNode *useTree = new UseTreeNode();
		useTree->currentOffset = 0;
	useTree->sourceValue = sourceArg;
	
	// Bypass Use Tree of argument
	// side effect - fill in LoadInstOffsets
	makeUseTreeForIntegerArgs(useTree,targetData);
	delete useTree;
}

void ConstantSubstitution(Function * func, void * args)
{
	LoadInstOffsets.clear();
	Function::arg_iterator AI = func->arg_begin();
	Value * sourceArg = (Value *)AI;
	
	//argument must be a pointer
	assert( sourceArg -> getType() -> isPointerTy() );
	TargetData * targetData = new TargetData(func->getParent());
	
	//compute offsets of args
	//each arg is LoadInst from some offset in structure of args
	computeLoadInstOffsets(sourceArg,targetData);

	if (verbose & KERNELGEN_VERBOSE_POLLYGEN)	
		std::cout << "    Integer args substituted: " << std::endl;
	for(MapIterator arg = LoadInstOffsets.begin(), argEnd = LoadInstOffsets.end();
	    arg != argEnd; arg++) {
		LoadInst * load = arg->first;
		Type * type = load->getType();
		
		// for each integer arg
		// replace uses of arg's LoadInst by Constant
		if(type->isIntegerTy()) {
			assert(targetData->getTypeStoreSize(type) <= 8);
			uint64_t value = 0;
			int offset = arg->second;
			memcpy(&value, ((char *)args) + offset, targetData->getTypeStoreSize(type));
			ConstantInt * constant = ConstantInt::get(cast<IntegerType>(type), value);
			load->replaceAllUsesWith(constant);
			load->eraseFromParent();
			if (verbose & KERNELGEN_VERBOSE_POLLYGEN)
				std::cout << "        offset = " << arg->second << ", value = " <<
					constant->getValue().toString(10,true) << std::endl;
		}
	}
	return;
}
