//===- lib/Linker/LinkModules.cpp - Module Linker Implementation ----------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the LLVM module linker.
//
//===----------------------------------------------------------------------===//

#include "llvm/Linker.h"
#include "llvm/Constants.h"
#include "llvm/DerivedTypes.h"
#include "llvm/Instructions.h"
#include "llvm/Module.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/Optional.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Transforms/Utils/Cloning.h"
#include "llvm/Transforms/Utils/ValueMapper.h"
#include "llvm-c/Linker.h"
#include <cctype>

#include "LinkFunctionBody.h"
//#include "runtime.h"
#include <iostream>

using namespace llvm;
using namespace kernelgen;
using namespace std;

//===----------------------------------------------------------------------===//
// TypeMap implementation.
//===----------------------------------------------------------------------===//

namespace {
class TypeMapTy : public ValueMapTypeRemapper {
  /// MappedTypes - This is a mapping from a source type to a destination type
  /// to use.
  DenseMap<Type*, Type*> MappedTypes;

  /// SpeculativeTypes - When checking to see if two subgraphs are isomorphic,
  /// we speculatively add types to MappedTypes, but keep track of them here in
  /// case we need to roll back.
  SmallVector<Type*, 16> SpeculativeTypes;
  
  /// SrcDefinitionsToResolve - This is a list of non-opaque structs in the
  /// source module that are mapped to an opaque struct in the destination
  /// module.
  SmallVector<StructType*, 16> SrcDefinitionsToResolve;
  
  /// DstResolvedOpaqueTypes - This is the set of opaque types in the
  /// destination modules who are getting a body from the source module.
  SmallPtrSet<StructType*, 16> DstResolvedOpaqueTypes;

public:
  /// addTypeMapping - Indicate that the specified type in the destination
  /// module is conceptually equivalent to the specified type in the source
  /// module.
  void addTypeMapping(Type *DstTy, Type *SrcTy);

  /// linkDefinedTypeBodies - Produce a body for an opaque type in the dest
  /// module from a type definition in the source module.
  void linkDefinedTypeBodies();
  
  /// get - Return the mapped type to use for the specified input type from the
  /// source module.
  Type *get(Type *SrcTy);

  FunctionType *get(FunctionType *T) {return cast<FunctionType>(get((Type*)T));}

  /// dump - Dump out the type map for debugging purposes.
  void dump() const {
    for (DenseMap<Type*, Type*>::const_iterator
           I = MappedTypes.begin(), E = MappedTypes.end(); I != E; ++I) {
      dbgs() << "TypeMap: ";
      I->first->dump();
      dbgs() << " => ";
      I->second->dump();
      dbgs() << '\n';
    }
  }

private:
  Type *getImpl(Type *T);
  /// remapType - Implement the ValueMapTypeRemapper interface.
  Type *remapType(Type *SrcTy) {
    return get(SrcTy);
  }
  
  bool areTypesIsomorphic(Type *DstTy, Type *SrcTy);
};
}

void TypeMapTy::addTypeMapping(Type *DstTy, Type *SrcTy) {
  Type *&Entry = MappedTypes[SrcTy];
  if (Entry) return;
  
  if (DstTy == SrcTy) {
    Entry = DstTy;
    return;
  }
  
  // Check to see if these types are recursively isomorphic and establish a
  // mapping between them if so.
  if (!areTypesIsomorphic(DstTy, SrcTy)) {
    // Oops, they aren't isomorphic.  Just discard this request by rolling out
    // any speculative mappings we've established.
    for (unsigned i = 0, e = SpeculativeTypes.size(); i != e; ++i)
      MappedTypes.erase(SpeculativeTypes[i]);
  }
  SpeculativeTypes.clear();
}

/// areTypesIsomorphic - Recursively walk this pair of types, returning true
/// if they are isomorphic, false if they are not.
bool TypeMapTy::areTypesIsomorphic(Type *DstTy, Type *SrcTy) {
  // Two types with differing kinds are clearly not isomorphic.
  if (DstTy->getTypeID() != SrcTy->getTypeID()) return false;

  // If we have an entry in the MappedTypes table, then we have our answer.
  Type *&Entry = MappedTypes[SrcTy];
  if (Entry)
    return Entry == DstTy;

  // Two identical types are clearly isomorphic.  Remember this
  // non-speculatively.
  if (DstTy == SrcTy) {
    Entry = DstTy;
    return true;
  }
  
  // Okay, we have two types with identical kinds that we haven't seen before.

  // If this is an opaque struct type, special case it.
  if (StructType *SSTy = dyn_cast<StructType>(SrcTy)) {
    // Mapping an opaque type to any struct, just keep the dest struct.
    if (SSTy->isOpaque()) {
      Entry = DstTy;
      SpeculativeTypes.push_back(SrcTy);
      return true;
    }

    // Mapping a non-opaque source type to an opaque dest.  If this is the first
    // type that we're mapping onto this destination type then we succeed.  Keep
    // the dest, but fill it in later.  This doesn't need to be speculative.  If
    // this is the second (different) type that we're trying to map onto the
    // same opaque type then we fail.
    if (cast<StructType>(DstTy)->isOpaque()) {
      // We can only map one source type onto the opaque destination type.
      if (!DstResolvedOpaqueTypes.insert(cast<StructType>(DstTy)))
        return false;
      SrcDefinitionsToResolve.push_back(SSTy);
      Entry = DstTy;
      return true;
    }
  }
  
  // If the number of subtypes disagree between the two types, then we fail.
  if (SrcTy->getNumContainedTypes() != DstTy->getNumContainedTypes())
    return false;
  
  // Fail if any of the extra properties (e.g. array size) of the type disagree.
  if (isa<IntegerType>(DstTy))
    return false;  // bitwidth disagrees.
  if (PointerType *PT = dyn_cast<PointerType>(DstTy)) {
    if (PT->getAddressSpace() != cast<PointerType>(SrcTy)->getAddressSpace())
      return false;
    
  } else if (FunctionType *FT = dyn_cast<FunctionType>(DstTy)) {
    if (FT->isVarArg() != cast<FunctionType>(SrcTy)->isVarArg())
      return false;
  } else if (StructType *DSTy = dyn_cast<StructType>(DstTy)) {
    StructType *SSTy = cast<StructType>(SrcTy);
    if (DSTy->isLiteral() != SSTy->isLiteral() ||
        DSTy->isPacked() != SSTy->isPacked())
      return false;
  } else if (ArrayType *DATy = dyn_cast<ArrayType>(DstTy)) {
    if (DATy->getNumElements() != cast<ArrayType>(SrcTy)->getNumElements())
      return false;
  } else if (VectorType *DVTy = dyn_cast<VectorType>(DstTy)) {
    if (DVTy->getNumElements() != cast<ArrayType>(SrcTy)->getNumElements())
      return false;
  }

  // Otherwise, we speculate that these two types will line up and recursively
  // check the subelements.
  Entry = DstTy;
  SpeculativeTypes.push_back(SrcTy);

  for (unsigned i = 0, e = SrcTy->getNumContainedTypes(); i != e; ++i)
    if (!areTypesIsomorphic(DstTy->getContainedType(i),
                            SrcTy->getContainedType(i)))
      return false;
  
  // If everything seems to have lined up, then everything is great.
  return true;
}

/// linkDefinedTypeBodies - Produce a body for an opaque type in the dest
/// module from a type definition in the source module.
void TypeMapTy::linkDefinedTypeBodies() {
  SmallVector<Type*, 16> Elements;
  SmallString<16> TmpName;
  
  // Note that processing entries in this loop (calling 'get') can add new
  // entries to the SrcDefinitionsToResolve vector.
  while (!SrcDefinitionsToResolve.empty()) {
    StructType *SrcSTy = SrcDefinitionsToResolve.pop_back_val();
    StructType *DstSTy = cast<StructType>(MappedTypes[SrcSTy]);
    
    // TypeMap is a many-to-one mapping, if there were multiple types that
    // provide a body for DstSTy then previous iterations of this loop may have
    // already handled it.  Just ignore this case.
    if (!DstSTy->isOpaque()) continue;
    assert(!SrcSTy->isOpaque() && "Not resolving a definition?");
    
    // Map the body of the source type over to a new body for the dest type.
    Elements.resize(SrcSTy->getNumElements());
    for (unsigned i = 0, e = Elements.size(); i != e; ++i)
      Elements[i] = getImpl(SrcSTy->getElementType(i));
    
    DstSTy->setBody(Elements, SrcSTy->isPacked());
    
    // If DstSTy has no name or has a longer name than STy, then viciously steal
    // STy's name.
    if (!SrcSTy->hasName()) continue;
    StringRef SrcName = SrcSTy->getName();
    
    if (!DstSTy->hasName() || DstSTy->getName().size() > SrcName.size()) {
      TmpName.insert(TmpName.end(), SrcName.begin(), SrcName.end());
      SrcSTy->setName("");
      DstSTy->setName(TmpName.str());
      TmpName.clear();
    }
  }
  
  DstResolvedOpaqueTypes.clear();
}

/// get - Return the mapped type to use for the specified input type from the
/// source module.
Type *TypeMapTy::get(Type *Ty) {
  Type *Result = getImpl(Ty);
  
  // If this caused a reference to any struct type, resolve it before returning.
  if (!SrcDefinitionsToResolve.empty())
    linkDefinedTypeBodies();
  return Result;
}

/// getImpl - This is the recursive version of get().
Type *TypeMapTy::getImpl(Type *Ty) {
  // If we already have an entry for this type, return it.
  Type **Entry = &MappedTypes[Ty];
  if (*Entry) return *Entry;
  
  // If this is not a named struct type, then just map all of the elements and
  // then rebuild the type from inside out.
  if (!isa<StructType>(Ty) || cast<StructType>(Ty)->isLiteral()) {
    // If there are no element types to map, then the type is itself.  This is
    // true for the anonymous {} struct, things like 'float', integers, etc.
    if (Ty->getNumContainedTypes() == 0)
      return *Entry = Ty;
    
    // Remap all of the elements, keeping track of whether any of them change.
    bool AnyChange = false;
    SmallVector<Type*, 4> ElementTypes;
    ElementTypes.resize(Ty->getNumContainedTypes());
    for (unsigned i = 0, e = Ty->getNumContainedTypes(); i != e; ++i) {
      ElementTypes[i] = getImpl(Ty->getContainedType(i));
      AnyChange |= ElementTypes[i] != Ty->getContainedType(i);
    }
    
    // If we found our type while recursively processing stuff, just use it.
    Entry = &MappedTypes[Ty];
    if (*Entry) return *Entry;
    
    // If all of the element types mapped directly over, then the type is usable
    // as-is.
    if (!AnyChange)
      return *Entry = Ty;
    
    // Otherwise, rebuild a modified type.
    switch (Ty->getTypeID()) {
    default: llvm_unreachable("unknown derived type to remap");
    case Type::ArrayTyID:
      return *Entry = ArrayType::get(ElementTypes[0],
                                     cast<ArrayType>(Ty)->getNumElements());
    case Type::VectorTyID: 
      return *Entry = VectorType::get(ElementTypes[0],
                                      cast<VectorType>(Ty)->getNumElements());
    case Type::PointerTyID:
      return *Entry = PointerType::get(ElementTypes[0],
                                      cast<PointerType>(Ty)->getAddressSpace());
    case Type::FunctionTyID:
      return *Entry = FunctionType::get(ElementTypes[0],
                                        makeArrayRef(ElementTypes).slice(1),
                                        cast<FunctionType>(Ty)->isVarArg());
    case Type::StructTyID:
      // Note that this is only reached for anonymous structs.
      return *Entry = StructType::get(Ty->getContext(), ElementTypes,
                                      cast<StructType>(Ty)->isPacked());
    }
  }

  // Otherwise, this is an unmapped named struct.  If the struct can be directly
  // mapped over, just use it as-is.  This happens in a case when the linked-in
  // module has something like:
  //   %T = type {%T*, i32}
  //   @GV = global %T* null
  // where T does not exist at all in the destination module.
  //
  // The other case we watch for is when the type is not in the destination
  // module, but that it has to be rebuilt because it refers to something that
  // is already mapped.  For example, if the destination module has:
  //  %A = type { i32 }
  // and the source module has something like
  //  %A' = type { i32 }
  //  %B = type { %A'* }
  //  @GV = global %B* null
  // then we want to create a new type: "%B = type { %A*}" and have it take the
  // pristine "%B" name from the source module.
  //
  // To determine which case this is, we have to recursively walk the type graph
  // speculating that we'll be able to reuse it unmodified.  Only if this is
  // safe would we map the entire thing over.  Because this is an optimization,
  // and is not required for the prettiness of the linked module, we just skip
  // it and always rebuild a type here.
  StructType *STy = cast<StructType>(Ty);
  
  // If the type is opaque, we can just use it directly.
  if (STy->isOpaque())
    return *Entry = STy;
  
  // Otherwise we create a new type and resolve its body later.  This will be
  // resolved by the top level of get().
  SrcDefinitionsToResolve.push_back(STy);
  StructType *DTy = StructType::create(STy->getContext());
  DstResolvedOpaqueTypes.insert(DTy);
  return *Entry = DTy;
}

/// LinkFunctionBody - Copy the source function over into the dest function and
/// fix up references to values.  Dest is an external function, and Src is not.
/// Also insert the used functions declarations.
void kernelgen::LinkFunctionBody(Function *Dst, Function *Src) {
  assert(Src && Dst && Dst->isDeclaration() && !Src->isDeclaration());

  // ValueMap - Mapping of values from what they used to be in Src, to what
  // they are now in DstM.  ValueToValueMapTy is a ValueMap, which involves
  // some overhead due to the use of Value handles which the Linker doesn't
  // actually need, but this allows us to reuse the ValueMapper code.
  ValueToValueMapTy ValueMap;
  
  TypeMapTy TypeMap; 

  // Go through and convert function arguments over, remembering the mapping.
  Function::arg_iterator DI = Dst->arg_begin();
  for (Function::arg_iterator I = Src->arg_begin(), E = Src->arg_end();
       I != E; ++I, ++DI) {
    DI->setName(I->getName());  // Copy the name over.

    // Add a mapping to our mapping.
    ValueMap[I] = DI;
  }

  // Clone the body of the function into the dest function.
  SmallVector<ReturnInst*, 8> Returns; // Ignore returns.
  CloneFunctionInto(Dst, Src, ValueMap, false, Returns, "", NULL, &TypeMap);
  
  // There is no need to map the arguments anymore.
  for (Function::arg_iterator I = Src->arg_begin(), E = Src->arg_end();
       I != E; ++I)
    ValueMap.erase(I);

  // Recursively link declarations of functions used by the one just inserted.
  for (Function::iterator bb = Dst->begin(), be = Dst->end(); bb != be; bb++)
    for (BasicBlock::iterator ii = bb->begin(), ie = bb->end(); ii != ie; ii++) {
     
       // Check if instruction in focus is a call.
      CallInst* call = dyn_cast<CallInst>(cast<Value>(ii));
      if (!call) continue;
      
    	Function* DstR = dyn_cast<Function>(
							call->getCalledValue()->stripPointerCasts());
      if (!DstR) continue;
      
	  // function already handled
	  if (DstR->getParent() == Dst->getParent()) continue;
	  
      Function* SrcR = Src->getParent()->getFunction(DstR->getName());
      if (!SrcR) continue;
	  assert(DstR == SrcR);
	  
      //Function* DstRNew = Function::Create(TypeMap.get(SrcR->getFunctionType()),
      //	SrcR->getLinkage(), SrcR->getName(), Dst->getParent());
      Function* DstRNew =  dyn_cast<Function>(Dst->getParent()->getOrInsertFunction(
	            SrcR->getName(), SrcR->getFunctionType(),SrcR->getAttributes()) -> stripPointerCasts());

	  if(!DstRNew) continue; // something wrong
	  else
		  //link if  DstRNew is not defined (has not body)
		  if(!SrcR->isDeclaration() && DstRNew->isDeclaration())
	  {
         // Use the maximum alignment, rather than just copying the alignment of SrcGV.
         unsigned Alignment = std::max(DstRNew->getAlignment(), DstR->getAlignment());
         DstRNew->setAlignment(Alignment);
		 LinkFunctionBody(DstRNew, SrcR);
      }
	  
	  // replace called function by DstRNew
		Type *calledFunctionType = call->getCalledValue()->getType();
        if(calledFunctionType == DstRNew -> getType())
     	   call -> setCalledFunction(DstRNew);
	    else
		{
		   cout << "Nested device call: " << DstRNew->getName().data()<< " need bitcast "<<endl;
           call -> setCalledFunction( ConstantExpr::getBitCast(DstRNew,calledFunctionType));
		}
      //if (verbose) cout << "Nested device call: " << DstRNew->getName().data()<< endl;
  }
}

