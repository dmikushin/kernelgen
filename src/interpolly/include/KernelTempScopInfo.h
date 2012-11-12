
#include "KernelVerification.h"
#include "polly/LinkAllPasses.h"
#include "llvm/Analysis/AliasAnalysis.h"
#include "llvm/Metadata.h"
#include "llvm/Module.h"
#include "llvm/Support/MDBuilder.h"

namespace kernelgen
{
class IRAccess
{
public:
	const Value *BaseAddress;

	const SCEV *Offset;

	// The type of the scev affine function
	enum TypeKind { READ, WRITE };

private:
	unsigned ElemBytes;
	TypeKind Type;
	bool IsAffine;

public:
	explicit IRAccess (TypeKind Type, const Value *BaseAddress,
	                   const SCEV *Offset, unsigned elemBytes, bool Affine)
		: BaseAddress(BaseAddress), Offset(Offset),
		  ElemBytes(elemBytes), Type(Type), IsAffine(Affine) {}

	enum TypeKind getType() const {
		return Type;
	}

	const Value *getBase() const {
		return BaseAddress;
	}

	const SCEV *getOffset() const {
		return Offset;
	}

	unsigned getElemSizeInBytes() const {
		return ElemBytes;
	}

	bool isAffine() const {
		return IsAffine;
	}

	bool isRead() const {
		return Type == READ;
	}

	void print(raw_ostream &OS) const {
		WriteAsOperand(OS, BaseAddress, false);
		OS  << "[ " << *Offset << " ]";
	}
	void dump() const {
		print(dbgs());
	}
	raw_ostream &operator<<(raw_ostream &OS) const {
		print(OS);
		return OS;
	}

};

class Comparison
{

	const SCEV *LHS;
	const SCEV *RHS;

	ICmpInst::Predicate Pred;

public:
	Comparison(const SCEV *LHS, const SCEV *RHS, ICmpInst::Predicate Pred)
		: LHS(LHS), RHS(RHS), Pred(Pred) {}

	const SCEV *getLHS() const {
		return LHS;
	}
	const SCEV *getRHS() const {
		return RHS;
	}

	ICmpInst::Predicate getPred() const {
		return Pred;
	}
	void print(raw_ostream &OS) const {
		OS << "( " <<*LHS << " vs " << *RHS << " )";
	}
	void dump() const {
		print(dbgs());
	}
	raw_ostream &operator<<(raw_ostream &OS) const {
		print(OS);
		return OS;
	}

};

//===---------------------------------------------------------------------===//
/// Types
// The condition of a Basicblock, combine brcond with "And" operator.
typedef SmallVector<Comparison, 4> BBCond;

/// Maps from a loop to the affine function expressing its backedge taken count.
/// The backedge taken count already enough to express iteration domain as we
/// only allow loops with canonical induction variable.
/// A canonical induction variable is:
/// an integer recurrence that starts at 0 and increments by one each time
/// through the loop.
typedef std::map<const Loop*, const SCEV*> LoopBoundMapType;

/// Mapping BBs to its condition constrains
typedef std::map<const BasicBlock*, BBCond> BBCondMapType;

typedef std::vector<std::pair<IRAccess, Instruction*> > AccFuncSetType;
typedef std::map<const BasicBlock*, AccFuncSetType> AccFuncMapType;


//===---------------------------------------------------------------------===//
/// @brief Scop represent with llvm objects.
///
/// A helper class for remembering the parameter number and the max depth in
/// this Scop, and others context.
class TempScop
{
public:
	// The Region.
	Function &kernelFunction;

	// The max loop depth of this Scop
	unsigned MaxLoopDepth;
public:
	struct  TempScopPart {
		LoopBoundMapType LoopBounds;
		BBCondMapType BBConds;
		AccFuncMapType AccFuncMap;
		SCEVTreeNode *SCEVTreeLeaf;
		TempScop &tempScop;
		TempScopPart(SCEVTreeNode *SCEVTreeLeaf, TempScop &tempScop)
			:LoopBounds(), BBConds(), AccFuncMap(),SCEVTreeLeaf(SCEVTreeLeaf), tempScop(tempScop) {}


		/// @brief Get the loop bounds of the given loop.
		///
		/// @param L The loop to get the bounds.
		///
		/// @return The bounds of the loop L in { Lower bound, Upper bound } form.
		///
		const SCEV *getLoopBound(const Loop *L) const {

			const LoopBoundMapType *boundsFromParentFunc = &LoopBounds;
			LoopBoundMapType::const_iterator at = boundsFromParentFunc->find(L);
			SCEVTreeNode *leaf = SCEVTreeLeaf;

			while( at == boundsFromParentFunc->end() && leaf->parentNode!=NULL) {
				leaf = leaf->parentNode;
				boundsFromParentFunc = &(tempScop[leaf].LoopBounds);
				at = boundsFromParentFunc->find(L);
			}

			assert(at != boundsFromParentFunc->end() && "Only valid loop is allow!");
			return at->second;
		}

		/// @brief Get the condition from entry block of the Scop to a BasicBlock
		///
		/// @param BB The BasicBlock
		///
		/// @return The condition from entry block of the Scop to a BB
		///
		const BBCond *getBBCond(const BasicBlock *BB) const {
			BBCondMapType::const_iterator at = BBConds.find(BB);
			assert(at != BBConds.end() && "Only valid loop is allow!");
			return &(at->second);
		}

		/// @brief Get all access functions in a BasicBlock
		///
		/// @param  BB The BasicBlock that containing the access functions.
		///
		/// @return All access functions in BB
		///
		const AccFuncSetType *getAccessFunctions(const BasicBlock* BB) const {
			AccFuncMapType::const_iterator at = AccFuncMap.find(BB);
			return at != AccFuncMap.end()? &(at->second) : 0;
		}
		//@}

		void printConditions(raw_ostream &OS,const BBCond &conditions, const BasicBlock *bb)const {
			OS << SCEVTreeLeaf->f->getName() << ":";
			WriteAsOperand(OS, bb, false);
			OS << ":cond:";
			if(conditions.size() != 0) {
				conditions[0].print(OS);
				for(unsigned i = 1; i < conditions.size(); i++) {
					OS << "&&";
					conditions[i].print(OS);
				}
			}
		}
		void printAccessFunctions(raw_ostream &OS, const AccFuncSetType & accesFunctions, const BasicBlock *bb) const {
			if(accesFunctions.size() != 0) {

				OS << SCEVTreeLeaf->f->getName() << ":";
				WriteAsOperand(OS, bb, false);
				OS << ":acc:";

				OS << *accesFunctions[0].second  << "-->" ;
				accesFunctions[0].first.print(OS);
				for(unsigned i =1; i < accesFunctions.size(); i++) {
					OS << "\n";

					OS << SCEVTreeLeaf->f->getName() << ":";
					WriteAsOperand(OS, bb, false);
					OS << ":acc:";

					OS << *accesFunctions[i].second  << "-->" ;
					accesFunctions[i].first.print(OS);
				}
			}
		}
		void print(raw_ostream &OS) const {
			AccFuncMapType::const_iterator iter = AccFuncMap.begin();
			for(unsigned i =0; i < AccFuncMap.size(); i++) {
				const BasicBlock *BB = iter->first;
				printConditions(OS, BBConds.find(BB)->second, BB);
				OS << "\n";
				printAccessFunctions(OS,iter->second,BB);
				OS << "\n";
				iter++;
			}
		}

		void dump() const {
			print(dbgs());
		}


	};

	std::map<SCEVTreeNode *, TempScopPart *> parts;

	friend class KernelTempScopInfo;

	explicit TempScop(Function &f)
		: kernelFunction(f), MaxLoopDepth(0), parts() {}

public:
	~TempScop() {
		//LoopBounds.clear();
		//BBConds.clear();
		//AccFuncMap.clear();
		parts.clear();
	}
	TempScopPart &operator[](SCEVTreeNode *part) {
		std::map<SCEVTreeNode *, TempScopPart *>::iterator at = parts.find(part);
		if(at == parts.end())
			parts[part]=new TempScopPart(part,*this);

		return *(parts[part]);
	}
	/// @brief Get the maximum Region contained by this Scop.
	///
	/// @return The maximum Region contained by this Scop.
	Function &getKernelFunction() const {
		return kernelFunction;
	}

	/// @brief Get the maximum loop depth of Region R.
	///
	/// @return The maximum loop depth of Region R.
	unsigned getMaxLoopDepth() const {
		return MaxLoopDepth;
	}

	void print(raw_ostream &OS) const {
		for(std::map<SCEVTreeNode *, TempScopPart *>::const_iterator iter = parts.begin(),
		    iterEnd = parts.end(); iter!=iterEnd; iter++) {
			iter->second->print(OS);
		}
	}

	void dump() const {
		print(dbgs());
	}

	/// @brief Print the Temporary Scop information.
	///
	/// @param OS The output stream the access functions is printed to.
	/// @param SE The ScalarEvolution that help printing Temporary Scop
	///           information.
	/// @param LI The LoopInfo that help printing the access functions.
	void print(raw_ostream &OS, ScalarEvolution *SE, LoopInfo *LI) const;

	/// @brief Print the access functions and loop bounds in this Scop.
	///
	/// @param OS The output stream the access functions is printed to.
	/// @param SE The ScalarEvolution that help printing the access functions.
	/// @param LI The LoopInfo that help printing the access functions.
	void printDetail(raw_ostream &OS, ScalarEvolution *SE,
	                 LoopInfo *LI, const Region *Reg, unsigned ind) const;
};

typedef std::map<const Region*, TempScop *> TempScopMapType;

//===----------------------------------------------------------------------===//
/// @brief The Function Pass to extract temporary information for Static control
///        part in llvm function.
///

class KernelTempScopInfo : public FunctionPass
{
public:
	//===-------------------------------------------------------------------===//
	// DO NOT IMPLEMENT
	KernelTempScopInfo(const KernelTempScopInfo &);
	// DO NOT IMPLEMENT
	const KernelTempScopInfo &operator=(const KernelTempScopInfo &);

	Function *f;
	SCEVTreeNode *SCEVTreeLeaf;
	DominatorTree *DT;
	PostDominatorTree *PDT;
	TempScop *tempScop;


	static char ID;

	TargetData *TD;
	KernelVerification *KV;

	std::vector<CallInst *> callInstructions;



	explicit KernelTempScopInfo()
		: FunctionPass(ID), SCEVTreeLeaf(NULL), tempScop(NULL),
		  KV(NULL),requireVerification(true), f(NULL) {}

	explicit KernelTempScopInfo(TempScop *tempScop, SCEVTreeNode *SCEVTreeLeaf)
		: FunctionPass(ID), SCEVTreeLeaf(SCEVTreeLeaf),
		  tempScop(tempScop), KV(NULL),requireVerification(false),f(NULL) {}

	void buildAccessFunctions(BasicBlock *basicBlock, TempScop::TempScopPart &tempScopPart) {

		AccFuncSetType Functions;// = tempScopPart.AccFuncMap[basicBlock];


		for (BasicBlock::iterator I = basicBlock->begin(), E = --basicBlock->end(); I != E; ++I) {
			Instruction &Inst = *I;
			if (isa<LoadInst>(&Inst) || isa<StoreInst>(&Inst)) {
				unsigned Size;
				enum IRAccess::TypeKind Type;

				if (LoadInst *Load = dyn_cast<LoadInst>(&Inst)) {
					Size = TD->getTypeStoreSize(Load->getType());
					Type = IRAccess::READ;
				} else {
					StoreInst *Store = cast<StoreInst>(&Inst);
					Size = TD->getTypeStoreSize(Store->getValueOperand()->getType());
					Type = IRAccess::WRITE;
				}

				const SCEV *AccessFunction = SCEVTreeLeaf->SE->getSCEV(getPointerOperand(Inst));
				const SCEVUnknown *BasePointer =
				    dyn_cast<SCEVUnknown>(SCEVTreeLeaf->SE->getPointerBase(AccessFunction));

				assert(BasePointer && "Could not find base pointer");
				AccessFunction = SCEVTreeLeaf->SE->getMinusSCEV(AccessFunction, BasePointer);

				bool IsAffine = isAffineExpr(NULL, AccessFunction, *SCEVTreeLeaf->SE,
				                             BasePointer->getValue());

				Functions.push_back(std::make_pair(IRAccess(Type,
				                                   BasePointer->getValue(),
				                                   AccessFunction, Size,
				                                   IsAffine),
				                                   &Inst));
			} else if(CallInst *callInst = dyn_cast<CallInst>(&Inst))
				if(!callInst->getCalledFunction()->isDeclaration())
					callInstructions.push_back(callInst);
		}

		if(!Functions.empty()) {
			AccFuncSetType &funcRef = tempScopPart.AccFuncMap[basicBlock];
			funcRef.insert(funcRef.begin(),Functions.begin(), Functions.end());
		}
	}

	void buildAffineCondition(Value &V, bool inverted,
	                          Comparison *&Comp) const {
		if (ConstantInt *C = dyn_cast<ConstantInt>(&V)) {
			// If this is always true condition, we will create 1 >= 0,
			// otherwise we will create 1 == 0.
			const SCEV *LHS = SCEVTreeLeaf->SE->getConstant(C->getType(), 0);
			const SCEV *RHS = SCEVTreeLeaf->SE->getConstant(C->getType(), 1);

			if (C->isOne() == inverted)
				Comp = new Comparison(RHS, LHS, ICmpInst::ICMP_NE);
			else
				Comp = new Comparison(LHS, LHS, ICmpInst::ICMP_EQ);

			return;
		}

		ICmpInst *ICmp = dyn_cast<ICmpInst>(&V);
		assert(ICmp && "Only ICmpInst of constant as condition supported!");

		const SCEV *LHS = SCEVTreeLeaf->SE->getSCEV(ICmp->getOperand(0)),
		            *RHS = SCEVTreeLeaf->SE->getSCEV(ICmp->getOperand(1));

		ICmpInst::Predicate Pred = ICmp->getPredicate();

		// Invert the predicate if needed.
		if (inverted)
			Pred = ICmpInst::getInversePredicate(Pred);

		switch (Pred) {
		case ICmpInst::ICMP_UGT:
		case ICmpInst::ICMP_UGE:
		case ICmpInst::ICMP_ULT:
		case ICmpInst::ICMP_ULE:
			// TODO: At the moment we need to see everything as signed. This is an
			//       correctness issue that needs to be solved.
			//AffLHS->setUnsigned();
			//AffRHS->setUnsigned();
			break;
		default:
			break;
		}

		Comp = new Comparison(LHS, RHS, Pred);
	}

	void buildCondition(BasicBlock *basicBlock, TempScop::TempScopPart &tempScopPart, BBCond & conditionsOfLaunch) {

		BBCond &conditions = tempScopPart.BBConds[basicBlock];
		conditions.insert(conditions.begin(),conditionsOfLaunch.begin(), conditionsOfLaunch.end());
		DomTreeNode *BBNode = DT->getNode(basicBlock);

		assert(BBNode && "Get null node while building condition!");

		// Walk up the dominance tree until reaching the entry node. Add all
		// conditions on the path to BB except if BB postdominates the block
		// containing the condition.
		while ( BBNode->getIDom() != NULL ) {

			BasicBlock *CurBB = BBNode->getBlock();
			BBNode = BBNode->getIDom();
			BasicBlock *IDomOfCurrBB = BBNode->getBlock();

			if (PDT->dominates(CurBB, IDomOfCurrBB))
				// Подореваю, что это выполняется тогда же, когда BBNode
				// является непосредственным доминатором только для CurBB, т.е. ветвления нету
				continue;

			BranchInst *Br = dyn_cast<BranchInst>(IDomOfCurrBB->getTerminator());
			assert(Br && "A Valid Scop should only contain branch instruction");

			if (Br->isUnconditional())
				continue;

			// Is BB on the ELSE side of the branch?
			bool inverted = DT->dominates(Br->getSuccessor(1), basicBlock);

			Comparison *Cmp = NULL;
			buildAffineCondition(*(Br->getCondition()), inverted, Cmp);
			assert(Cmp);
			conditions.push_back(*Cmp);
		}
		assert( BBNode->getBlock() == &f->getEntryBlock());
	}

	int buildBoundsOfSubloops(Loop *L, TempScop::TempScopPart &tempScopPart) {

		unsigned MaxLoopDepth = L -> getLoopDepth();
		typedef GraphTraits<Loop *> LoopTraits;
		for (typename LoopTraits::ChildIteratorType iter = LoopTraits::child_begin(L), SE = LoopTraits::child_end(L);
		     iter != SE; ++iter) {

			Loop * L = *iter;
			const SCEV *BackedgeTakenCount = SCEVTreeLeaf->SE->getBackedgeTakenCount(L);
			tempScopPart.LoopBounds[L] = BackedgeTakenCount;

			unsigned LoopDepth = buildBoundsOfSubloops(L, tempScopPart);
			MaxLoopDepth = (LoopDepth > MaxLoopDepth)? LoopDepth : MaxLoopDepth;

		}
		return MaxLoopDepth;
	}

	void buildLoopBounds(TempScop::TempScopPart &tempScopPart) {
		unsigned MaxLoopDepth = 0;

		for(LoopInfo::iterator iter = SCEVTreeLeaf->LI->begin(), iterEnd = SCEVTreeLeaf->LI->end();
		    iter != iterEnd; iter++) {
			Loop * L = *iter;
			const SCEV *BackedgeTakenCount = SCEVTreeLeaf->SE->getBackedgeTakenCount(L);
			tempScopPart.LoopBounds[L] = BackedgeTakenCount;

			unsigned LoopDepth = buildBoundsOfSubloops(L, tempScopPart);
			MaxLoopDepth = (LoopDepth > MaxLoopDepth)? LoopDepth : MaxLoopDepth;
		}
		tempScop->MaxLoopDepth = MaxLoopDepth;
	}

	void buildTempScopPart() {

		TempScop::TempScopPart &tempScopPart = (*tempScop)[SCEVTreeLeaf];

		BBCond conditionsOfLaunch;

		if(SCEVTreeLeaf -> parentNode) {
			TempScop::TempScopPart &parentTempScopPart = (*tempScop)[ SCEVTreeLeaf ->parentNode];
			BBCond &conditions = parentTempScopPart.BBConds[SCEVTreeLeaf->invokeCallInst->getParent()];
			conditionsOfLaunch.insert(conditionsOfLaunch.begin(),  conditions.begin(),  conditions.end());
		}

		for(Function::iterator block = f->begin(), blockEnd=f->end();
		    block!=blockEnd; block++) {
			buildAccessFunctions(block, tempScopPart);
			buildCondition(block, tempScopPart, conditionsOfLaunch);
		}
		buildLoopBounds(tempScopPart);
	}
	
	virtual bool runOnFunction(Function &F) {

		if(parsingType == InterprocedureParsing)
			assert(!f);

		f = &F;
		TD = &getAnalysis<TargetData>();
		KV = NULL;
		DT = &getAnalysis<DominatorTree>();
		PDT =  &getAnalysis<PostDominatorTree>();


		LLVMContext &context = f->getParent()->getContext();
		Type *int64Ty = Type::getInt64Ty( context );

		if(parsingType == InterprocedureParsing) {
			if(tempScop == NULL) {
				KV = &getAnalysis<KernelVerification>();
				assert(KV);
				if(KV->verificationResult == true) {
					tempScop = new TempScop(F);
					SCEVTreeLeaf = KV->SCEVTreeLeaf;
					assert(SCEVTreeLeaf->parentNode == NULL);
					
					// Поставить текущей функции метку с глубиной нуль
					//NamedMDNode *nodeForFunction = f->getParent()->getOrInsertNamedMetadata(f->getName());
					//nodeForFunction->addOperand(MDNode::get(context,ConstantInt::get(int64Ty,0)) );
					
				} else
					// verifaction failed
					// We can not parse scop
					return false;
			} else {
				SCEVTreeLeaf->updateAnalysis();
				assert(!KV && SCEVTreeLeaf->parentNode != NULL);
			}
			assert(SCEVTreeLeaf);
		} else { 
		    // parsingType == ParsingType::CodegenParsing
			tempScop = NULL;
			SCEVTreeLeaf = NULL;
			// check if this is not generated function
			NamedMDNode *nodeForFunction = f->getParent()->getNamedMetadata(f->getName());
			if(!nodeForFunction)
				return false;
				
			assert(!SCEVTreeLeaf);

			SCEVTreeLeaf = new SCEVTreeNode(f, &getAnalysis<ScalarEvolution>(), DT, &getAnalysis<LoopInfo>());
			tempScop = new TempScop(F);
			
		}

		
		buildTempScopPart();

     
		if(parsingType == InterprocedureParsing) {
			for(std::vector<CallInst *>::iterator iter = callInstructions.begin(), iterEnd = callInstructions.end();
			    iter != iterEnd; iter++) {
				SCEVTreeNode *childNode =SCEVTreeLeaf->childNodes[*iter];
				assert(childNode);
				FunctionPassManager manager(f->getParent());
				manager.add(new TargetData(f->getParent()));
				manager.add(new KernelTempScopInfo(tempScop, childNode));
				manager.run(*(*iter)->getCalledFunction());
				
				//поставить метку = глубина callInst
				//в данный момент анализ в правильном состоянии
				
				/*CallInst * callInst = *iter;
				uint64_t callInstDepth = SCEVTreeLeaf->rootLI->getLoopFor(callInst->getParent())->getLoopDepth();
				unsigned MDKindID = f->getParent()->getMDKindID("depthOfCallInst");
				
				MDNode * MDNodeForCallInst = MDNode::get(context,ConstantInt::get(int64Ty,callInstDepth));
				callInst->setMetadata(MDKindID,MDNodeForCallInst);
				
				NamedMDNode *nodeForFunction = f->getParent()->getOrInsertNamedMetadata(callInst->getCalledFunction()->getName());
				bool isDepthAlreadyInOperands = false;
				for(int i =0; i < nodeForFunction->getNumOperands() && !isDepthAlreadyInOperands; i++)
				{
					uint64_t depth = cast<ConstantInt>(nodeForFunction->getOperand(i)->getOperand(0))->getZExtValue();
					if(depth == callInstDepth)
						isDepthAlreadyInOperands = true;
				}
				if(!isDepthAlreadyInOperands)
					nodeForFunction->addOperand(MDNode::get(context,ConstantInt::get(int64Ty,callInstDepth)) );
				*/
				
			}
		}

		if(parsingType == CodegenParsing)
			DEBUG(tempScop->dump());
		else {
			if(KV)
				DEBUG(tempScop->dump());//SCEVTreeNode::freeTree(SCEVTreeLeaf);
			else
				SCEVTreeLeaf->reestablishAnalysis();
		}
		
		return false;
	}
	void releaseMemory()
	{
		if(parsingType == CodegenParsing && SCEVTreeLeaf) {
			//SCEVTreeNode::freeTree(SCEVTreeLeaf);
			//assert(!SCEVTreeLeaf);
			delete SCEVTreeLeaf;
			SCEVTreeLeaf = NULL;
		}
	}
	bool requireVerification;
	void getAnalysisUsage(AnalysisUsage &AU) const {
		AU.addRequired<TargetData>();
		//AU.addRequiredID(polly::IndependentBlocksID);
		if(requireVerification && parsingType == InterprocedureParsing)
			AU.addRequiredTransitive<KernelVerification>();
		if(parsingType == CodegenParsing)
		{
		    AU.addRequiredTransitive<LoopInfo>();
			AU.addRequiredTransitive<ScalarEvolution>();
		}
		AU.addRequiredTransitive<DominatorTree>();
		AU.addRequiredTransitive<PostDominatorTree>();
		AU.setPreservesAll();
	} // end namespace polly
};
}

namespace llvm
{
class PassRegistry;
void initializeKernelTempScopInfoPass(llvm::PassRegistry&);
}
