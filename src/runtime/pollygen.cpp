/*
 * KernelGen - the LLVM-based compiler with GPU kernels generation over C backend.
 *
 * Copyright (c) 2011 Dmitry Mikushin
 *
 * This software is provided 'as-is', without any express or implied warranty.
 * In no event will the authors be held liable for any damages arising 
 * from the use of this software.
 * Permission is granted to anyone to use this software for any purpose, 
 * including commercial applications, and to alter it and redistribute it freely,
 * subject to the following restrictions:
 *
 * 1. The origin of this software must not be misrepresented;
 * you must not claim that you wrote the original software.
 * If you use this software in a product, an acknowledgment
 * in the product documentation would be appreciated but is not required.
 * 2. Altered source versions must be plainly marked as such,
 * and must not be misrepresented as being the original software.
 * 3. This notice may not be removed or altered from any source distribution.
 */

#include "pollygen.h"

#include "polly/LinkAllPasses.h"
#include "llvm/Analysis/Passes.h"
#include "llvm/Transforms/Scalar.h"
#include "llvm/Target/TargetData.h"

using namespace llvm;
using namespace polly;

PassManager kernelgen::runtime::pollygen(Module* m)
{
	PassManager polly;
	PassRegistry &Registry = *PassRegistry::getPassRegistry();
	initializeCore(Registry);
	initializeScalarOpts(Registry);
	initializeIPO(Registry);
	initializeAnalysis(Registry);
	initializeIPA(Registry);
	initializeTransformUtils(Registry);
	initializeInstCombine(Registry);
	initializeInstrumentation(Registry);
	initializeTarget(Registry);

	polly.add(new TargetData(m));
	polly.add(createBasicAliasAnalysisPass());	// -basicaa
	polly.add(createPromoteMemoryToRegisterPass());	// -mem2reg
	polly.add(createCFGSimplificationPass());	// -simplifycfg
	polly.add(createInstructionCombiningPass());	// -instcombine
	polly.add(createTailCallEliminationPass());	// -tailcallelim
	polly.add(createLoopSimplifyPass());		// -loop-simplify
	polly.add(createLCSSAPass());			// -lcssa
	polly.add(createLoopRotatePass());		// -loop-rotate
	polly.add(createLCSSAPass());			// -lcssa
	polly.add(createLoopUnswitchPass());		// -loop-unswitch
	polly.add(createInstructionCombiningPass());	// -instcombine
	polly.add(createLoopSimplifyPass());		// -loop-simplify
	polly.add(createLCSSAPass());			// -lcssa
	polly.add(createIndVarSimplifyPass());		// -indvars
	polly.add(createLoopDeletionPass());		// -loop-deletion
	polly.add(createInstructionCombiningPass());	// -instcombine		
	polly.add(createCodePreperationPass());		// -polly-prepare
	polly.add(createRegionSimplifyPass());		// -polly-region-simplify
	polly.add(createIndVarSimplifyPass());		// -indvars
	polly.add(createBasicAliasAnalysisPass());	// -basicaa
	polly.add(createScheduleOptimizerPass());	// -polly-optimize-isl

	return polly;
}

