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

#include "CodeGeneration.h"

using namespace llvm;
using namespace polly;

PassManager kernelgen::runtime::pollygen(Module* m, unsigned int mode, bool codegen)
{
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

	PassManager manager;
	manager.add(new TargetData(m));
	manager.add(createBasicAliasAnalysisPass());			// -basicaa
	manager.add(createPromoteMemoryToRegisterPass());		// -mem2reg
	manager.add(createCFGSimplificationPass());			// -simplifycfg
	manager.add(createInstructionCombiningPass());			// -instcombine
	manager.add(createTailCallEliminationPass());			// -tailcallelim
	manager.add(createLoopSimplifyPass());				// -loop-simplify
	manager.add(createLCSSAPass());					// -lcssa
	manager.add(createLoopRotatePass());				// -loop-rotate
	manager.add(createLCSSAPass());					// -lcssa
	manager.add(createLoopUnswitchPass());				// -loop-unswitch
	manager.add(createInstructionCombiningPass());			// -instcombine
	manager.add(createLoopSimplifyPass());				// -loop-simplify
	manager.add(createLCSSAPass());					// -lcssa
	manager.add(createIndVarSimplifyPass());			// -indvars
	manager.add(createLoopDeletionPass());				// -loop-deletion
	manager.add(createInstructionCombiningPass());			// -instcombine
	manager.add(createCodePreperationPass());			// -polly-prepare
	manager.add(createRegionSimplifyPass());			// -polly-region-simplify
	manager.add(createIndVarSimplifyPass());			// -indvars
	manager.add(createBasicAliasAnalysisPass());			// -basicaa
	manager.add(createScheduleOptimizerPass());			// -polly-optimize-isl

	if (codegen)
	{
		manager.add(kernelgen::createCodeGenerationPass());	// -polly-codegen
		kernelgen::set_flags(0);
	}

	return manager;
}

