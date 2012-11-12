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

#ifndef TRACKED_PASS_MANAGER
#define TRACKED_PASS_MANAGER

#include "llvm/LLVMContext.h"
#include "llvm/Module.h"
#include "llvm/PassManager.h"
#include "llvm/Support/Signals.h"
#include "llvm/Transforms/Utils/Cloning.h"

#include "BugDriver.h"

#include <iostream>
#include <list>

using namespace llvm;
using namespace std;

// A type of the pass tracker callback to launch after
// reducing the bug point.
typedef void (*PassTrackerCallback)(void* arg);

// Tracker to catch and inspect the crashing LLVM passes.
class PassTracker
{
	list<Pass*> passes;
	string input;
	PassTrackerCallback callback;
	void* callback_arg;
	OwningPtr<Module> module;

	// Run the same passes in the bugpoint.
	static void handler(void* instance)
	{
		PassTracker* tracker = (PassTracker*)instance;

		do
		{
			// Do not track if not explicitly requested to.
			char* cbugpoint = getenv("KERNELGEN_BUGPOINT");
			if (!cbugpoint) break;
			if (cbugpoint)
			{
				int bugpoint = atoi(cbugpoint);
				if (!bugpoint) break;
			}

			if (!tracker->module) break;

			// Initialize the bug driver
			bool FindBugs = false;
			int TimeoutValue = 300;
			int MemoryLimit = -1;
			bool UseValgrind = false;
			BugDriver D("kernelgen-simple", FindBugs, TimeoutValue, MemoryLimit,
				UseValgrind, getGlobalContext());

			// Add currently tracked passes.
			for (list<Pass*>::iterator i = tracker->passes.begin(),
				e = tracker->passes.end(); i != e; i++)
			{
				const void *ID = (*i)->getPassID();
				if (!ID) continue;
				const PassInfo *PI = PassRegistry::getPassRegistry()->getPassInfo(ID);
				if (!PI) continue;
				const char* arg = PI->getPassArgument();
				if (!arg) continue;
				if (!strcmp(arg, "targetdata")) continue;
				if (!strcmp(arg, "targetpassconfig")) continue;
				if (!strcmp(arg, "machinemoduleinfo")) continue;
				D.addPass(arg);
				
			}

			// We don't delete module here, since it could get
			// trashed by buggy passes and deletion may fail.
			Module* module_to_break = CloneModule(tracker->module.get());
			D.setNewProgram(module_to_break);

			// Reduce the test case.
			D.debugOptimizerCrash(tracker->input);
		}
		while (0);

		// After the test case is reduced, fallback to the
		// regular compilation, if explicitly requested to recover.
		char* crecover = getenv("KERNELGEN_RECOVER");
		if (crecover)
		{
			int recover = atoi(crecover);
			if (!recover) return;

			if (tracker->callback)
				tracker->callback(tracker->callback_arg);
		}
	}

public:
	PassTracker(const char* input,
		PassTrackerCallback callback, void* callback_arg) :
		input(input ? input : ""), callback(callback), callback_arg(callback_arg), module(NULL)
	{
		sys::AddSignalHandler(PassTracker::handler, this);
	}

	void reset()
	{
		passes.clear();
		if (module)
		{
			Module* m = module.take();
			delete m;
		}
	}

	void add(Pass *P)
	{
		passes.push_back(P);
	}

	void run(Module* M)
	{
		module.reset(CloneModule(M));
	}
};

extern PassTracker* tracker;


class TrackedPassManager : public PassManager
{
	PassTracker* tracker;

public:
	TrackedPassManager(PassTracker* tracker) :
		PassManager(), tracker(tracker) { }

	virtual void add(Pass *P)
	{
		tracker->add(P);
		PassManager::add(P);
	}

	virtual bool run(Module &M)
	{
		tracker->run(&M);
		return PassManager::run(M);
	}

	~TrackedPassManager()
	{
		tracker->reset();
	}
};

#endif // TRACKED_PASS_MANAGER

