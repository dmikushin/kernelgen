GCC_ROOT ?= ${HOME}/rpmbuild/BUILD/gcc-4.6.3
PLUGIN_INC = -I$(GCC_ROOT)/gcc/ -I$(GCC_ROOT)/build/gcc/ -I$(GCC_ROOT)/include/ -I$(GCC_ROOT)/libcpp/include/

LLVM_ROOT ?= ${HOME}/rpmbuild/BUILD/llvm
LLVM_DEF = -D_GNU_SOURCE -D__STDC_LIMIT_MACROS -D__STDC_CONSTANT_MACROS
LLVM_INC = -I$(LLVM_ROOT)/build/include $(LLVM_DEF)
LLVM_LIB = -L$(LLVM_ROOT)/build/Debug+Asserts/lib -lLLVM-3.1svn

KERNELGEN_INC = -I.. -I../include -I../libasfermi/tests/dyloader
KERNELGEN_LIB = -lelf

BUGPOINT_ROOT ?= $(LLVM_ROOT)/tools/bugpoint

all: libkernelgen-ct.so libkernelgen-lt.so libkernelgen-opt.so kernelgen-simple

BUGPOINT := BugDriver CrashDebugger ExtractFunction Miscompilation ToolRunner \
	bugpoint ExecutionDriver FindBugs OptimizerDriver
BUGPOINT := $(addsuffix .bugpoint, $(BUGPOINT))

OBJECTS_CT := kernelgen-ct $(BUGPOINT)
OBJECTS_CT := $(addsuffix .o, $(OBJECTS_CT))

OBJECTS_LT := kernelgen-lt
OBJECTS_LT := $(addsuffix .o, $(OBJECTS_LT))

OBJECTS_OPT := BranchedLoopExtractor BranchedCodeExtractor FixPointers
OBJECTS_OPT := $(addsuffix .o, $(OBJECTS_OPT))

OBJECTS_KS := simple $(BUGPOINT)
OBJECTS_KS := $(addsuffix .o, $(OBJECTS_KS))

libkernelgen-ct.so: $(OBJECTS_CT) libkernelgen-opt.so
	g++ -g -shared $(OBJECTS_CT) -o $@ $(LLVM_LIB) -L. -lkernelgen-opt

libkernelgen-lt.so: $(OBJECTS_LT)
	g++ -g -shared $^ -o $@ $(LLVM_LIB)

libkernelgen-opt.so: $(OBJECTS_OPT)
	g++ -g -shared $^ -o $@ $(LLVM_LIB)

kernelgen-simple: $(OBJECTS_KS) libkernelgen-opt.so
	g++ -g $(OBJECTS_KS) -o $@ $(KERNELGEN_LIB) $(LLVM_LIB) -L. -lkernelgen-opt

kernelgen-ct.o: compile.cpp tracker.h
	g++ -g -fPIC $(PLUGIN_INC) $(LLVM_INC) -fPIC -c $< -o $@

kernelgen-lt.o: link.cpp
	g++ -g -fPIC $(KERNELGEN_INC) $(LLVM_INC) -fPIC -c $< -o $@

simple.o: simple.cpp tracker.h
	g++ -g -I$(BUGPOINT_ROOT) $(KERNELGEN_INC) $(LLVM_INC) -c $< -o $@

BranchedLoopExtractor.o: BranchedLoopExtractor.cpp 
	g++ -g -fPIC $(LLVM_INC) -fno-rtti -c $< -o $@

BranchedCodeExtractor.o: BranchedCodeExtractor.cpp
	g++ -g -fPIC $(LLVM_INC) -c $< -o $@

FixPointers.o: FixPointers.cpp
	g++ -g -fPIC $(LLVM_INC) -c $< -o $@

%.bugpoint.o: $(BUGPOINT_ROOT)/%.cpp
	g++ -Dmain=bugpoint -g -fPIC $(LLVM_INC) -c $< -o $@

clean:
	rm -rf kernelgen-simple libkernelgen-ct.so libkernelgen-lt.so libkernelgen-opt.so *.o

