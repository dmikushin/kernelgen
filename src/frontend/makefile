.PHONY: kernelgen-version.c

OPT ?= 0

ROOT ?= /opt/kernelgen

GCC_ROOT ?= ${HOME}/rpmbuild/BUILD/gcc-snapshot
PLUGIN_INC = -I$(GCC_ROOT)/gcc/ -I$(GCC_ROOT)/build/gcc/ -I$(GCC_ROOT)/include/ -I$(GCC_ROOT)/libcpp/include/

LLVM_ROOT ?= ${HOME}/rpmbuild/BUILD/llvm
LLVM_VERSION ?= 3.2
LLVM_DEF = -D_GNU_SOURCE -D__STDC_LIMIT_MACROS -D__STDC_CONSTANT_MACROS
LLVM_INC = -I$(ROOT)/include $(LLVM_DEF)
LLVM_LIB = -L$(ROOT)/lib -lLLVM-$(LLVM_VERSION)svn

ELF_INC = /usr/include/libelf

BUGPOINT_ROOT ?= $(LLVM_ROOT)/tools/bugpoint
BUGPOINT_INC = ../bugpoint
BUGPOINT_LIBS = -L../bugpoint -lbugpoint

all: libkernelgen-ct.so libkernelgen-opt.so kernelgen-simple kernelgen-version

OBJECTS_CT := kernelgen-ct
OBJECTS_CT := $(addsuffix .o, $(OBJECTS_CT))

OBJECTS_OPT := BranchedLoopExtractor BranchedCodeExtractor FixPointers GlobalDependences
OBJECTS_OPT := $(addsuffix .o, $(OBJECTS_OPT))

OBJECTS_KS := simple
OBJECTS_KS := $(addsuffix .o, $(OBJECTS_KS))

CXXFLAGS = -g -fPIC -O$(OPT)

libkernelgen-ct.so: $(OBJECTS_CT) libkernelgen-opt.so
	$(CXX) -g -shared $(OBJECTS_CT) -o $@ -L. -lkernelgen-opt $(BUGPOINT_LIBS) $(LLVM_LIB)

libkernelgen-opt.so: $(OBJECTS_OPT)
	$(CXX) -g -shared $^ -o $@ $(LLVM_LIB)

LinkFunctionBody.o: LinkFunctionBody.cpp LinkFunctionBody.h
	$(CXX) $(CXXFLAGS) $(LLVM_INC) $(LLVM_DEF) -g -fno-rtti -c $< -o  $@

kernelgen-simple: $(OBJECTS_KS) LinkFunctionBody.o libkernelgen-opt.so
	$(CXX) -g $(OBJECTS_KS) LinkFunctionBody.o -o $@ -lelf -L. -lkernelgen-opt $(BUGPOINT_LIBS) $(LLVM_LIB) -L$(ROOT)/lib -L$(ROOT)/lib64 -liberty

kernelgen-ct.o: compile.cpp $(BUGPOINT_INC)/TrackedPassManager.h
	$(CXX) $(CXXFLAGS) -I$(BUGPOINT_ROOT) -I$(BUGPOINT_INC) $(PLUGIN_INC) $(LLVM_INC) -fPIC -c $< -o $@

simple.o: simple.cpp $(BUGPOINT_INC)/TrackedPassManager.h
	$(CXX) $(CXXFLAGS) -I$(BUGPOINT_ROOT) -I$(BUGPOINT_INC) -I$(ELF_INC) $(LLVM_INC) -c $< -o $@

BranchedLoopExtractor.o: BranchedLoopExtractor.cpp 
	$(CXX) $(CXXFLAGS) $(LLVM_INC) -fno-rtti -c $< -o $@

BranchedCodeExtractor.o: BranchedCodeExtractor.cpp
	$(CXX) $(CXXFLAGS) $(LLVM_INC) -c $< -o $@

FixPointers.o: FixPointers.cpp
	$(CXX) $(CXXFLAGS) $(LLVM_INC) -c $< -o $@

GlobalDependences.o : GlobalDependences.cpp
	$(CXX) $(CXXFLAGS) $(LLVM_INC) -fno-rtti -c $< -o $@

kernelgen-version.c:
	perl kernelgen-version.pl > $@

kernelgen-version: kernelgen-version.c
	gcc $< -o $@

clean:
	rm -rf kernelgen-simple libkernelgen-ct.so libkernelgen-opt.so *.o

