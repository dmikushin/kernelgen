OPT ?= 0

GCC_ROOT ?= ${HOME}/rpmbuild/BUILD/gcc-4.6.3
PLUGIN_INC = -I$(GCC_ROOT)/gcc/ -I$(GCC_ROOT)/build/gcc/ -I$(GCC_ROOT)/include/ -I$(GCC_ROOT)/libcpp/include/

LLVM_ROOT ?= ${HOME}/rpmbuild/BUILD/llvm
LLVM_MODE ?= Debug+Asserts
LLVM_VERSION ?= 3.2
LLVM_DEF = -D_GNU_SOURCE -D__STDC_LIMIT_MACROS -D__STDC_CONSTANT_MACROS
LLVM_INC = -I$(LLVM_ROOT)/build/include $(LLVM_DEF)
LLVM_LIB = -L$(LLVM_ROOT)/build/$(LLVM_MODE)/lib -lLLVM-$(LLVM_VERSION)svn

ELF_INC = /usr/include/libelf

BUGPOINT_ROOT ?= $(LLVM_ROOT)/tools/bugpoint
BUGPOINT_INC = ../bugpoint
BUGPOINT_LIBS = -L../bugpoint -lbugpoint

all: libkernelgen-ct.so libkernelgen-opt.so kernelgen-simple

OBJECTS_CT := kernelgen-ct
OBJECTS_CT := $(addsuffix .o, $(OBJECTS_CT))

OBJECTS_OPT := BranchedLoopExtractor BranchedCodeExtractor FixPointers
OBJECTS_OPT := $(addsuffix .o, $(OBJECTS_OPT))

OBJECTS_KS := simple
OBJECTS_KS := $(addsuffix .o, $(OBJECTS_KS))

CXXFLAGS = -g -fPIC -O$(OPT)

libkernelgen-ct.so: $(OBJECTS_CT) libkernelgen-opt.so
	$(CXX) -g -shared $(OBJECTS_CT) -o $@ $(LLVM_LIB) -L. -lkernelgen-opt

libkernelgen-opt.so: $(OBJECTS_OPT)
	$(CXX) -g -shared $^ -o $@ $(BUGPOINT_LIBS) $(LLVM_LIB)

kernelgen-simple: $(OBJECTS_KS) libkernelgen-opt.so
	$(CXX) -g $(OBJECTS_KS) -o $@ -lelf $(BUGPOINT_LIBS) $(LLVM_LIB) -L. -lkernelgen-opt

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

clean:
	rm -rf kernelgen-simple libkernelgen-ct.so libkernelgen-opt.so *.o

