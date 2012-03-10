GCC_ROOT ?= ${HOME}/rpmbuild/BUILD/gcc-4.6.3
PLUGIN_INC = -I$(GCC_ROOT)/gcc/ -I$(GCC_ROOT)/build/gcc/ -I$(GCC_ROOT)/include/ -I$(GCC_ROOT)/libcpp/include/

LLVM_ROOT ?= ${HOME}/rpmbuild/BUILD/llvm
LLVM_DEF = -D_GNU_SOURCE -D__STDC_LIMIT_MACROS -D__STDC_CONSTANT_MACROS
LLVM_INC = -I$(LLVM_ROOT)/build/include $(LLVM_DEF)
LLVM_LIB = -L$(LLVM_ROOT)/build/Debug+Asserts/lib -lLLVM-3.1svn

all: libkernelgen-ct.so

OBJECTS := compile BranchedLoopExtractor BranchedCodeExtractor FixUsingOfMalloc
OBJECTS := $(addsuffix .o, $(OBJECTS))

libkernelgen-ct.so: $(OBJECTS)
	g++ -g -fPIC $(PLUGIN_INC) $(LLVM_INC) -shared $^ -o $@ $(LLVM_LIB)

kernelgen-ct.o: compile.cpp
	g++ -g -fPIC $(PLUGIN_INC) $(LLVM_INC) -fPIC -c $< -o $@

BranchedLoopExtractor.o: BranchedLoopExtractor.cpp 
	g++ -g -fPIC $(LLVM_INC) -fno-rtti -c $< -o $@

BranchedCodeExtractor.o: BranchedCodeExtractor.cpp
	g++ -g -fPIC $(LLVM_INC) -c $< -o $@

FixUsingOfMalloc.o: FixUsingOfMalloc.cpp
	g++ -g -fPIC $(LLVM_INC) -c $< -o $@

clean:
	rm -rf libkernelgen-ct.so *.o

