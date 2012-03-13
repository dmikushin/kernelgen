GCC_ROOT ?= ${HOME}/rpmbuild/BUILD/gcc-4.6.3
PLUGIN_INC = -I$(GCC_ROOT)/gcc/ -I$(GCC_ROOT)/build/gcc/ -I$(GCC_ROOT)/include/ -I$(GCC_ROOT)/libcpp/include/

LLVM_ROOT ?= ${HOME}/rpmbuild/BUILD/llvm
LLVM_DEF = -D_GNU_SOURCE -D__STDC_LIMIT_MACROS -D__STDC_CONSTANT_MACROS
LLVM_INC = -I$(LLVM_ROOT)/build/include $(LLVM_DEF)
LLVM_LIB = -L$(LLVM_ROOT)/build/Debug+Asserts/lib -lLLVM-3.1svn

KERNELGEN_INC = -I.. -I../include -I../libasfermi/tests/dyloader
KERNELGEN_LIB = -L../runtime -lkernelgen -lelf -lmhash -ldl \
	-L../libasfermi/tests/dyloader -ldyloader -L../libasfermi \
	$(LLVM_ROOT)/build/tools/polly/Debug+Asserts/lib/LLVMPolly.so \
	-L${HOME}/rpmbuild/BUILD/cloog/isl/.libs/ -lisl \
	-L${HOME}/rpmbuild/BUILD/cloog/.libs/ -lcloog-isl -lffi -lrt -lgmp

all: libkernelgen-ct.so libkernelgen-lt.so kernelgen-simple

OBJECTS_CT := kernelgen-ct BranchedLoopExtractor BranchedCodeExtractor FixPointers
OBJECTS_CT := $(addsuffix .o, $(OBJECTS_CT))

OBJECTS_LT := kernelgen-lt
OBJECTS_LT := $(addsuffix .o, $(OBJECTS_LT))

OBJECTS_KS := simple BranchedLoopExtractor BranchedCodeExtractor FixPointers
OBJECTS_KS := $(addsuffix .o, $(OBJECTS_KS))

libkernelgen-ct.so: $(OBJECTS_CT)
	g++ -g -shared $^ -o $@ $(LLVM_LIB)

libkernelgen-lt.so: $(OBJECTS_LT)
	g++ -g -shared $^ -o $@ $(LLVM_LIB)

kernelgen-simple: $(OBJECTS_KS)
	g++ -g $^ -o $@ $(KERNELGEN_LIB) $(LLVM_LIB)

kernelgen-ct.o: compile.cpp
	g++ -g -fPIC $(PLUGIN_INC) $(LLVM_INC) -fPIC -c $< -o $@

kernelgen-lt.o: link.cpp
	g++ -g -fPIC $(KERNELGEN_INC) $(LLVM_INC) -fPIC -c $< -o $@

simple.o: simple.cpp
	g++ -g $(KERNELGEN_INC) $(LLVM_INC) -c $< -o $@

BranchedLoopExtractor.o: BranchedLoopExtractor.cpp 
	g++ -g -fPIC $(LLVM_INC) -fno-rtti -c $< -o $@

BranchedCodeExtractor.o: BranchedCodeExtractor.cpp
	g++ -g -fPIC $(LLVM_INC) -c $< -o $@

FixPointers.o: FixPointers.cpp
	g++ -g -fPIC $(LLVM_INC) -c $< -o $@

clean:
	rm -rf kernelgen-simple libkernelgen-ct.so libkernelgen-lt.so *.o

