##
## KGen - the LLVM-based compiler with GPU kernels generation over C backend.
##
## Copyright (c) 2011 Dmitry Mikushin
##
## This software is provided 'as-is', without any express or implied warranty.
## In no event will the authors be held liable for any damages arising 
## from the use of this software.
## Permission is granted to anyone to use this software for any purpose, 
## including commercial applications, and to alter it and redistribute it freely,
## subject to the following restrictions:
##
## 1. The origin of this software must not be misrepresented;
## you must not claim that you wrote the original software.
## If you use this software in a product, an acknowledgment
## in the product documentation would be appreciated but is not required.
## 2. Altered source versions must be plainly marked as such,
## and must not be misrepresented as being the original software.
## 3. This notice may not be removed or altered from any source distribution.
##

.PHONY: compile.o

OPT ?= 0

ROOT ?= /opt/kernelgen

LLVM_ROOT ?= ${HOME}/rpmbuild/BUILD/llvm
LLVM_VERSION ?= 3.2
LLVM_DEF = -D_GNU_SOURCE -D__STDC_LIMIT_MACROS -D__STDC_CONSTANT_MACROS
LLVM_INC = \
	-I/usr/include/i686-linux-gnu \
	-I/usr/include/x86_64-linux-gnu \
	-I/usr/lib64/libffi-3.0.9/include \
	-I${HOME}/rpmbuild/BUILD/llvm/build/include \
	-I${HOME}/rpmbuild/BUILD/llvm/tools/polly/include \
	-I${HOME}/rpmbuild/BUILD/cloog/include \
	-I${HOME}/rpmbuild/BUILD/cloog/isl/include \
	-I${HOME}/rpmbuild/BUILD/llvm/build/tools/polly/include
LLVM_LIB = \
	-Wl,-rpath $(RPATH) -L../bindings -lkernelgen-gpu \
	-L$(ROOT)/lib -lLLVM-$(LLVM_VERSION)svn \
	-L$(ROOT)/lib -lLLVMPolly

ELF_INC = /usr/include/libelf

BUGPOINT_ROOT ?= $(LLVM_ROOT)/tools/bugpoint
BUGPOINT_INC = ../bugpoint
BUGPOINT_LIBS = -L../bugpoint -lbugpoint

CXXFLAGS = \
	-g -fPIC -O$(OPT) -I../include -I../bindings \
	-I../libasfermi/tests/dyloader $(LLVM_INC) $(LLVM_DEF)

OBJECTS := \
	entry compile launch finish timer io elf execute codegen \
	memory hostcall wrappers \
        ConstantSubstitution SizeOfLoops RuntimeAliasAnalysis LinkFunctionBody \
    ScopInspection TransformAccesses

OBJECTS := $(addsuffix .o, $(OBJECTS))

all: libkernelgen-rt.so kernelgen-inspector

libkernelgen-rt.so: $(OBJECTS)
	$(CXX) -shared $(OBJECTS) -o $@ $(BUGPOINT_LIBS) $(LLVM_LIB) -L../bindings -lkernelgen-gpu \
		-L../libasfermi/ -L../libasfermi/tests/dyloader -ldyloader -lasfermi \
		-lelf -lrt -lgmp -lmhash -ldl -lffi

entry.o: entry.cpp $(BUGPOINT_INC)/TrackedPassManager.h
	$(CXX) -I$(ELF_INC) -I$(BUGPOINT_ROOT) -I$(BUGPOINT_INC) $(CXXFLAGS) -c $< -o $@

KERNELGEN_VERSION = $(shell perl kernelgen-version.pl)

compile.o: compile.cpp runtime.h
	$(CXX) -DKERNELGEN_VERSION=\"$(KERNELGEN_VERSION)\" -I$(ELF_INC) $(CXXFLAGS) -c $< -o $@

launch.o: launch.cpp runtime.h
	$(CXX) -I$(ELF_INC) $(CXXFLAGS) -c $< -o $@

finish.o: finish.cpp runtime.h
	$(CXX) $(CXXFLAGS) -c $< -o $@

timer.o: timer.cxx timer.h
	$(CXX) $(CXXFLAGS) -D_GNU_SOURCE -c $< -o $@

io.o: io.cpp io.h
	$(CXX) $(CXXFLAGS) -c $< -o $@

elf.o: elf.cpp elf.h
	$(CXX) -I$(ELF_INC) $(CXXFLAGS) -c $< -o $@

execute.o: execute.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

codegen.o: codegen.cpp $(BUGPOINT_INC)/TrackedPassManager.h
	$(CXX) -I$(ELF_INC) -I$(BUGPOINT_ROOT) -I$(BUGPOINT_INC) $(CXXFLAGS) -c $< -o $@

memory.o: memory.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

hostcall.o: hostcall.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

wrappers.o: wrappers.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

ConstantSubstitution.o: ConstantSubstitution.cpp
	$(CXX) $(CXXFLAGS) -fno-rtti -c $< -o $@

SizeOfLoops.o: SizeOfLoops.cpp
	$(CXX) $(CXXFLAGS) -fno-rtti -c $< -o  $@

RuntimeAliasAnalysis.o: RuntimeAliasAnalysis.cpp
	$(CXX) $(CXXFLAGS) -fno-rtti -c $< -o  $@

LinkFunctionBody.o: LinkFunctionBody.cpp LinkFunctionBody.h
	$(CXX) $(CXXFLAGS) -fno-rtti -c $< -o  $@

kernelgen-inspector: inspector.cpp libkernelgen-rt.so
	$(CXX) -I$(ELF_INC) -I$(BUGPOINT_ROOT) -I$(BUGPOINT_INC) $(CXXFLAGS) $< -o $@ \
	-L. -lkernelgen-rt $(BUGPOINT_LIBS) $(LLVM_LIB) -L../bindings -lkernelgen-gpu \
	-L../libasfermi/ -L../libasfermi/tests/dyloader -ldyloader -lasfermi -lelf -lrt \
	-lgmp -lmhash -ldl -lffi -L${HOME}/rpmbuild/BUILD/cloog/isl/.libs/ -lisl \
	-L${HOME}/rpmbuild/BUILD/cloog/.libs/ -lcloog-isl

ScopInspection.o: ScopInspection.cpp
	$(CXX) $(CXXFLAGS) -fno-rtti -c $< -o  $@

TransformAccesses.o : TransformAccesses.cpp
	$(CXX) $(CXXFLAGS) -fno-rtti -c $< -o  $@

clean:
	rm -rf *.o libkernelgen-rt.so kernelgen-inspector

