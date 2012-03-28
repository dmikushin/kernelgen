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

LLVM_ROOT ?= ${HOME}/rpmbuild/BUILD/llvm
LLVM_MODE ?= Debug+Asserts
LLVMDEF = -D_GNU_SOURCE -D__STDC_LIMIT_MACROS -D__STDC_CONSTANT_MACROS
LLVMINC = \
	-I/usr/include/i686-linux-gnu \
	-I/usr/include/x86_64-linux-gnu \
	-I/usr/lib64/libffi-3.0.9/include \
	-I${HOME}/rpmbuild/BUILD/llvm/build/include \
	-I${HOME}/rpmbuild/BUILD/llvm/tools/polly/include \
	-I${HOME}/rpmbuild/BUILD/cloog/include \
	-I${HOME}/rpmbuild/BUILD/cloog/isl/include \
	-I${HOME}/rpmbuild/BUILD/llvm/build/tools/polly/include
LLVM_LIB = \
	$(LLVM_ROOT)/build/tools/polly/$(LLVM_MODE)/lib/LLVMPolly.so \
	-L$(LLVM_ROOT)/build/$(LLVM_MODE)/lib -lLLVM-3.1svn

CXXFLAGS = -g -fPIC -O$(OPT) -I../include -I../libasfermi/tests/dyloader $(LLVMINC) $(LLVMDEF)

OBJECTS := cuda

OBJECTS := $(addsuffix .o, $(OBJECTS))

all: libkernelgen-gpu.so

libkernelgen-gpu.so: $(OBJECTS)
	$(CXX) -shared $(OBJECTS) -o $@ -ldl

cuda.o: cuda.cpp cuda.h
	$(CXX) $(CXXFLAGS) -c $< -o $@

clean:
	rm -rf *.o libkernelgen-gpu.so

