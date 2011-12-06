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

LLVMDEF = -D_GNU_SOURCE -D__STDC_LIMIT_MACROS -D__STDC_CONSTANT_MACROS
LLVMINC = -I/home/marcusmae/rpmbuild/BUILD/llvm/build/include
LLVMLIB = -L/opt/kernelgen/lib -lLLVM-3.0svn /opt/kernelgen/lib/LLVMPolly.so -L/opt/kernelgen/lib -lisl -lcloog-isl
EXECINC = -I..
EXECLIB = -L../runtime/32 -L../runtime/64 -lkernelgen -lelf -lmhash -ldl

CXX = g++ -g -I../include $(LLVMINC) $(LLVMDEF)

OBJECTS = kernelgen.o compile.o link.o BranchedLoopExtractor.o BranchedCodeExtractor.o

all: kernelgen

kernelgen: $(OBJECTS)
	$(CXX) $(OBJECTS) -o $@ $(EXECLIB) $(LLVMLIB)

kernelgen.o: kernelgen.cpp
	$(CXX) $(EXECINC) -c $< -o $@

compile.o: compile.cpp
	$(CXX) $(EXECINC) -c $< -o $@

link.o: link.cpp
	$(CXX) $(EXECINC) -c $< -o $@

BranchedLoopExtractor.o: BranchedLoopExtractor.cpp 
	$(CXX) $(EXECINC) -fno-rtti -c $< -o $@

BranchedCodeExtractor.o: BranchedCodeExtractor.cpp
	$(CXX) $(EXECINC) -c $< -o $@

clean:
	rm -rf *.o kernelgen

