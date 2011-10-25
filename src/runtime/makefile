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

include ../../makefile.in

OPT ?= 0

CFLAGS = -g -fPIC -std=c99 -O$(OPT)
CXXFLAGS = -g -fPIC -O$(OPT)

OBJECTS = entry compile launch

OBJECTS32 := $(addsuffix .o, $(addprefix 32/, $(OBJECTS)))
OBJECTS64 := $(addsuffix .o, $(addprefix 64/, $(OBJECTS)))

ifeq ($(ARCH), 32)
CC32 = $(CC)
CXX32 = $(CXX)
all: 32/libkernelgen.a
else
CC32 = $(CC) -m32
CXX32 = $(CXX) -m32
CC64 = $(CC)
CXX64 = $(CXX)
all: 32/libkernelgen.a 64/libkernelgen.a
endif

LLVMDEF = -D_GNU_SOURCE -D__STDC_LIMIT_MACROS -D__STDC_CONSTANT_MACROS
LLVMINC = -I/home/marcusmae/rpmbuild/BUILD/llvm/build/include
LLVMLIB = -L/opt/kernelgen/lib -lLLVM-3.0svn
UTILINC = -I..
UTILLIB32 = ../util/32/libutil.a
UTILLIB64 = ../util/64/libutil.a

32/libkernelgen.a: $(OBJECTS32)
	$(AR) rcs $@ $(OBJECTS32) $(UTILLIB32)
64/libkernelgen.a: $(OBJECTS64)
	$(AR) rcs $@ $(OBJECTS64) $(UTILLIB64)

32/entry.o: entry.cpp
	$(CXX32) $(CXXFLAGS) $(UTILINC) -c $< -o $@
64/entry.o: entry.cpp
	$(CXX64) $(CXXFLAGS) $(UTILINC) -c $< -o $@

32/compile.o: compile.cpp runtime.h
	$(CXX32) $(CXXFLAGS) $(UTILINC) $(LLVMINC) $(LLVMDEF) -c $< -o $@
64/compile.o: compile.cpp runtime.h
	$(CXX64) $(CXXFLAGS) $(UTILINC) $(LLVMINC) $(LLVMDEF) -c $< -o $@

32/launch.o: launch.cpp runtime.h
	$(CXX32) $(CXXFLAGS) $(UTILINC) $(LLVMINC) $(LLVMDEF) -c $< -o $@
64/launch.o: launch.cpp runtime.h
	$(CXX64) $(CXXFLAGS) $(UTILINC) $(LLVMINC) $(LLVMDEF) -c $< -o $@

clean:
	rm -rf 32/*.o 64/*.o 32/*.a 64/*.a

