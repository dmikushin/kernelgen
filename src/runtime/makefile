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

LLVMDEF = -D_GNU_SOURCE -D__STDC_LIMIT_MACROS -D__STDC_CONSTANT_MACROS
LLVMINC = \
	-I/usr/lib64/libffi-3.0.9/include \
	-I${HOME}/rpmbuild/BUILD/llvm/build/include \
	-I${HOME}/rpmbuild/BUILD/llvm/tools/polly/include \
	-I${HOME}/rpmbuild/BUILD/cloog/include \
	-I${HOME}/rpmbuild/BUILD/cloog/isl/include \
	-I${HOME}/rpmbuild/BUILD/llvm/build/tools/polly/include

CFLAGS = -g -fPIC -std=c99 -O$(OPT) -I../include
CXXFLAGS = -g -fPIC -O$(OPT) -I../include $(LLVMINC) $(LLVMDEF)

OBJECTS = \
	entry compile launch finish timing io elf execute bind nvopencc \
	memory hostcall wrappers \
	BlockGenerator ClastExpCodegen ClastStmtCodeGen CodeGeneration

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

32/libkernelgen.a: $(OBJECTS32)
	$(AR) rcs $@ $(OBJECTS32)
64/libkernelgen.a: $(OBJECTS64)
	$(AR) rcs $@ $(OBJECTS64)

32/entry.o: entry.cpp bind.h
	$(CXX32) $(CXXFLAGS) -c $< -o $@
64/entry.o: entry.cpp bind.h
	$(CXX64) $(CXXFLAGS) -c $< -o $@

32/compile.o: compile.cpp runtime.h
	$(CXX32) $(CXXFLAGS) -c $< -o $@
64/compile.o: compile.cpp runtime.h
	$(CXX64) $(CXXFLAGS) -c $< -o $@

32/launch.o: launch.cpp runtime.h
	$(CXX32) $(CXXFLAGS) -c $< -o $@
64/launch.o: launch.cpp runtime.h
	$(CXX64) $(CXXFLAGS) -c $< -o $@

32/finish.o: finish.cpp runtime.h
	$(CXX32) $(CXXFLAGS) -c $< -o $@
64/finish.o: finish.cpp runtime.h
	$(CXX64) $(CXXFLAGS) -c $< -o $@

32/timing.o: timing.c util.h
	$(CC32) $(CFLAGS) -D_GNU_SOURCE -c $< -o $@
64/timing.o: timing.c util.h
	$(CC64) $(CFLAGS) -D_GNU_SOURCE -c $< -o $@

32/io.o: io.cpp io.h
	$(CXX32) $(CXXFLAGS) -c $< -o $@
64/io.o: io.cpp io.h
	$(CXX64) $(CXXFLAGS) -c $< -o $@

32/elf.o: elf.cpp elf.h
	$(CXX32) $(CXXFLAGS) -c $< -o $@
64/elf.o: elf.cpp elf.h
	$(CXX64) $(CXXFLAGS) -c $< -o $@

32/elf_read_test: elf_read_test.cpp elf.h util.h
	$(CXX32) $(CXXFLAGS) $< -o $@ -L32 -lkernelgen -lelf
64/elf_read_test: elf_read_test.cpp elf.h util.h
	$(CXX64) $(CXXFLAGS) $< -o $@ -L64 -lkernelgen -lelf

32/elf_write_test: elf_write_test.c elf.h util.h
	$(CC32) $(CFLAGS) $(INCFFI) $< -o $@ -L32 -lkernelgen
64/elf_write_test: elf_write_test.c elf.h util.h
	$(CC64) $(CFLAGS) $(INCFFI) $< -o $@ -L64 -lkernelgen

NUM_SYMBOLS ?= 1

32/test: elf_read_test.cpp elf_write_test
	32/elf_write_test 32/foo.o $(NUM_SYMBOLS) sym hello
	32/elf_read_test /proc/self/exe sym
64/test: elf_read_test.cpp elf_write_test
	64/elf_write_test 64/foo.o $(NUM_SYMBOLS) sym hello
	64/elf_read_test /proc/self/exe sym

32/execute.o: execute.cpp
	$(CXX32) $(CXXFLAGS) -c $< -o $@
64/execute.o: execute.cpp
	$(CXX64) $(CXXFLAGS) -c $< -o $@

32/bind.o: bind.cpp bind.h
	$(CXX32) $(CXXFLAGS) -c $< -o $@
64/bind.o: bind.cpp bind.h
	$(CXX64) $(CXXFLAGS) -c $< -o $@

32/nvopencc.o: nvopencc.cpp
	$(CXX32) $(CXXFLAGS) -c $< -o $@
64/nvopencc.o: nvopencc.cpp
	$(CXX64) $(CXXFLAGS) -c $< -o $@

32/memory.o: memory.cpp
	$(CXX32) $(CXXFLAGS) -c $< -o $@
64/memory.o: memory.cpp
	$(CXX64) $(CXXFLAGS) -c $< -o $@

32/hostcall.o: hostcall.cpp
	$(CXX32) $(CXXFLAGS) -c $< -o $@
64/hostcall.o: hostcall.cpp
	$(CXX64) $(CXXFLAGS) -c $< -o $@

32/wrappers.o: wrappers.cpp
	$(CXX32) $(CXXFLAGS) -c $< -o $@
64/wrappers.o: wrappers.cpp
	$(CXX64) $(CXXFLAGS) -c $< -o $@

32/BlockGenerator.o: BlockGenerator.cpp
	$(CXX32) $(CXXFLAGS) -fno-rtti -c $< -o $@
64/BlockGenerator.o: BlockGenerator.cpp
	$(CXX64) $(CXXFLAGS) -fno-rtti -c $< -o $@

32/ClastExpCodegen.o: ClastExpCodegen.cpp
	$(CXX32) $(CXXFLAGS) -fno-rtti -c $< -o $@
64/ClastExpCodegen.o: ClastExpCodegen.cpp
	$(CXX64) $(CXXFLAGS) -fno-rtti -c $< -o $@

32/ClastStmtCodeGen.o: ClastStmtCodeGen.cpp
	$(CXX32) $(CXXFLAGS) -fno-rtti -c $< -o $@
64/ClastStmtCodeGen.o: ClastStmtCodeGen.cpp
	$(CXX64) $(CXXFLAGS) -fno-rtti -c $< -o $@

32/CodeGeneration.o: CodeGeneration.cpp
	$(CXX32) $(CXXFLAGS) -fno-rtti -c $< -o $@
64/CodeGeneration.o: CodeGeneration.cpp
	$(CXX64) $(CXXFLAGS) -fno-rtti -c $< -o $@

clean:
	rm -rf 32/*.o 64/*.o 32/*.a 64/*.a \
	32/elf_write_test 32/elf_read_test 64/elf_write_test 64/elf_read_test

