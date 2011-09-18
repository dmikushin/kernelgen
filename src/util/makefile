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

OBJECTS = timing elf_read elf_write execute

OBJECTS32 := $(addsuffix .o, $(addprefix 32/, $(OBJECTS)))
OBJECTS64 := $(addsuffix .o, $(addprefix 64/, $(OBJECTS)))

ifeq ($(ARCH), 32)
CC32 = $(CC)
CXX32 = $(CXX)
all: 32/libutil.a
else
CC32 = $(CC) -m32
CXX32 = $(CXX) -m32
CC64 = $(CC)
CXX64 = $(CXX)
all: 32/libutil.a 64/libutil.a
endif

32/libutil.a: $(OBJECTS32)
	$(AR) rcs $@ $(OBJECTS32)
64/libutil.a: $(OBJECTS64)
	$(AR) rcs $@ $(OBJECTS64)

32/timing.o: timing.c util.h
	$(CC32) $(CFLAGS) -D_GNU_SOURCE -c $< -o $@
64/timing.o: timing.c util.h
	$(CC64) $(CFLAGS) -D_GNU_SOURCE -c $< -o $@

32/elf_read.o: elf_read.c util.h
	$(CC32) $(CFLAGS) -c $< -o $@
64/elf_read.o: elf_read.c util.h
	$(CC64) $(CFLAGS) -c $< -o $@

32/elf_write.o: elf_write.c util.h
	$(CC32) $(CFLAGS) -c $< -o $@
64/elf_write.o: elf_write.c util.h
	$(CC64) $(CFLAGS) -c $< -o $@

32/elf_write_test: elf_write_test.c elf_write.o util.h
	$(CC32) $(CFLAGS) $(INCFFI) $< -o $@ -L32 -lkernelgen
64/elf_write_test: elf_write_test.c elf_write.o util.h
	$(CC64) $(CFLAGS) $(INCFFI) $< -o $@ -L64 -lkernelgen

NUM_SYMBOLS ?= 1

32/test: elf_read_test.c elf_write_test
	32/elf_write_test 32/foo.o $(NUM_SYMBOLS) sym hello
	$(CC32) $(CFLAGS) $< -o 32/elf_read_test 32/foo.o -lelf
	32/elf_read_test /proc/self/exe sym
64/test: elf_read_test.c elf_write_test
	64/elf_write_test 64/foo.o $(NUM_SYMBOLS) sym hello
	$(CC64) $(CFLAGS) $< -o 64/elf_read_test 64/foo.o -lelf
	64/elf_read_test /proc/self/exe sym

32/execute.o: execute.cpp
	$(CXX32) $(CXXFLAGS) -c $< -o $@
64/execute.o: execute.cpp
	$(CXX64) $(CXXFLAGS) -c $< -o $@

clean:
	rm -rf 32/*.o 64/*.o 32/*.a 64/*.a \
	32/elf_write_test 32/elf_read_test 64/elf_write_test 64/elf_read_test

