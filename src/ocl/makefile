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

include ../makefile.in

NUM_SYMBOLS ?= 1

INCFFI = -I/usr/lib64/libffi-3.0.9/include

all: kgen-ocl

kgen-ocl: ocl.c elf_write.o
	$(CC) -g -std=c99 -D_GNU_SOURCE -I. $< -o $@ elf_write.o -lelf $(OPENCL_LIBPATH) -lOpenCL

elf_write.o: elf_write.c elf_write.h
	$(CC) -g -std=c99 -c $< -o $@

elf_write_test: elf_write_test.c elf_write.o
	$(CC) -g -std=c99 $(INCFFI) $< -o $@ elf_write.o -lelf -lffi

test: elf_read_test.c elf_write_test
	./elf_write_test foo.o $(NUM_SYMBOLS) sym hello
	$(CC) -g -std=c99 $< -o elf_read_test DCT8x8.o -lelf
	./elf_read_test /proc/self/exe sym

clean:
	rm -rf kgen-ocl elf_write_test elf_read_test *.o
