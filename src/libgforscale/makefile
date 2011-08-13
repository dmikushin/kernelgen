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

CFLAGS = -g -fPIC -std=c99
CXXFLAGS = -g -fPIC

FC = gfortran
DFC = dragonegg-gfortran
FFLAGS = -g -fPIC

ifeq ($(HAVE_CUDA), 1)
CC += -DHAVE_CUDA
CXX += -DHAVE_CUDA
NVCC += -DHAVE_CUDA
NVFLAGS = -g -Xcompiler -fPIC -Xcompiler -std=c99
else
NVCC = $(CXX)
NVFLAGS = $(CXXFLAGS) -malign-double
endif

INCFFI = -I/usr/lib64/libffi-3.0.9/include -I/usr/include/i686-linux-gnu

OBJECTS = \
	init launch compare error timing merge parse_args \
	parse_modsyms_cpu parse_modsyms_cuda parse_modsyms_opencl \
	map_cpu unmap_cpu map_cuda unmap_cuda map_opencl unmap_opencl \
	build_cpu build_cuda build_opencl launch_cpu launch_cuda launch_opencl \
	reset_cpu reset_cuda reset_opencl devaddr_opencl \
	elf_read elf_write load_source kernelgen

OBJECTS32 := $(addsuffix .o, $(addprefix 32/, $(OBJECTS)))
OBJECTS64 := $(addsuffix .o, $(addprefix 64/, $(OBJECTS)))

ifeq ($(ARCH), 32)
CC32 = $(CC)
CXX32 = $(CXX)
FC32 = $(FC) -J32
DFC32 = $(DFC) -J32
NVCC32 = $(NVCC)
CC64 = $(CC)
CXX64 = $(CXX)
FC64 = $(FC) -J64
DFC64 = $(DFC) -J64
NVCC64 = $(NVCC)
all: 32/libkernelgen.so 32/merge_test
else
CC32 = $(CC) -m32
CXX32 = $(CXX) -m32
FC32 = $(FC) -m32 -J32
DFC32 = $(DFC) -m32 -J32
NVCC32 = $(NVCC) -m32
CC64 = $(CC)
CXX64 = $(CXX)
FC64 = $(FC) -J64
DFC64 = $(DFC) -J64
NVCC64 = $(NVCC)
all: 32/libkernelgen.so 32/merge_test 64/libkernelgen.so 64/merge_test
endif

ifeq ($(HAVE_OPENCL), 1)
CC32 += -DHAVE_OPENCL
CXX32 += -DHAVE_OPENCL
NVCC32 += -DHAVE_OPENCL
endif

ifeq ($(HAVE_OPENCL), 1)
32/libkernelgen.so: $(OBJECTS32)
	$(NVCC32) -shared -o $@ $(OBJECTS32) -lelf -lffi -lrt -lOpenCL
else
32/libkernelgen.so: $(OBJECTS32)
	$(NVCC32) -shared -o $@ $(OBJECTS32) -lelf -lffi -lrt
endif
64/libkernelgen.so: $(OBJECTS64)
	$(NVCC64) -shared -o $@ $(OBJECTS64) -lelf -lffi -lrt

32/init.o: init.cpp init.h kernelgen.h kernelgen_int.h
	$(NVCC32) $(NVFLAGS) -c $< -o $@
64/init.o: init.cpp init.h kernelgen.h kernelgen_int.h
	$(NVCC64) $(NVFLAGS) -c $< -o $@

32/launch.o: launch.c kernelgen.h kernelgen_int.h
	$(NVCC32) $(NVFLAGS) -D_GNU_SOURCE -c $< -o $@
64/launch.o: launch.c kernelgen.h kernelgen_int.h
	$(NVCC64) $(NVFLAGS) -D_GNU_SOURCE -c $< -o $@

32/compare.o: compare.c kernelgen.h kernelgen_int.h
	$(CC32) $(CFLAGS) $(INCFFI) -c $< -o $@
64/compare.o: compare.c kernelgen.h kernelgen_int.h
	$(CC64) $(CFLAGS) $(INCFFI) -c $< -o $@

32/error.o: error.cpp kernelgen.h
	$(NVCC32) $(NVFLAGS) -c $< -o $@
64/error.o: error.cpp kernelgen.h
	$(NVCC64) $(NVFLAGS) -c $< -o $@

32/timing.o: timing.c kernelgen.h
	$(CC32) $(CFLAGS) -D_GNU_SOURCE -c $< -o $@
64/timing.o: timing.c kernelgen.h
	$(CC64) $(CFLAGS) -D_GNU_SOURCE -c $< -o $@

32/merge.o: merge.cpp kernelgen.h kernelgen_int.h
	$(CXX32) $(CXXFLAGS) -c $< -o $@
64/merge.o: merge.cpp kernelgen.h kernelgen_int.h
	$(CXX64) $(CXXFLAGS) -c $< -o $@

32/merge_test: merge_test.c kernelgen_int.h 32/libkernelgen.so
	$(CC32) $(CFLAGS) $< -o $@ -L32 -lkernelgen
64/merge_test: merge_test.c kernelgen_int.h 64/libkernelgen.so
	$(CC64) $(CFLAGS) $< -o $@ -L64 -lkernelgen

32/parse_args.o: parse_args.c parse_args.h
	$(CC32) $(CFLAGS) -c $< -o $@
64/parse_args.o: parse_args.c parse_args.h
	$(CC64) $(CFLAGS) -c $< -o $@

32/parse_modsyms_cpu.o: parse_modsyms_cpu.c
	$(CC32) $(CFLAGS) -c $< -o $@
64/parse_modsyms_cpu.o: parse_modsyms_cpu.c
	$(CC64) $(CFLAGS) -c $< -o $@

32/parse_modsyms_cuda.o: parse_modsyms_cuda.c parse_modsyms_cuda.h
	$(NVCC32) $(NVFLAGS) -c $< -o $@
64/parse_modsyms_cuda.o: parse_modsyms_cuda.c parse_modsyms_cuda.h
	$(NVCC64) $(NVFLAGS) -c $< -o $@

32/parse_modsyms_opencl.o: parse_modsyms_opencl.c
	$(CC32) $(CFLAGS) -c $< -o $@
64/parse_modsyms_opencl.o: parse_modsyms_opencl.c
	$(CC64) $(CFLAGS) -c $< -o $@

32/map_cpu.o: map_cpu.c
	$(CC32) $(CFLAGS) -c $< -o $@
64/map_cpu.o: map_cpu.c
	$(CC64) $(CFLAGS) -c $< -o $@

32/map_cuda.o: map_cuda.c map_cuda.h
	$(NVCC32) $(NVFLAGS) -c $< -o $@
64/map_cuda.o: map_cuda.c map_cuda.h
	$(NVCC64) $(NVFLAGS) -c $< -o $@

32/map_opencl.o: map_opencl.c
	$(CC32) $(CFLAGS) -c $< -o $@
64/map_opencl.o: map_opencl.c
	$(CC64) $(CFLAGS) -c $< -o $@

32/unmap_cpu.o: unmap_cpu.c
	$(CC32) $(FLAGS) -c $< -o $@
64/unmap_cpu.o: unmap_cpu.c
	$(CC64) $(FLAGS) -c $< -o $@

32/unmap_cuda.o: unmap_cuda.c unmap_cuda.h
	$(NVCC32) $(NVFLAGS) -c $< -o $@
64/unmap_cuda.o: unmap_cuda.c unmap_cuda.h
	$(NVCC64) $(NVFLAGS) -c $< -o $@

32/unmap_opencl.o: unmap_opencl.c
	$(CC32) $(CFLAGS) -c $< -o $@
64/unmap_opencl.o: unmap_opencl.c
	$(CC64) $(CFLAGS) -c $< -o $@

32/build_cpu.o: build_cpu.c
	$(CC32) $(CFLAGS) -c $< -o $@
64/build_cpu.o: build_cpu.c
	$(CC64) $(CFLAGS) -c $< -o $@

32/build_cuda.o: build_cuda.c
	$(NVCC32) $(NVFLAGS) -c $< -o $@
64/build_cuda.o: build_cuda.c
	$(NVCC64) $(NVFLAGS) -c $< -o $@

32/build_opencl.o: build_opencl.c
	$(CC32) $(CFLAGS) -c $< -o $@
64/build_opencl.o: build_opencl.c
	$(CC64) $(CFLAGS) -c $< -o $@

32/launch_cpu.o: launch_cpu.c
	$(CC32) $(CFLAGS) -c $< -o $@
64/launch_cpu.o: launch_cpu.c
	$(CC64) $(CFLAGS) -c $< -o $@

32/launch_cuda.o: launch_cuda.c
	$(NVCC32) $(NVFLAGS) -c $< -o $@
64/launch_cuda.o: launch_cuda.c
	$(NVCC64) $(NVFLAGS) -c $< -o $@

32/launch_opencl.o: launch_opencl.c
	$(CC32) $(CFLAGS) -c $< -o $@
64/launch_opencl.o: launch_opencl.c
	$(CC64) $(CFLAGS) -c $< -o $@

32/reset_cpu.o: reset_cpu.c
	$(CC32) $(CFLAGS) -c $< -o $@
64/reset_cpu.o: reset_cpu.c
	$(CC64) $(CFLAGS) -c $< -o $@

32/reset_cuda.o: reset_cuda.c
	$(NVCC32) $(NVFLAGS) -c $< -o $@
64/reset_cuda.o: reset_cuda.c
	$(NVCC64) $(NVFLAGS) -c $< -o $@

32/reset_opencl.o: reset_opencl.c
	$(CC32) $(CFLAGS) -c $< -o $@
64/reset_opencl.o: reset_opencl.c
	$(CC64) $(CFLAGS) -c $< -o $@

32/devaddr_opencl.o: devaddr_opencl.c
	$(CC32) $(CFLAGS) -c $< -o $@
64/devaddr_opencl.o: devaddr_opencl.c
	$(CC64) $(CFLAGS) -c $< -o $@

32/kernelgen.o: kernelgen.f90
	$(DFC32) $(FFLAGS) -c $< -o $@
	$(FC32) $(FFLAGS) -c $< -o $@
64/kernelgen.o: kernelgen.f90
	$(DFC64) $(FFLAGS) -c $< -o $@
	$(FC64) $(FFLAGS) -c $< -o $@

32/elf_read.o: elf_read.c
	$(CC32) $(CFLAGS) -c $< -o $@
64/elf_read.o: elf_read.c
	$(CC64) $(CFLAGS) -c $< -o $@

32/elf_write.o: elf_write.c
	$(CC32) $(CFLAGS) -c $< -o $@
64/elf_write.o: elf_write.c
	$(CC64) $(CFLAGS) -c $< -o $@

32/elf_write_test: elf_write_test.c elf_write.o
	$(CC32) $(CFLAGS) $(INCFFI) $< -o $@ -L32 -lkernelgen
64/elf_write_test: elf_write_test.c elf_write.o
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

32/load_source.o: load_source.c
	$(CC32) $(CFLAGS) -c $< -o $@
64/load_source.o: load_source.c
	$(CC64) $(CFLAGS) -c $< -o $@

clean:
	rm -rf 32/*.o 64/*.o 32/*.so 64/*.so 32/*.mod 64/*.mod 32/merge_test 64/merge_test \
	32/elf_write_test 32/elf_read_test 64/elf_write_test 64/elf_read_test

