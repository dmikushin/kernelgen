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

CFLAGS = -g -fPIC -std=c99 -O$(OPT) -I../util
CXXFLAGS = -g -fPIC -O$(OPT) -I../util

FC = gfortran
KFC = kernelgen-gfortran
FFLAGS = -g -fPIC -O$(OPT)

ifeq ($(HAVE_CUDA), 1)
CC += -DHAVE_CUDA
CXX += -DHAVE_CUDA
NVCC += -DHAVE_CUDA
NVFLAGS = -g -Xcompiler -fPIC -Xcompiler -std=c99 -I../util
else
NVCC = $(CXX)
NVFLAGS = $(CXXFLAGS) -malign-double
endif

INCFFI = -I/usr/lib64/libffi-3.0.9/include -I/usr/include/x86_64-linux-gnu -I/usr/include/i686-linux-gnu

OBJECTS = \
	init launch compare stats error merge parse_args \
	parse_modsyms_cpu parse_modsyms_cuda parse_modsyms_opencl \
	map_cpu unmap_cpu map_cuda unmap_cuda map_opencl unmap_opencl \
	build_cpu build_cuda build_opencl launch_cpu launch_cuda launch_opencl \
	reset_cpu reset_cuda reset_opencl devaddr_opencl kernelgen

OBJECTS32 := $(addsuffix .o, $(addprefix 32/, $(OBJECTS)))
OBJECTS64 := $(addsuffix .o, $(addprefix 64/, $(OBJECTS)))

ifeq ($(ARCH), 32)
CC32 = $(CC)
CXX32 = $(CXX)
FC32 = $(FC) -J32
KFC32 = $(KFC) -J32
NVCC32 = $(NVCC)
LIB32 = -L../util/32 -lutil
CC64 = $(CC)
CXX64 = $(CXX)
FC64 = $(FC) -J64
KFC64 = $(KFC) -J64
NVCC64 = $(NVCC)
all: 32/libkernelgen.so 32/merge_test
else
CC32 = $(CC) -m32
CXX32 = $(CXX) -m32
FC32 = $(FC) -m32 -J32
KFC32 = $(KFC) -m32 -J32
NVCC32 = $(NVCC) -m32
LIB32 = -L../util/32 -lutil
CC64 = $(CC)
CXX64 = $(CXX)
FC64 = $(FC) -J64
KFC64 = $(KFC) -J64
NVCC64 = $(NVCC)
LIB64 = -L../util/64 -lutil
all: 32/libkernelgen.so 32/merge_test 64/libkernelgen.so 64/merge_test
endif

ifeq ($(HAVE_OPENCL), 1)
CC32 += -DHAVE_OPENCL
CXX32 += -DHAVE_OPENCL
NVCC32 += -DHAVE_OPENCL
CC64 += -DHAVE_OPENCL
CXX64 += -DHAVE_OPENCL
NVCC64 += -DHAVE_OPENCL
endif

ifeq ($(HAVE_OPENCL), 1)
32/libkernelgen.so: $(OBJECTS32)
	$(NVCC32) -shared -o $@ $(OBJECTS32) -lelf -lffi $(OPENCL_LIBPATH32) -lOpenCL $(LIB32) -lrt
64/libkernelgen.so: $(OBJECTS64)
	$(NVCC64) -shared -o $@ $(OBJECTS64) -lelf -lffi $(OPENCL_LIBPATH64) -lOpenCL $(LIB64) -lrt
else
32/libkernelgen.so: $(OBJECTS32)
	$(NVCC32) -shared -o $@ $(OBJECTS32) -lelf -lffi $(LIB32) -lrt
64/libkernelgen.so: $(OBJECTS64)
	$(NVCC64) -shared -o $@ $(OBJECTS64) -lelf -lffi $(LIB64) -lrt
endif

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

32/stats.o: stats.cpp kernelgen.h kernelgen_int.h
	$(CXX32) $(CXXFLAGS) $(INCFFI) -c $< -o $@
64/stats.o: stats.cpp kernelgen.h kernelgen_int.h
	$(CXX64) $(CXXFLAGS) $(INCFFI) -c $< -o $@

32/error.o: error.cpp kernelgen.h
	$(NVCC32) $(NVFLAGS) -c $< -o $@
64/error.o: error.cpp kernelgen.h
	$(NVCC64) $(NVFLAGS) -c $< -o $@

32/merge.o: merge.cpp kernelgen.h kernelgen_int.h
	$(CXX32) $(CXXFLAGS) -c $< -o $@
64/merge.o: merge.cpp kernelgen.h kernelgen_int.h
	$(CXX64) $(CXXFLAGS) -c $< -o $@

32/merge_test: merge_test.c kernelgen_int.h 32/libkernelgen.so
	$(CC32) $(CFLAGS) $< -o $@ -L32 -lkernelgen $(LIB32)
64/merge_test: merge_test.c kernelgen_int.h 64/libkernelgen.so
	$(CC64) $(CFLAGS) $< -o $@ -L64 -lkernelgen $(LIB64)

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
	$(KFC32) $(FFLAGS) -c $< -o $@
	$(FC32) $(FFLAGS) -c $< -o $@
64/kernelgen.o: kernelgen.f90
	$(KFC64) $(FFLAGS) -c $< -o $@
	$(FC64) $(FFLAGS) -c $< -o $@

clean:
	rm -rf 32/*.o 64/*.o 32/*.so 64/*.so 32/*.mod 64/*.mod 32/merge_test 64/merge_test

