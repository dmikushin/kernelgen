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

include ../../../src/makefile.in

HAVE_CPU ?= 1

CCOMP ?= $(CC)
FCOMP ?= kgen-gfortran -Wk,--host-compiler=$(FC)

ifeq ($(HAVE_CPU), 1)
FCOMP += -Wk,--cpu-compiler=gcc
HAVE_CPU = cpu,
else
HAVE_CPU =
endif

HAVE_CUDA =
HAVE_OPENCL =

FCOMP += -Wk,--kernel-target=$(HAVE_CPU)$(HAVE_CUDA)$(HAVE_OPENCL)

all: szlogical

szlogical: main.o size.o
	$(FCOMP) main.o size.o -o szlogical

main.o: main.f90
	$(FCOMP) -g -c main.f90

size.o: size.c
	$(CCOMP) -g -c size.c

clean:
	rm -rf *.bc* *.o *.mod *_gforscale.* *.host.* *.device.* szlogical
