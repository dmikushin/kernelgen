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

#
# Red Hat notation, could be different, for example, for Debian
#
LIB32 ?= lib
LIB64 ?= lib64

PREFIX = $(ROOT)/opt/kernelgen
CUDA_PREFIX = undefined
ifeq ($(HAVE_CUDA), 1)
CUDA_PREFIX = /opt/cuda
endif

.PHONY: frontend libasfermi

all: frontend libasfermi runtime/32/libruntime.a runtime/64/libruntime.a

frontend: runtime/32/libruntime.a runtime/64/libruntime.a libasfermi
	cd frontend && $(MAKE)

libasfermi:
	cd libasfermi && $(MAKE)

runtime/32/libruntime.a:
	cd runtime && $(MAKE)

runtime/64/libruntime.a:
	cd runtime && $(MAKE)

clean:
	cd frontend && $(MAKE) clean
	cd runtime && $(MAKE) clean
	cd libasfermi && $(MAKE) clean

install: frontend runtime include/kernelgen_runtime.h
	perl install $(ARCH) $(PREFIX) $(LIB32) $(LIB64) $(CUDA_PREFIX)

