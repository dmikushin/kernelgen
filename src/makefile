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

KGEN_PREFIX = /opt/kgen
KGEN_BUILD_PREFIX = $(PREFIX)$(KGEN_PREFIX)
LLVM_BIN_PREFIX = $(KGEN_PREFIX)/bin/
GCC_BIN_PREFIX = $(KGEN_PREFIX)/bin/dragonegg-
ifeq ($(ARCH), 64)
DRAGONEGG_PREFIX = $(KGEN_PREFIX)/lib${ARCH}/
else
DRAGONEGG_PREFIX = $(KGEN_PREFIX)/lib/
endif
CUDA_PREFIX = undefined
ifeq ($(HAVE_CUDA), 1)
CUDA_PREFIX = /opt/cuda
endif

.PHONY: kgen-kernelgen g95xml-tree g95xml-refids libkernelgen kgen-ocl

all: kgen-kernelgen libkernelgen kgen-ocl

kgen-kernelgen: g95xml-tree g95xml-refids
	
g95xml-tree:
	cd g95xml_tree && $(MAKE)

g95xml-refids:
	cd g95xml_refids && $(MAKE)

libkernelgen:
	cd libkernelgen && $(MAKE)

kgen-ocl: libkernelgen
	cd ocl && $(MAKE)

clean:
	cd g95xml_tree && $(MAKE) clean
	cd g95xml_refids && $(MAKE) clean
	cd libkernelgen && $(MAKE) clean
	cd ocl && $(MAKE) clean

install: kgen-kernelgen libkernelgen kgen-ocl
	perl kgen-install $(ARCH) \
		$(KGEN_BUILD_PREFIX) $(KGEN_PREFIX) $(LLVM_BIN_PREFIX) $(GCC_BIN_PREFIX) \
		$(DRAGONEGG_PREFIX) $(CUDA_PREFIX)

