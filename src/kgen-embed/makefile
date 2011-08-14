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

KERNELGEN = ../libkernelgen

all: kgen-ir-embed kgen-cuda-embed kgen-opencl-embed

kgen-ir-embed: kgen-ir-embed.c
	$(CC) -g -std=c99 -D_GNU_SOURCE -I$(KERNELGEN) $< -o $@ -L$(KERNELGEN)/$(ARCH) -lkernelgen

kgen-cuda-embed: kgen-cuda-embed.c
	$(CC) -g -std=c99 $< -o $@

kgen-opencl-embed: kgen-opencl-embed.c
	$(CC) -g -std=c99 -D_GNU_SOURCE -I$(KERNELGEN) $< -o $@ -L$(KERNELGEN)/$(ARCH) -lkernelgen $(OPENCL_LIBPATH) -lOpenCL -lelf

clean:
	rm -rf kgen-ir-embed kgen-cuda-embed kgen-opencl-embed *.o
