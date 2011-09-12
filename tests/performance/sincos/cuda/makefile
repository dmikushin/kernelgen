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

LEVEL = ../../../..

include ../../../makefile.in

NVCC32 = nvcc -arch=sm_20 --ptxas-options="-v" -I/opt/kgen/include -L/opt/kgen/lib -lkernelgen
NVCC64 = nvcc -arch=sm_20 --ptxas-options="-v" -I/opt/kgen/include64 -L/opt/kgen/lib64 -lkernelgen

ifeq ($(ARCH), 64)
NVCC32 += -m32
all: 32/sincos 64/sincos
else
all: 32/sincos
endif

32/sincos: sincos.cu
	$(NVCC32) -g -O$(OPT) $< -o $@

64/sincos: sincos.cu
	$(NVCC64) -g -O$(OPT) $< -o $@

test32:
	@echo "CUDA 32-bit"
	32/sincos $(BLOCKS) $(THREADS) 0

test64:
	@echo "CUDA 64-bit"
	64/sincos $(BLOCKS) $(THREADS) 0

clean:
	rm -rf 32/sincos 64/sincos

