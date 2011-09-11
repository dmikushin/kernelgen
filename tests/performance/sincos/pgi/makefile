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

SERFCOMP = pgfortran
SERCCOMP = gcc
PARFCOMP32 = pgfortran -Mnomain -Minfo=accel -ta=nvidia -m32 -I/opt/kgen/include -L/opt/kgen/lib -lkernelgen
PARFCOMP64 = pgfortran -Mnomain -Minfo=accel -ta=nvidia -I/opt/kgen/include64 -L/opt/kgen/lib64 -lkernelgen

include ../makefile.in

test32:
	@echo "PGI Accelerator 32-bit"
	32/sincos $(DIMS)

test64:
	@echo "PGI Accelerator 64-bit"
	64/sincos $(DIMS)
