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

TESTS = \
	behavior/szlogical/szlogical \
	behavior/axpy/axpy \
	behavior/logexp/logexp \
	behavior/allocatable/allocatable \
	behavior/module/module \
	behavior/mixed/mixed \
	behavior/multidim/multidim \
	behavior/sincos/sincos \
	behavior/shape/shape \
	behavior/derived/derived \
	behavior/inline/inline

all: $(TESTS)

ifeq ($(ARCH), 64)
tests: tests32 tests64
else
tests: tests32
endif

tests32: $(TESTS)
	./runtests 32
tests64: $(TESTS)
	./runtests 64

behavior/szlogical/szlogical:
	cd behavior/szlogical && $(MAKE)

behavior/axpy/axpy:
	cd behavior/axpy && $(MAKE)

behavior/logexp/logexp:
	cd behavior/logexp && $(MAKE)

behavior/allocatable/allocatable:
	cd behavior/allocatable && $(MAKE)

behavior/module/module:
	cd behavior/module && $(MAKE)

behavior/mixed/mixed:
	cd behavior/mixed && $(MAKE)

behavior/multidim/multidim:
	cd behavior/multidim && $(MAKE)

behavior/sincos/sincos:
	cd behavior/sincos && $(MAKE)

behavior/shape/shape:
	cd behavior/shape && $(MAKE)

behavior/derived/derived:
	cd behavior/derived && $(MAKE)

behavior/inline/inline:
	cd behavior/inline && $(MAKE)

clean:
	cd behavior/szlogical && $(MAKE) clean
	cd behavior/axpy && $(MAKE) clean
	cd behavior/logexp && $(MAKE) clean
	cd behavior/allocatable && $(MAKE) clean
	cd behavior/module && $(MAKE) clean
	cd behavior/mixed && $(MAKE) clean
	cd behavior/multidim && $(MAKE) clean
	cd behavior/sincos && $(MAKE) clean
	cd behavior/shape && $(MAKE) clean
	cd behavior/derived && $(MAKE) clean
	cd behavior/inline && $(MAKE) clean

