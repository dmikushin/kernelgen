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
	public/szlogical/szlogical \
	public/axpy/axpy \
	public/logexp/logexp \
	public/allocatable/allocatable \
	public/module/module \
	public/mixed/mixed \
	public/multidim/multidim \
	public/sincos/sincos \
	public/shape/shape \
	public/derived/derived \
	public/inline/inline

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

public/szlogical/szlogical:
	cd public/szlogical && $(MAKE)

public/axpy/axpy:
	cd public/axpy && $(MAKE)

public/logexp/logexp:
	cd public/logexp && $(MAKE)

public/allocatable/allocatable:
	cd public/allocatable && $(MAKE)

public/module/module:
	cd public/module && $(MAKE)

public/mixed/mixed:
	cd public/mixed && $(MAKE)

public/multidim/multidim:
	cd public/multidim && $(MAKE)

public/sincos/sincos:
	cd public/sincos && $(MAKE)

public/shape/shape:
	cd public/shape && $(MAKE)

public/derived/derived:
	cd public/derived && $(MAKE)

public/inline/inline:
	cd public/inline && $(MAKE)

clean:
	cd public/szlogical && $(MAKE) clean
	cd public/axpy && $(MAKE) clean
	cd public/logexp && $(MAKE) clean
	cd public/allocatable && $(MAKE) clean
	cd public/module && $(MAKE) clean
	cd public/mixed && $(MAKE) clean
	cd public/multidim && $(MAKE) clean
	cd public/sincos && $(MAKE) clean
	cd public/shape && $(MAKE) clean
	cd public/derived && $(MAKE) clean
	cd public/inline && $(MAKE) clean

