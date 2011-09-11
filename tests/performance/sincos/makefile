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

DIMS = 512 512 64

all: kgen/sincos pgi/sincos

kgen/sincos:
	cd kgen && $(MAKE)

pgi/sincos:
	cd pgi && $(MAKE)

ifeq ($(ARCH), 64)
test: test32 test64
else
test: test32
endif

test32: kgen/sincos pgi/sincos
	cd kgen && $(MAKE) test32 DIMS="$(DIMS)"
	cd pgi && $(MAKE) test32 DIMS="$(DIMS)"

test64: kgen/sincos pgi/sincos
	cd kgen && $(MAKE) test64 DIMS="$(DIMS)"
	cd pgi && $(MAKE) test64 DIMS="$(DIMS)"

clean:
	cd kgen && $(MAKE) clean
	cd pgi && $(MAKE) clean

