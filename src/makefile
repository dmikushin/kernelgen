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

#
# Red Hat notation, could be different, for example, for Debian
#
LIB32 ?= lib
LIB64 ?= lib

#
# Optimization level
#
OPT ?= 0

RPATH ?= /opt/kernelgen
PREFIX = $(ROOT)/$(RPATH)

.PHONY: frontend runtime asfermi bindings

all: frontend runtime

frontend:
	cd frontend && $(MAKE)

asfermi: bindings
	cd libasfermi && $(MAKE)

runtime: asfermi bindings
	cd runtime && $(MAKE) RPATH=$(RPATH)

bindings:
	cd bindings && $(MAKE)

clean:
	cd frontend && $(MAKE) clean
	cd runtime && $(MAKE) clean
	cd libasfermi && $(MAKE) clean
	cd bindings && $(MAKE) clean

install: frontend runtime
	perl install $(PREFIX) $(LIB32) $(LIB64)

