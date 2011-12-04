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

LEVEL = ../../..

include ../../makefile.in

ifeq ($(ARCH), 64)
all: 32/hello 64/hello
else
all: 32/hello
endif

32/hello: 32/main.o
	$(FCOMP32) 32/main.o -o $@
64/hello: 64/main.o
	$(FCOMP64) 64/main.o -o $@

32/main.o: main.f90
	$(FCOMP32) -g -c $< -o $@
64/main.o: main.f90
	$(FCOMP64) -g -c $< -o $@

clean:
	rm -rf 32/*.o 32/*.mod 32/hello 64/*.o 64/*.mod 64/hello \
	*.bc* *.o *.mod *_kernelgen.* *.host.*
