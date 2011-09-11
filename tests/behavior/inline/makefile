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
all: 32/inline 64/inline
else
all: 32/inline
endif

32/inline: 32/inline.o 32/main.o
	$(FCOMP32) 32/main.o 32/inline.o -o $@
64/inline: 64/inline.o 64/main.o
	$(FCOMP64) 64/main.o 64/inline.o -o $@

32/main.o: main.c
	$(CCOMP32) -g -std=c99 -I/opt/kgen/include -c $< -o $@
64/main.o: main.c
	$(CCOMP64) -g -std=c99 -I/opt/kgen/include64 -c $< -o $@

32/inline.o: inline.f90
	$(FCOMP32) -g -c $< -o $@
64/inline.o: inline.f90
	$(FCOMP64) -g -c $< -o $@

clean:
	rm -rf 32/*.o 32/*.mod 32/inline 64/*.o 64/*.mod 64/inline \
	*.bc* *.o *.mod *_kernelgen.* *.host.*

