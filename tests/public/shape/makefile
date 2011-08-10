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

ifeq ($(ARCH), 64)
all: 32/shape 64/shape
else
all: 32/shape
endif

32/shape: 32/shapes.o 32/sincos.o 32/main.o
	$(FCOMP32) 32/main.o 32/shapes.o 32/sincos.o -o $@
64/shape: 64/shapes.o 64/sincos.o 64/main.o
	$(FCOMP64) 64/main.o 64/shapes.o 64/sincos.o -o $@

32/main.o: main.c
	$(CCOMP32) -g -std=c99 -I/opt/kgen/include -c $< -o $@
64/main.o: main.c
	$(CCOMP64) -g -std=c99 -I/opt/kgen/include64 -c $< -o $@

32/shapes.o: shapes.f90
	$(FCOMP32) -J32 -g -c $< -o $@
64/shapes.o: shapes.f90
	$(FCOMP64) -J64 -g -c $< -o $@

32/sincos.o: sincos.F90 32/shapes.o
	$(FCOMP32) -I32 -g -c $< -o $@
64/sincos.o: sincos.F90 64/shapes.o
	$(FCOMP64) -I64 -g -c $< -o $@

clean:
	rm -rf 32/*.o 32/*.mod 32/shape 64/*.o 64/*.mod 64/shape \
	*.bc* *.o *.mod *_gforscale.* *.host.*
