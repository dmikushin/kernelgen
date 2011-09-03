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

OPT = 3

include ../../makefile.in

ifeq ($(ARCH), 64)
all: 32/sincos 64/sincos
else
all: 32/sincos
endif

PGFCOMP = pgfortran -Mnomain -Minfo=accel -ta=nvidia
PGFCOMP32 = $(PGFCOMP) -I/opt/kgen/include -m32 -L/opt/kgen/lib64 -lkernelgen
PGFCOMP64 = $(PGFCOMP) -I/opt/kgen/include64 -L/opt/kgen/lib64 -lkernelgen

32/sincos: 32/sincos.o 32/main.o
	$(FCOMP32) 32/main.o 32/sincos.o -o $@
64/sincos: 64/sincos.o 64/main.o
	$(FCOMP64) 64/main.o 64/sincos.o -o $@

32/main.o: main.c
	$(CCOMP32) -O$(OPT) -g -std=c99 -I/opt/kgen/include -c $< -o $@
64/main.o: main.c
	$(CCOMP64) -O$(OPT) -g -std=c99 -I/opt/kgen/include64 -c $< -o $@

32/sincos.o: sincos.f90
	$(FCOMP32) -O$(OPT) -g -c $< -o $@
64/sincos.o: sincos.f90
	$(FCOMP64) -O$(OPT) -g -c $< -o $@

pgi/32/sincos: pgi/32/sincos.o pgi/32/main.o
	$(PGFCOMP32) pgi/32/main.o pgi/32/sincos.o -o $@
pgi/64/sincos: pgi/64/sincos.o pgi/64/main.o
	$(PGFCOMP64) pgi/64/main.o pgi/64/sincos.o -o $@

pgi/32/main.o: main.c
	$(PGFCOMP32) -O$(OPT) -g -c $< -o $@
pgi/64/main.o: main.c
	$(PGFCOMP64) -O$(OPT) -g -c $< -o $@

pgi/32/sincos.o: sincos.f90
	$(PGFCOMP32) -O$(OPT) -g -c $< -o $@
pgi/64/sincos.o: sincos.f90
	$(PGFCOMP64) -O$(OPT) -g -c $< -o $@

clean:
	rm -rf 32/*.o 32/*.mod 32/sincos 64/*.o 64/*.mod 64/sincos \
	rm -rf pgi/32/*.o pgi/32/*.mod pgi/32/sincos pgi/64/*.o pgi/64/*.mod pgi/64/sincos \
	*.bc* *.o *.mod *_kernelgen.* *.host.*
