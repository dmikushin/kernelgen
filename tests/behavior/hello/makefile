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

.SUFFIXES:
.SUFFIXES: .c.o .f90.o

all: hello_c hello_f

hello_c: main.c.o
	kernelgen-gcc $< -o $@

main.c.o: main.c
	kernelgen-gcc -g -c $< -o $@

hello_f: main.f90.o
	kernelgen-gfortran $< -o $@

main.f90.o: main.f90
	kernelgen-gfortran -g -c $< -o $@

clean:
	rm -rf *.o hello_c hello_f

