all: sincos

sincos: sincos.o main.o
	kernelgen-gfortran sincos.o main.o -o $@

main.o: ../main.c
	kernelgen-gcc -std=c99 -c $< -o $@

sincos.o: ../sincos.f90
	kernelgen-gfortran -c $< -o $@

clean:
	rm -rf *.o sincos

